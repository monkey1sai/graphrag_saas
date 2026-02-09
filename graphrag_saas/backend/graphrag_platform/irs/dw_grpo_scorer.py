"""DW-GRPO Scorer：動態權重 + CPU Offload

用於 IRS (Iterative Rejection Sampling) 的評分階段：
- r_rel (相關性)：Cross-Encoder reranker
- r_faith (忠實度)：Embedding cosine similarity (BERTScore 替代)
- r_conc (簡潔性)：長度比 heuristic

動態權重更新：
- 監測各指標的滑動窗口斜率
- 斜率越小（停滯）權重越大
- 使用 softmax 歸一化 + momentum 平滑更新
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 可選依賴：sentence-transformers / bert-score
try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    CrossEncoder = None  # type: ignore
    SentenceTransformer = None  # type: ignore

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class ScoredCandidate:
    """單一候選答案的評分結果"""

    candidate: str
    r_rel: float
    r_faith: float
    r_conc: float
    weighted_score: float
    rank: int = 0


@dataclass
class SelectionResult:
    """Best-of-N 選擇結果"""

    query: str
    context: str
    best: ScoredCandidate
    all_candidates: list[ScoredCandidate]
    current_weights: dict[str, float]
    avg_rewards: dict[str, float]


@dataclass
class DWGRPOConfig:
    """DW-GRPO 評分器配置"""

    device: str = "cpu"
    history_window: int = 5
    temperature: float = 2.0
    momentum: float = 0.8
    initial_weights: dict[str, float] = field(
        default_factory=lambda: {"rel": 1.0, "faith": 1.0, "conc": 0.5}
    )
    # 模型 ID
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    encoder_model: str = "BAAI/bge-m3"


class DWGRPOScorer:
    """DW-GRPO 動態權重評分器（CPU Offload 版）

    適用於 8GB VRAM 環境：
    - Cross-Encoder reranker 在 CPU 執行
    - Embedding model 在 CPU 執行
    - 利用 64GB RAM 優勢
    """

    def __init__(self, config: DWGRPOConfig | None = None) -> None:
        self.config = config or DWGRPOConfig(
            device=os.getenv("REWARD_DEVICE", "cpu")
        )

        # 權重與歷史
        self.weights = dict(self.config.initial_weights)
        self.history: dict[str, list[float]] = {k: [] for k in self.weights}

        # 延遲載入模型
        self._reranker: CrossEncoder | None = None
        self._encoder: SentenceTransformer | None = None
        self._models_loaded = False

    def _ensure_models_loaded(self) -> None:
        """延遲載入 CPU Reward Models"""
        if self._models_loaded:
            return

        if not ST_AVAILABLE:
            raise ImportError(
                "DWGRPOScorer requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            )

        device = self.config.device
        print(f"🐢 Loading Reward Models on {device}...")

        # Cross-Encoder for r_rel
        self._reranker = CrossEncoder(self.config.reranker_model, device=device)

        # SentenceTransformer for r_faith
        self._encoder = SentenceTransformer(self.config.encoder_model, device=device)

        self._models_loaded = True
        print("✅ Reward Models loaded successfully.")

    def compute_rewards(
        self, query: str, context: str, candidates: list[str]
    ) -> list[dict[str, float]]:
        """計算每個候選答案的三項原始分數"""
        self._ensure_models_loaded()

        rewards_list: list[dict[str, float]] = []

        for cand in candidates:
            # r_rel: 相關性（Cross-Encoder）
            r_rel = float(self._reranker.predict([(query, cand)])[0])  # type: ignore

            # r_faith: 忠實度（Embedding Cosine Similarity）
            # 使用 embedding 相似度作為 BERTScore 快速替代
            cand_emb = self._encoder.encode(  # type: ignore
                cand, convert_to_tensor=True, show_progress_bar=False
            )
            ctx_emb = self._encoder.encode(  # type: ignore
                context, convert_to_tensor=True, show_progress_bar=False
            )

            if TORCH_AVAILABLE and torch is not None:
                r_faith = float(
                    torch.cosine_similarity(cand_emb, ctx_emb, dim=0).item()
                )
            else:
                # Fallback: numpy cosine similarity
                cand_np = np.array(cand_emb)
                ctx_np = np.array(ctx_emb)
                r_faith = float(
                    np.dot(cand_np, ctx_np)
                    / (np.linalg.norm(cand_np) * np.linalg.norm(ctx_np) + 1e-8)
                )

            # r_conc: 簡潔性（長度懲罰）
            # r_conc = max(0, 1 - len(C)/len(K))
            r_conc = max(0.0, 1.0 - (len(cand) / (len(context) + 1e-5)))

            rewards_list.append({"rel": r_rel, "faith": r_faith, "conc": r_conc})

        return rewards_list

    def select_best(
        self, query: str, context: str, candidates: list[str]
    ) -> SelectionResult:
        """DW-GRPO 核心流程：計算分數 → 更新權重 → 選擇最佳"""
        # 1. 計算原始獎勵
        raw_rewards = self.compute_rewards(query, context, candidates)

        # 2. 計算加權總分
        scored: list[ScoredCandidate] = []
        for idx, (cand, r) in enumerate(zip(candidates, raw_rewards, strict=False)):
            weighted = (
                self.weights["rel"] * r["rel"]
                + self.weights["faith"] * r["faith"]
                + self.weights["conc"] * r["conc"]
            )
            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    r_rel=r["rel"],
                    r_faith=r["faith"],
                    r_conc=r["conc"],
                    weighted_score=weighted,
                    rank=0,
                )
            )

        # 3. 排序並選擇最佳
        scored.sort(key=lambda x: x.weighted_score, reverse=True)
        for rank, s in enumerate(scored):
            s.rank = rank + 1

        best = scored[0]

        # 4. 記錄歷史（使用平均值代表本輪表現）
        avg_rewards = {
            k: float(np.mean([r[k] for r in raw_rewards])) for k in self.weights
        }
        for k, v in avg_rewards.items():
            self.history[k].append(v)

        # 5. 動態權重更新
        self._update_weights()

        return SelectionResult(
            query=query,
            context=context,
            best=best,
            all_candidates=scored,
            current_weights=dict(self.weights),
            avg_rewards=avg_rewards,
        )

    def _update_weights(self) -> None:
        """根據歷史斜率調整權重：提升慢的指標權重增加"""
        window = self.config.history_window
        if len(self.history["rel"]) < window:
            return

        slopes: dict[str, float] = {}
        for k, vals in self.history.items():
            # 計算最近 N 輪的變化斜率（Linear Regression Slope）
            y = np.array(vals[-window:])
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[k] = slope

        # DW-GRPO 核心：斜率越小（stagnant），權重應越大
        # 使用 Softmax 歸一化反向斜率
        # w_j = exp(-slope_j / T) / sum(...)
        T = self.config.temperature
        exps = {k: np.exp(-v / T) for k, v in slopes.items()}
        total = sum(exps.values())

        # 乘 3 保持總權重約為 3（與初始權重總和一致）
        new_weights = {k: 3.0 * (v / total) for k, v in exps.items()}

        # Momentum 平滑更新
        alpha = self.config.momentum
        self.weights = {
            k: alpha * self.weights[k] + (1 - alpha) * new_weights[k]
            for k in self.weights
        }

    def get_state(self) -> dict[str, Any]:
        """取得當前狀態（用於持久化/回報）"""
        return {
            "weights": dict(self.weights),
            "history_length": {k: len(v) for k, v in self.history.items()},
            "models_loaded": self._models_loaded,
            "device": self.config.device,
        }

    def reset(self) -> None:
        """重置權重與歷史"""
        self.weights = dict(self.config.initial_weights)
        self.history = {k: [] for k in self.weights}


class HeuristicDWGRPOScorer:
    """輕量級 DW-GRPO Scorer（不需要 sentence-transformers）

    用於：
    - 快速測試
    - 無 ML 依賴環境
    - Fallback 場景
    """

    def __init__(self, config: DWGRPOConfig | None = None) -> None:
        self.config = config or DWGRPOConfig()
        self.weights = dict(self.config.initial_weights)
        self.history: dict[str, list[float]] = {k: [] for k in self.weights}

    def compute_rewards(
        self, query: str, context: str, candidates: list[str]
    ) -> list[dict[str, float]]:
        """使用 heuristic 計算獎勵（無 ML 模型）"""
        rewards_list: list[dict[str, float]] = []

        query_tokens = set(query.lower().split())
        context_tokens = set(context.lower().split())

        for cand in candidates:
            cand_tokens = set(cand.lower().split())

            # r_rel: 與 query 的 token overlap
            if query_tokens:
                r_rel = len(cand_tokens & query_tokens) / len(query_tokens)
            else:
                r_rel = 0.0

            # r_faith: 與 context 的 token overlap
            if context_tokens:
                r_faith = len(cand_tokens & context_tokens) / len(context_tokens)
            else:
                r_faith = 0.0

            # r_conc: 長度比
            r_conc = max(0.0, 1.0 - (len(cand) / (len(context) + 1e-5)))

            rewards_list.append({"rel": r_rel, "faith": r_faith, "conc": r_conc})

        return rewards_list

    def select_best(
        self, query: str, context: str, candidates: list[str]
    ) -> SelectionResult:
        """同 DWGRPOScorer 但使用 heuristic rewards"""
        raw_rewards = self.compute_rewards(query, context, candidates)

        scored: list[ScoredCandidate] = []
        for cand, r in zip(candidates, raw_rewards, strict=False):
            weighted = (
                self.weights["rel"] * r["rel"]
                + self.weights["faith"] * r["faith"]
                + self.weights["conc"] * r["conc"]
            )
            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    r_rel=r["rel"],
                    r_faith=r["faith"],
                    r_conc=r["conc"],
                    weighted_score=weighted,
                )
            )

        scored.sort(key=lambda x: x.weighted_score, reverse=True)
        for rank, s in enumerate(scored):
            s.rank = rank + 1

        avg_rewards = {
            k: float(np.mean([r[k] for r in raw_rewards])) for k in self.weights
        }
        for k, v in avg_rewards.items():
            self.history[k].append(v)

        self._update_weights()

        return SelectionResult(
            query=query,
            context=context,
            best=scored[0],
            all_candidates=scored,
            current_weights=dict(self.weights),
            avg_rewards=avg_rewards,
        )

    def _update_weights(self) -> None:
        """同 DWGRPOScorer 的權重更新邏輯"""
        window = self.config.history_window
        if len(self.history["rel"]) < window:
            return

        slopes = {}
        for k, vals in self.history.items():
            y = np.array(vals[-window:])
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            slopes[k] = slope

        T = self.config.temperature
        exps = {k: np.exp(-v / T) for k, v in slopes.items()}
        total = sum(exps.values())
        new_weights = {k: 3.0 * (v / total) for k, v in exps.items()}

        alpha = self.config.momentum
        self.weights = {
            k: alpha * self.weights[k] + (1 - alpha) * new_weights[k]
            for k in self.weights
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "history_length": {k: len(v) for k, v in self.history.items()},
            "type": "heuristic",
        }

    def reset(self) -> None:
        self.weights = dict(self.config.initial_weights)
        self.history = {k: [] for k in self.weights}


def get_default_scorer(config: DWGRPOConfig | None = None) -> DWGRPOScorer | HeuristicDWGRPOScorer:
    """取得預設 scorer（自動選擇 model-based 或 heuristic）"""
    import os

    mode = os.getenv("DWGRPO_SCORER", "").strip().lower()
    if mode in {"heuristic", "simple"}:
        return HeuristicDWGRPOScorer(config)
    if mode in {"model", "st", "sentence-transformers"}:
        if not ST_AVAILABLE:
            raise RuntimeError("DWGRPO_SCORER=model requested but sentence-transformers is not available.")
        return DWGRPOScorer(config)

    if ST_AVAILABLE:
        return DWGRPOScorer(config)
    else:
        print("⚠️ sentence-transformers not available, using HeuristicDWGRPOScorer")
        return HeuristicDWGRPOScorer(config)
