"""IRS Runner：Iterative Rejection Sampling 訓練迴圈

適用於 8GB VRAM 環境的 PPO 替代方案：
- Generate 階段：載入 4-bit Unsloth Model 生成 N 個候選答案
- Score 階段：卸載模型，在 CPU 使用 DW-GRPO 評分並選擇最佳
- SFT 階段：重新載入模型，用最佳答案微調（QLoRA）

每個階段只載入一個模型，避免 OOM。
"""

from __future__ import annotations

import contextlib
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .dw_grpo_scorer import DWGRPOConfig, get_default_scorer

# 可選依賴
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    FastLanguageModel = None  # type: ignore


@dataclass
class IRSConfig:
    """IRS 訓練配置"""

    # 模型配置
    base_model: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    # 8GB VRAM 目標下，2048 可能在 SFT phase 觸發額外記憶體需求（例如 fused CE）
    # 預設下調，smoke test 與 8GB 部署更穩定。
    max_seq_length: int = 1024
    load_in_4bit: bool = True

    # 生成配置
    n_candidates: int = 4  # Best-of-N
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    # 訓練配置
    n_iterations: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # DW-GRPO 配置
    dw_config: DWGRPOConfig = field(default_factory=DWGRPOConfig)

    # 輸出配置
    output_dir: str = "/app/data/jobs"
    job_id: str = ""


@dataclass
class IRSPhaseStatus:
    """單一 Phase 的狀態"""

    phase: str  # "generate" | "score" | "sft"
    progress: float  # 0.0 ~ 1.0
    samples_processed: int
    samples_total: int
    start_time: str
    elapsed_seconds: float = 0.0


@dataclass
class IRSIterationStatus:
    """單一 Iteration 的狀態"""

    iteration: int
    current_phase: IRSPhaseStatus
    reward_breakdown: dict[str, float]
    current_weights: dict[str, float]
    best_sample: str | None = None


class IRSRunner:
    """IRS 訓練迴圈執行器

    設計原則：
    1. 每個階段只載入一個模型
    2. 階段切換時使用 torch.cuda.empty_cache()
    3. 全部 Reward scoring 在 CPU 執行
    4. 使用 JSONL 記錄每個步驟
    """

    def __init__(
        self,
        config: IRSConfig,
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.status_callback = status_callback
        self.scorer = get_default_scorer(config.dw_config)

        # 確保輸出目錄存在
        self.job_dir = Path(config.output_dir) / f"train_{config.job_id}"
        self.job_dir.mkdir(parents=True, exist_ok=True)

        # JSONL 檔案路徑
        self.metrics_path = self.job_dir / "metrics.jsonl"
        self.samples_path = self.job_dir / "samples.jsonl"

        # 模型與 tokenizer（延遲載入）
        self._model = None
        self._tokenizer = None

    def _log_metrics(self, data: dict[str, Any]) -> None:
        """寫入 metrics.jsonl"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_sample(self, data: dict[str, Any]) -> None:
        """寫入 samples.jsonl"""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        with open(self.samples_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _update_status(self, status: IRSIterationStatus) -> None:
        """更新訓練狀態（透過 callback）"""
        if self.status_callback:
            self.status_callback(
                {
                    "iteration": status.iteration,
                    "phase": status.current_phase.phase,
                    "progress": status.current_phase.progress,
                    "reward_breakdown": status.reward_breakdown,
                    "current_weights": status.current_weights,
                }
            )

    def _load_model_for_generation(self) -> None:
        """載入模型用於生成（4-bit，不啟用 PEFT adapter）"""
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError(
                "Unsloth is not available. Please run in Unsloth container."
            )

        self._log_metrics({"event": "load_model_generation_start"})
        start = time.time()

        # Unsloth 的 FastLanguageModel 會自動處理 4-bit
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,  # Auto-detect
        )

        # 設定 inference mode
        FastLanguageModel.for_inference(self._model)

        elapsed = time.time() - start
        self._log_metrics({"event": "load_model_generation_done", "elapsed_s": elapsed})

    def _load_model_for_training(self) -> None:
        """載入模型用於 SFT（加上 LoRA adapter）"""
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError(
                "Unsloth is not available. Please run in Unsloth container."
            )

        self._log_metrics({"event": "load_model_training_start"})
        start = time.time()

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=None,
        )

        # 加上 LoRA adapter
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        elapsed = time.time() - start
        self._log_metrics({"event": "load_model_training_done", "elapsed_s": elapsed})

    def _unload_model(self) -> None:
        """卸載模型並清理 GPU 記憶體"""
        self._log_metrics({"event": "unload_model_start"})

        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None

        if TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
        gc.collect()

        self._log_metrics({"event": "unload_model_done"})

    def _generate_candidates(
        self, question: str, context: str, n: int
    ) -> list[str]:
        """使用模型生成 N 個候選答案"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model_for_generation first.")

        # 構建 prompt
        system_prompt = "You are a helpful AI assistant. Answer the question based on the given context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Tokenize
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if TORCH_AVAILABLE and torch is not None:
            inputs = inputs.to(self._model.device)

        # 生成 N 個候選
        candidates = []
        for _ in range(n):
            with torch.inference_mode() if TORCH_AVAILABLE else contextlib.nullcontext():
                outputs = self._model.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id
                    or self._tokenizer.eos_token_id,
                )

            # Decode 只取生成的部分
            generated = self._tokenizer.decode(
                outputs[0][inputs.shape[-1] :], skip_special_tokens=True
            )
            candidates.append(generated.strip())

        return candidates

    def _run_sft_step(self, samples: list[dict[str, str]]) -> dict[str, float]:
        """使用選中的最佳樣本執行 SFT step"""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model_for_training first.")

        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except ImportError:
            raise RuntimeError("TRL is required for SFT. Install with: pip install trl")

        # 構建訓練資料集
        from datasets import Dataset

        def format_sample(sample: dict[str, str]) -> str:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer the question based on the given context.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{sample['context']}\n\nQuestion: {sample['question']}\n\nAnswer:",
                },
                {"role": "assistant", "content": sample["answer"]},
            ]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

        formatted = [{"text": format_sample(s)} for s in samples]
        dataset = Dataset.from_list(formatted)

        # 訓練參數
        checkpoint_dir = self.job_dir / "checkpoints" / f"iter_{self._current_iteration}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 檢測 bfloat16 支援（CUDA 8.0+ 支援 bf16）
        use_bf16 = False
        use_fp16 = False
        if TORCH_AVAILABLE and torch.cuda.is_available():
            cuda_capability = torch.cuda.get_device_capability()
            if cuda_capability[0] >= 8:  # Ampere (RTX 30xx) 或更新
                use_bf16 = True
            else:
                use_fp16 = True

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="no",  # 手動保存
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self._model,
            tokenizer=self._tokenizer,
            train_dataset=dataset,
            args=training_args,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
        )

        # 訓練
        result = trainer.train()

        # 儲存 adapter
        self._model.save_pretrained(str(checkpoint_dir))
        self._tokenizer.save_pretrained(str(checkpoint_dir))

        return {
            "train_loss": result.training_loss,
            "train_samples": len(samples),
        }

    def run(self, questions: list[dict[str, str]]) -> dict[str, Any]:
        """執行完整 IRS 訓練迴圈

        Args:
            questions: 訓練問題列表，每個元素包含 {"question": str, "context": str}

        Returns:
            訓練摘要
        """
        self._log_metrics(
            {
                "event": "irs_run_start",
                "n_questions": len(questions),
                "n_iterations": self.config.n_iterations,
                "n_candidates": self.config.n_candidates,
            }
        )

        total_start = time.time()
        iteration_results = []

        for iteration in range(1, self.config.n_iterations + 1):
            self._current_iteration = iteration
            iter_start = time.time()
            self._log_metrics({"event": "iteration_start", "iteration": iteration})

            # ===== Phase 1: Generate =====
            phase_status = IRSPhaseStatus(
                phase="generate",
                progress=0.0,
                samples_processed=0,
                samples_total=len(questions),
                start_time=datetime.now(timezone.utc).isoformat(),
            )

            self._load_model_for_generation()
            generated_data: list[dict[str, Any]] = []

            for idx, q in enumerate(questions):
                candidates = self._generate_candidates(
                    question=q["question"],
                    context=q["context"],
                    n=self.config.n_candidates,
                )
                generated_data.append(
                    {
                        "question": q["question"],
                        "context": q["context"],
                        "candidates": candidates,
                    }
                )

                phase_status.samples_processed = idx + 1
                phase_status.progress = (idx + 1) / len(questions)
                self._update_status(
                    IRSIterationStatus(
                        iteration=iteration,
                        current_phase=phase_status,
                        reward_breakdown={},
                        current_weights=self.scorer.weights,
                    )
                )

            self._unload_model()

            # ===== Phase 2: Score =====
            phase_status = IRSPhaseStatus(
                phase="score",
                progress=0.0,
                samples_processed=0,
                samples_total=len(questions),
                start_time=datetime.now(timezone.utc).isoformat(),
            )

            selected_samples: list[dict[str, str]] = []
            total_rewards = {"rel": 0.0, "faith": 0.0, "conc": 0.0}

            for idx, gd in enumerate(generated_data):
                result = self.scorer.select_best(
                    query=gd["question"],
                    context=gd["context"],
                    candidates=gd["candidates"],
                )

                selected_samples.append(
                    {
                        "question": gd["question"],
                        "context": gd["context"],
                        "answer": result.best.candidate,
                    }
                )

                # 累計獎勵
                for k in total_rewards:
                    total_rewards[k] += result.avg_rewards[k]

                # 記錄樣本
                self._log_sample(
                    {
                        "iteration": iteration,
                        "question": gd["question"],
                        "best_answer": result.best.candidate,
                        "best_score": result.best.weighted_score,
                        "rewards": {
                            "rel": result.best.r_rel,
                            "faith": result.best.r_faith,
                            "conc": result.best.r_conc,
                        },
                        "weights": result.current_weights,
                    }
                )

                phase_status.samples_processed = idx + 1
                phase_status.progress = (idx + 1) / len(questions)
                self._update_status(
                    IRSIterationStatus(
                        iteration=iteration,
                        current_phase=phase_status,
                        reward_breakdown=result.avg_rewards,
                        current_weights=result.current_weights,
                        best_sample=result.best.candidate[:100],
                    )
                )

            avg_rewards = {k: v / len(questions) for k, v in total_rewards.items()}
            self._log_metrics(
                {
                    "event": "phase_score_done",
                    "iteration": iteration,
                    "avg_rewards": avg_rewards,
                    "final_weights": self.scorer.weights,
                }
            )

            # ===== Phase 3: SFT =====
            phase_status = IRSPhaseStatus(
                phase="sft",
                progress=0.0,
                samples_processed=0,
                samples_total=len(selected_samples),
                start_time=datetime.now(timezone.utc).isoformat(),
            )

            self._update_status(
                IRSIterationStatus(
                    iteration=iteration,
                    current_phase=phase_status,
                    reward_breakdown=avg_rewards,
                    current_weights=self.scorer.weights,
                )
            )

            self._load_model_for_training()
            sft_result = self._run_sft_step(selected_samples)
            self._unload_model()

            phase_status.progress = 1.0
            phase_status.samples_processed = len(selected_samples)

            iter_elapsed = time.time() - iter_start
            self._log_metrics(
                {
                    "event": "iteration_done",
                    "iteration": iteration,
                    "elapsed_s": iter_elapsed,
                    "sft_loss": sft_result["train_loss"],
                    "avg_rewards": avg_rewards,
                    "weights": self.scorer.weights,
                }
            )

            iteration_results.append(
                {
                    "iteration": iteration,
                    "elapsed_s": iter_elapsed,
                    "avg_rewards": avg_rewards,
                    "weights": dict(self.scorer.weights),
                    "sft_loss": sft_result["train_loss"],
                }
            )

        total_elapsed = time.time() - total_start
        summary = {
            "job_id": self.config.job_id,
            "total_elapsed_s": total_elapsed,
            "n_iterations": self.config.n_iterations,
            "n_questions": len(questions),
            "iteration_results": iteration_results,
            "final_weights": dict(self.scorer.weights),
            "final_checkpoint": str(
                self.job_dir
                / "checkpoints"
                / f"iter_{self.config.n_iterations}"
            ),
        }

        self._log_metrics({"event": "irs_run_done", "summary": summary})

        # 儲存 summary
        with open(self.job_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return summary


# 方便 import
import contextlib  # noqa: E402

__all__ = ["IRSRunner", "IRSConfig", "IRSPhaseStatus", "IRSIterationStatus"]
