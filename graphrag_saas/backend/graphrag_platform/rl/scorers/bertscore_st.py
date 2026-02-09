from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .base import ComponentScorer
from ..types import PromptContext


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or str(v).strip() == "" else str(v).strip()


def _device() -> str:
    # Default to CPU to avoid stealing VRAM from PPO/QLoRA.
    return _env("REWARD_DEVICE", "cpu")


def _st_model_id() -> str:
    return _env("ST_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")


def _bertscore_model_type() -> str:
    # Multilingual baseline works for zh/en.
    return _env("BERTSCORE_MODEL_TYPE", "bert-base-multilingual-cased")


@lru_cache(maxsize=1)
def _get_sentence_transformer() -> Any:
    from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

    model = SentenceTransformer(_st_model_id(), device=_device())
    return model


@lru_cache(maxsize=2048)
def _encode(text: str) -> Any:
    model = _get_sentence_transformer()
    return model.encode(text or "", convert_to_tensor=True, normalize_embeddings=True)


class BertScoreSentenceTransformerScorer(ComponentScorer):
    """Model-based scorer: rel via sentence-transformers, faith via BERTScore, conc via length.

    Notes:
    - This is expensive; expect slower PPO steps.
    - Default device is CPU to avoid GPU VRAM contention with PPO.
    """

    def __init__(self, *, target_len_tokens: int = 120) -> None:
        self._target_len_tokens = int(target_len_tokens)

    def score_rel(self, *, item: PromptContext, response: str) -> float:
        # cos(embed(question), embed(response)) in [ -1, 1 ] -> clamp to [0, 1]
        try:
            from sentence_transformers import util  # pyright: ignore[reportMissingImports]

            q = _encode(item.question)
            r = _encode(response or "")
            sim = float(util.cos_sim(q, r).item())
            return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
        except Exception:
            return 0.0

    def score_faith(self, *, item: PromptContext, response: str) -> float:
        # BERTScore F1 between response and provided context.
        try:
            from bert_score import score as bertscore  # pyright: ignore[reportMissingImports]

            cand = [response or ""]
            ref = [item.context_text or ""]

            _, _, f1 = bertscore(
                cand,
                ref,
                model_type=_bertscore_model_type(),
                device=_device(),
                verbose=False,
                rescale_with_baseline=False,
            )
            v = float(f1[0].detach().cpu().item())
            # f1 is usually in [0,1], but clamp defensively
            return float(max(0.0, min(1.0, v)))
        except Exception:
            return 0.0

    def score_conc(self, *, item: PromptContext, response: str, target_len_tokens: int) -> float:
        # Lightweight conciseness proxy; keep stable.
        # Approx token count by whitespace split; good enough for a penalty curve.
        resp_len = max(1, len((response or "").split()))
        tgt = max(1, int(target_len_tokens or self._target_len_tokens))
        return float(1.0 / (1.0 + (resp_len / tgt)))
