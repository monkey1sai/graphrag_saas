from __future__ import annotations

import math
import re

from .base import ComponentScorer
from ..types import PromptContext


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    tokens.extend(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text))
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seq) < 2:
            continue
        if len(seq) == 2:
            tokens.append(seq)
            continue
        tokens.extend(seq[i : i + 2] for i in range(0, len(seq) - 1))
    return tokens


class HeuristicScorer(ComponentScorer):
    def score_rel(self, *, item: PromptContext, response: str) -> float:
        # Map rel_signal to [0, 1)
        return float(1.0 / (1.0 + math.exp(-3.0 * float(item.rel_signal))))

    def score_faith(self, *, item: PromptContext, response: str) -> float:
        ctx_tokens = set(_tokenize(item.context_text))
        resp_tokens = [t for t in _tokenize(response) if len(t) >= 2]
        if not resp_tokens or not ctx_tokens:
            return 0.0
        hit = sum(1 for t in resp_tokens if t in ctx_tokens)
        return float(hit / max(1, len(resp_tokens)))

    def score_conc(self, *, item: PromptContext, response: str, target_len_tokens: int) -> float:
        resp_len = max(1, len(_tokenize(response)))
        return float(1.0 / (1.0 + (resp_len / max(1, int(target_len_tokens)))))
