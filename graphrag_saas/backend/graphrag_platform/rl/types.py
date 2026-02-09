from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Weights for composite reward.

    Expected to sum to ~1.0, but code should be robust if they do not.
    """

    w_rel: float = 0.4
    w_faith: float = 0.3
    w_conc: float = 0.3

    def normalized(self) -> "RewardWeights":
        total = float(self.w_rel + self.w_faith + self.w_conc)
        if total <= 0:
            return RewardWeights(w_rel=1 / 3, w_faith=1 / 3, w_conc=1 / 3)
        return RewardWeights(
            w_rel=float(self.w_rel) / total,
            w_faith=float(self.w_faith) / total,
            w_conc=float(self.w_conc) / total,
        )


@dataclass(frozen=True)
class RewardComponents:
    rel: float
    faith: float
    conc: float


@dataclass(frozen=True)
class PromptContext:
    """A single RL sample context.

    - `prompt` is what the policy sees.
    - `context_text` is a compact concatenation of retrieved chunks, used by reward.
    - `rel_signal` is derived from retrieval scores (mean top-k similarity).
    """

    question: str
    prompt: str
    context_text: str
    rel_signal: float
