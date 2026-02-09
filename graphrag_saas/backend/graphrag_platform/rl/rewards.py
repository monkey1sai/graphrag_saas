from __future__ import annotations

from dataclasses import dataclass

from .scorers import get_default_scorer
from .types import PromptContext, RewardComponents, RewardWeights


def reward_components(*, item: PromptContext, response: str, target_len_tokens: int = 120) -> RewardComponents:
    """Compute (rel, faith, conc) in [0, 1]-ish range.

    Default scorer uses sentence-transformers (rel) + BERTScore (faith), and falls
    back to a lightweight heuristic scorer if deps are missing.

    Controls:
    - REWARD_SCORER: 'bertscore_st' (default) | 'heuristic'
    - REWARD_DEVICE: 'cpu' (default) | 'cuda'
    - ST_MODEL_ID, BERTSCORE_MODEL_TYPE
    """

    scorer = get_default_scorer()
    rel = float(scorer.score_rel(item=item, response=response))
    faith = float(scorer.score_faith(item=item, response=response))
    conc = float(scorer.score_conc(item=item, response=response, target_len_tokens=int(target_len_tokens)))

    return RewardComponents(rel=float(rel), faith=float(faith), conc=float(conc))


def composite_reward(*, weights: RewardWeights, comps: RewardComponents) -> float:
    w = weights.normalized()
    # Clamp to prevent extreme negative/positive spikes.
    r = (w.w_rel * comps.rel) + (w.w_faith * comps.faith) + (w.w_conc * comps.conc)
    return float(max(0.0, min(1.0, r)))


@dataclass(frozen=True)
class RewardResult:
    reward: float
    components: RewardComponents


def compute_reward(*, item: PromptContext, response: str, weights: RewardWeights) -> RewardResult:
    comps = reward_components(item=item, response=response)
    return RewardResult(reward=composite_reward(weights=weights, comps=comps), components=comps)
