from __future__ import annotations

import os

from .base import ComponentScorer
from .heuristic import HeuristicScorer


def get_default_scorer() -> ComponentScorer:
    name = (os.getenv("REWARD_SCORER") or "bertscore_st").strip().lower()
    if name in {"heuristic", "simple"}:
        return HeuristicScorer()

    # Default: try model-based scorer, fall back silently.
    try:
        from .bertscore_st import BertScoreSentenceTransformerScorer

        return BertScoreSentenceTransformerScorer()
    except Exception:
        return HeuristicScorer()
