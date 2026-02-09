from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import RewardComponents, RewardWeights


def _softmax(xs: list[float], temperature: float) -> list[float]:
    t = max(1e-6, float(temperature))
    m = max(xs) if xs else 0.0
    exps = [math.exp((x - m) / t) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [e / s for e in exps]


@dataclass
class DWGRPOWeightScheduler:
    """Dynamic weight scheduler for (rel, faith, conc).

    Implements a simplified sliding-window slope based update:
    - Track recent component rewards.
    - Estimate slope per component.
    - Softmax slopes to allocate more weight to improving signals.

    This is a pragmatic implementation aimed at Phase 4 wiring, not a paper-exact replica.
    """

    window: int = 20
    temperature: float = 0.25
    min_weight: float = 0.05
    history: list[RewardComponents] = field(default_factory=list)

    def push(self, comps: RewardComponents) -> None:
        self.history.append(comps)
        if len(self.history) > max(5, self.window * 3):
            # keep bounded memory
            self.history = self.history[-self.window * 3 :]

    def current_weights(self, base: RewardWeights) -> RewardWeights:
        if len(self.history) < max(4, self.window):
            return base.normalized()

        w = int(self.window)
        recent = self.history[-w:]
        prev = self.history[-2 * w : -w] if len(self.history) >= 2 * w else self.history[:w]

        def avg(xs: list[RewardComponents]) -> RewardComponents:
            if not xs:
                return RewardComponents(rel=0.0, faith=0.0, conc=0.0)
            return RewardComponents(
                rel=sum(x.rel for x in xs) / len(xs),
                faith=sum(x.faith for x in xs) / len(xs),
                conc=sum(x.conc for x in xs) / len(xs),
            )

        a1 = avg(prev)
        a2 = avg(recent)

        slopes = [a2.rel - a1.rel, a2.faith - a1.faith, a2.conc - a1.conc]
        alloc = _softmax(slopes, temperature=self.temperature)

        # mix with base weights to avoid oscillation
        base_n = base.normalized()
        mixed = [
            0.5 * base_n.w_rel + 0.5 * alloc[0],
            0.5 * base_n.w_faith + 0.5 * alloc[1],
            0.5 * base_n.w_conc + 0.5 * alloc[2],
        ]

        # enforce min weight
        mixed = [max(self.min_weight, x) for x in mixed]
        s = sum(mixed)
        mixed = [x / s for x in mixed]

        return RewardWeights(w_rel=mixed[0], w_faith=mixed[1], w_conc=mixed[2])
