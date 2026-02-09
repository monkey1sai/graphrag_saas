from __future__ import annotations

from typing import Protocol

from ..types import PromptContext


class ComponentScorer(Protocol):
    def score_rel(self, *, item: PromptContext, response: str) -> float: ...

    def score_faith(self, *, item: PromptContext, response: str) -> float: ...

    def score_conc(self, *, item: PromptContext, response: str, target_len_tokens: int) -> float: ...
