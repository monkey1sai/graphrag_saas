"""Basic TF‑IDF retriever wrapper.

This file defines a simple wrapper around the ``TfidfRetriever`` class
for convenience. It exposes a ``retrieve`` method returning top
documents for a query.
"""

from __future__ import annotations

from typing import List, Tuple

from .tfidf import TfidfRetriever


class Retriever:
    def __init__(self, documents: List[str]) -> None:
        self.tfidf = TfidfRetriever(documents)

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        return self.tfidf.retrieve(query, top_n)