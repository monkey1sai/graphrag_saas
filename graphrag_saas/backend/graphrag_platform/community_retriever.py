"""Community‑aware TF‑IDF retriever for SaaS deployment.

This is a copy of the original community retriever. It restricts
search to communities inferred from entity names using the
``CommunityIndex``. If no community matches, all chunks are searched.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set

from .tfidf import TfidfRetriever
from .community import CommunityIndex


class CommunityRetriever:
    """TF‑IDF retriever that restricts search to entity communities."""

    def __init__(self, chunks: List[str], entity_names: Dict[int, str]) -> None:
        self.chunks = chunks
        self.tfidf = TfidfRetriever(chunks)
        self.comm_index = CommunityIndex(entity_names, chunks)

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        comms = self.comm_index.get_comm_for_query(query)
        if comms:
            allowed_chunks: Set[int] = set()
            for comm in comms:
                allowed_chunks.update(self.comm_index.get_chunks_for_comm(comm))
        else:
            allowed_chunks = set(range(len(self.chunks)))
        q_vec = self.tfidf._vectorise_query(query)
        scores: List[Tuple[int, float]] = []
        for idx in allowed_chunks:
            vec = self.tfidf.vectors[idx]
            score = self.tfidf._cosine_similarity(q_vec, vec)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]