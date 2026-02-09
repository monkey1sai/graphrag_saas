"""Hierarchical TF‑IDF retriever for SaaS.

This module implements a two‑level hierarchical search strategy used in
the upgraded GraphRAG platform. It groups entities by first and
second letters using ``HierarchyBuilder`` and restricts search to
communities inferred from the query. It is a simplified analogue of
the hierarchical beam search described in Deep GraphRAG【447073342866022†L129-L135】.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Set

from .tfidf import TfidfRetriever
from .hierarchy import HierarchyBuilder


class HierarchicalRetriever:
    """A retriever that filters candidate chunks using a two‑level hierarchy."""

    def __init__(self, chunks: List[str], entity_names: Dict[int, str]) -> None:
        self.chunks = chunks
        self.tfidf = TfidfRetriever(chunks)
        self.hierarchy = HierarchyBuilder.build(entity_names)
        self.eid_to_lvl1: Dict[int, int] = {}
        self.eid_to_lvl2: Dict[int, int] = {}
        lvl1 = self.hierarchy[1]
        lvl2 = self.hierarchy[2]
        for lvl1_id, members in lvl1.items():
            for eid in members:
                self.eid_to_lvl1[eid] = lvl1_id
        for lvl2_id, members in lvl2.items():
            for lvl1_id in members:
                for eid in lvl1[lvl1_id]:
                    self.eid_to_lvl2[eid] = lvl2_id
        self.name_to_lvl1: Dict[str, int] = {}
        self.name_to_lvl2: Dict[str, int] = {}
        for eid, name in entity_names.items():
            n_lower = name.lower()
            if eid in self.eid_to_lvl1:
                self.name_to_lvl1[n_lower] = self.eid_to_lvl1[eid]
            if eid in self.eid_to_lvl2:
                self.name_to_lvl2[n_lower] = self.eid_to_lvl2[eid]
        self.lvl1_to_chunks: Dict[int, Set[int]] = {}
        self.lvl2_to_chunks: Dict[int, Set[int]] = {}
        # assign chunks to communities
        for idx, chunk in enumerate(chunks):
            text = chunk.lower()
            matched_lvl1: Set[int] = set()
            matched_lvl2: Set[int] = set()
            for name_lower, lvl1_id in self.name_to_lvl1.items():
                if name_lower in text:
                    matched_lvl1.add(lvl1_id)
            for name_lower, lvl2_id in self.name_to_lvl2.items():
                if name_lower in text:
                    matched_lvl2.add(lvl2_id)
            for lvl1_id in matched_lvl1:
                self.lvl1_to_chunks.setdefault(lvl1_id, set()).add(idx)
            for lvl2_id in matched_lvl2:
                self.lvl2_to_chunks.setdefault(lvl2_id, set()).add(idx)

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        q_lower = query.lower()
        matched_lvl2: Set[int] = set()
        for name_lower, lvl2_id in self.name_to_lvl2.items():
            if name_lower in q_lower:
                matched_lvl2.add(lvl2_id)
        if matched_lvl2:
            allowed_chunks: Set[int] = set()
            for lvl2_id in matched_lvl2:
                allowed_chunks.update(self.lvl2_to_chunks.get(lvl2_id, set()))
        else:
            matched_lvl1: Set[int] = set()
            for name_lower, lvl1_id in self.name_to_lvl1.items():
                if name_lower in q_lower:
                    matched_lvl1.add(lvl1_id)
            if matched_lvl1:
                allowed_chunks = set()
                for lvl1_id in matched_lvl1:
                    allowed_chunks.update(self.lvl1_to_chunks.get(lvl1_id, set()))
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