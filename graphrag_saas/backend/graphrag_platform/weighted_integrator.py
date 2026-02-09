"""Weighted integrator module for SaaS deployment.

See original implementation in Graphrag platform. This integrator
computes relevance, faithfulness and conciseness scores for candidate
sentences and selects the top ones according to weighted sum.
"""

from __future__ import annotations

import re
from typing import List, Tuple


class WeightedIntegrator:
    def __init__(self, w_rel: float = 0.4, w_faith: float = 0.3, w_conc: float = 0.3) -> None:
        self.w_rel = w_rel
        self.w_faith = w_faith
        self.w_conc = w_conc

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        tokens.extend(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text))
        for seq in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(seq) < 2:
                continue
            if len(seq) == 2:
                tokens.append(seq)
                continue
            tokens.extend(seq[i : i + 2] for i in range(0, len(seq) - 1))
        return tokens

    def integrate(self, query: str, ranked_chunks: List[Tuple[int, float]], chunks: List[str], max_sentences: int = 4) -> str:
        query_tokens = [t for t in self._tokenize(query) if len(t) >= 2]
        candidates: List[Tuple[str, float]] = []
        for idx, score in ranked_chunks:
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            parts = re.split(r"(?<=[.!?。！？])\s*", chunk.strip())
            for part in parts:
                if not part:
                    continue
                tokens_in_sentence = self._tokenize(part)
                if not tokens_in_sentence:
                    continue
                match_count = sum(1 for t in query_tokens if t in tokens_in_sentence)
                faithfulness = match_count / len(query_tokens) if query_tokens else 0.0
                word_count = len(tokens_in_sentence)
                conciseness = 1.0 / (word_count + 1e-6)
                final_score = self.w_rel * score + self.w_faith * faithfulness + self.w_conc * conciseness
                candidates.append((part.strip(), final_score))
        if not candidates:
            return "對不起，未能從知識庫中找到相關資訊。"
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [sent for sent, _s in candidates[:max_sentences]]
        answer = " ".join(selected)
        words = answer.split()
        if len(words) > 100:
            answer = " ".join(words[:100]) + " …"
        return answer