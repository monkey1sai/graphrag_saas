"""Simple knowledge integration module for SaaS.

This module contains two integrators: a simple chunk integrator that
selects sentences containing query tokens and a weighted integrator that
balances relevance, faithfulness and conciseness. These are copies of
their counterparts in the original Graphrag platform.
"""

import re
from typing import Iterable, List, Tuple


class ChunkIntegrator:
    """Integrate top‑ranked text chunks into a concise answer."""

    def integrate(self, query: str, ranked_chunks: List[Tuple[int, float]], chunks: List[str]) -> str:
        query_tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", query) if len(w) > 2]
        sentences: List[str] = []
        for idx, _score in ranked_chunks:
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            parts = re.split(r"(?<=[.!?。！？])\s+", chunk.strip())
            matched = 0
            for part in parts:
                if not part:
                    continue
                lower_part = part.lower()
                if any(tok in lower_part for tok in query_tokens):
                    sentences.append(part.strip())
                    matched += 1
                if matched >= 2:
                    break
            if len(sentences) >= 6:
                break
        if not sentences:
            for idx, _score in ranked_chunks:
                if idx < 0 or idx >= len(chunks):
                    continue
                parts = re.split(r"(?<=[.!?。！？])\s+", chunks[idx].strip())
                for part in parts[:2]:
                    if part:
                        sentences.append(part.strip())
                if len(sentences) >= 6:
                    break
            if not sentences:
                return "對不起，未能從知識庫中找到相關資訊。"
        answer = " ".join(sentences)
        words = answer.split()
        if len(words) > 80:
            answer = " ".join(words[:80]) + " …"
        return answer