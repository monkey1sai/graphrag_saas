"""TF‑IDF retrieval module for SaaS.

This module implements a basic TF‑IDF retriever without external
dependencies. It vectorises text using term frequency and inverse
document frequency and computes cosine similarities for ranking.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


class TfidfRetriever:
    def __init__(self, documents: List[str]) -> None:
        self.documents = documents
        # build vocabulary and term frequencies
        self.vocab: Dict[str, int] = {}
        self.doc_freq: Dict[str, int] = {}
        self.vectors: List[Dict[str, float]] = []
        for doc in documents:
            tf: Dict[str, int] = {}
            for token in self._tokenize(doc):
                tf[token] = tf.get(token, 0) + 1
            for token in tf.keys():
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
            self.vectors.append(tf)
        # compute idf values
        self.idf: Dict[str, float] = {}
        N = len(documents)
        for token, df in self.doc_freq.items():
            self.idf[token] = math.log((N + 1) / (df + 1)) + 1
        # normalise vectors
        self._normalise_vectors()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into a mix of latin words and CJK character n-grams.

        The original implementation only handled `[A-Za-z0-9]+`, which makes
        retrieval effectively fail on Chinese OCR content.
        """

        import re

        if not text:
            return []

        tokens: List[str] = []
        tokens.extend(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text))

        # CJK sequences → bigrams (fallback to unigrams for very short sequences)
        cjk_seqs = re.findall(r"[\u4e00-\u9fff]+", text)
        for seq in cjk_seqs:
            seq = seq.strip()
            if len(seq) < 2:
                continue
            if len(seq) == 2:
                tokens.append(seq)
                continue
            for i in range(0, len(seq) - 1):
                tokens.append(seq[i : i + 2])

        return tokens

    def _normalise_vectors(self) -> None:
        normalised: List[Dict[str, float]] = []
        for tf in self.vectors:
            vec: Dict[str, float] = {}
            norm = 0.0
            for token, freq in tf.items():
                weight = freq * self.idf.get(token, 0.0)
                vec[token] = weight
                norm += weight * weight
            norm = math.sqrt(norm) or 1.0
            for token in vec:
                vec[token] /= norm
            normalised.append(vec)
        self.vectors = normalised

    def _vectorise_query(self, query: str) -> Dict[str, float]:
        tf: Dict[str, int] = {}
        for token in self._tokenize(query):
            tf[token] = tf.get(token, 0) + 1
        vec: Dict[str, float] = {}
        norm = 0.0
        for token, freq in tf.items():
            weight = freq * self.idf.get(token, 0.0)
            vec[token] = weight
            norm += weight * weight
        norm = math.sqrt(norm) or 1.0
        for token in vec:
            vec[token] /= norm
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1
        score = 0.0
        for token, weight in vec1.items():
            if token in vec2:
                score += weight * vec2[token]
        return score

    def retrieve(self, query: str, top_n: int = 5) -> List[Tuple[int, float]]:
        q_vec = self._vectorise_query(query)
        scores: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.vectors):
            score = self._cosine_similarity(q_vec, doc_vec)
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]