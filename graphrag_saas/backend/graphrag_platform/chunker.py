"""Simple text chunker module for splitting documents.

This file is copied from the `graphrag_platform` used in previous
iterations. It breaks input text into fixed‑size chunks separated by
newlines. The chunk size and overlap can be adjusted but remain
static for demonstration purposes.
"""

from __future__ import annotations

from typing import Iterable


class TextChunker:
    """Chunk large text documents into smaller pieces."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> Iterable[str]:
        """Yield overlapping chunks of the input text.

        Args:
            text: The raw text to chunk.

        Yields:
            Pieces of the text with specified length and overlap.
        """
        text = (text or "").strip()
        if not text:
            return

        words = text.split()
        # Many CJK OCR outputs contain very few spaces. When whitespace tokenisation
        # degenerates, fall back to fixed-length character chunking.
        if len(words) <= 2 and len(text) > self.chunk_size:
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(text), step):
                end = min(start + self.chunk_size, len(text))
                chunk = text[start:end].strip()
                if chunk:
                    yield chunk
                if end == len(text):
                    break
            return

        start = 0
        n = len(words)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                yield chunk
            if end == n:
                break
            start = end - self.overlap