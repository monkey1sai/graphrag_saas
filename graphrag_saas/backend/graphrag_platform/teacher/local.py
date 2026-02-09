from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schema import Citation, TeacherOutputV1


@dataclass(frozen=True)
class LocalTeacher:
    """Local teacher using existing retrieval + integrator.

    This preserves the current Plan-B behavior while emitting strict JSON schema.
    """

    retrieve_fn: Callable[[str, int], list[tuple[int, float]]]
    answer_fn: Callable[[str, list[tuple[int, float]]], str]
    chunk_meta: list[dict[str, Any]] | None = None

    def generate(self, *, question: str, top_k: int) -> TeacherOutputV1:
        ranked = self.retrieve_fn(question, top_k)
        answer = self.answer_fn(question, ranked)

        citations: list[Citation] = []
        for chunk_id, score in ranked:
            source_path = None
            if self.chunk_meta is not None and 0 <= chunk_id < len(self.chunk_meta):
                source_path = self.chunk_meta[chunk_id].get("source_path")
            citations.append(Citation(chunk_id=int(chunk_id), score=float(score), source_path=source_path))

        prompt = f"Question: {question}\nAnswer:"  # stable prompt prefix
        completion = f" {answer}\n"  # leading space helps some tokenizers

        return TeacherOutputV1(
            question=question,
            prompt=prompt,
            completion=completion,
            citations=citations,
            teacher={"provider": "local"},
        )
