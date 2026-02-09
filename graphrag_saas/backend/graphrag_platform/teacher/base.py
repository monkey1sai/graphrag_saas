from __future__ import annotations

from typing import Protocol

from .schema import TeacherOutputV1


class Teacher(Protocol):
    """Teacher provider interface (structural typing).

    Implementations must return a validated `TeacherOutputV1`.
    """

    def generate(self, *, question: str, top_k: int) -> TeacherOutputV1: ...
