from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int = Field(..., ge=0)
    source_path: str | None = None
    score: float | None = None


class TeacherOutputV1(BaseModel):
    """Strict teacher output contract.

    This is used to generate supervised fine-tuning samples deterministically.
    Set `extra='forbid'` to ensure any unexpected keys cause validation failure.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["v1"] = "v1"

    question: str = Field(..., min_length=1)

    # `prompt` should include an explicit answer prefix, e.g.
    # "Question: ...\nAnswer:" so `completion` can follow directly.
    prompt: str = Field(..., min_length=1)
    completion: str = Field(..., min_length=1)

    citations: list[Citation] = Field(default_factory=list)

    # Provider metadata (for traceability); keep flexible but still JSON-friendly.
    teacher: dict[str, Any] = Field(default_factory=dict)
