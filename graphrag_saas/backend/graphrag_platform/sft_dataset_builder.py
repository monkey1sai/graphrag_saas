from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from .benchmark import QAItem
from .teacher.schema import Citation, TeacherOutputV1


class RetrievedChunkV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    chunk_id: int = Field(..., ge=0)
    score: float | None = None
    source_path: str | None = None
    text: str = Field(default="")


class AnswerStyleV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    format: list[str] = Field(default_factory=lambda: ["grounded", "concise"])
    max_length_chars: int = Field(default=1200, ge=0)


class MustIncludeGroupV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1)
    items: list[str] = Field(default_factory=list)


class SourceRefV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: str = Field(..., min_length=1)
    page: int = Field(default=1, ge=1)
    anchors: list[str] = Field(default_factory=lambda: ["chunk:0"], min_length=1)


class ItemSourceV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    item: str = Field(..., min_length=1)
    sources: list[SourceRefV1] = Field(default_factory=list, min_length=1)


class ExpectedV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    must_include: list[MustIncludeGroupV1] = Field(default_factory=list)
    keywords: dict[str, list[str]] = Field(default_factory=dict)
    answer_style: AnswerStyleV1 = Field(default_factory=AnswerStyleV1)
    sources: list[ItemSourceV1] = Field(default_factory=list)


class GradingV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    coverage_weight: float = 0.34
    grounding_weight: float = 0.33
    conciseness_weight: float = 0.33


class SFTQuestionItemV1(BaseModel):
    """A record compatible with docs/question.schema.json.

    The schema used in rag_eval requires: id, question, expected.
    We also attach extra fields (prompt/completion/retrieval/chunks) under additionalProperties.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    expected: ExpectedV1
    grading: GradingV1 | None = None

    # Extra fields for SFT/RAG traceability.
    prompt: str | None = None
    completion: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    teacher: dict[str, Any] = Field(default_factory=dict)
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    context_text: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class SFTDatasetArtifacts:
    dataset_path: str
    sha256: str
    count: int


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_keypoints(text: str, *, max_items: int = 6) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []

    lines = [ln.strip(" \t\r") for ln in t.split("\n") if ln.strip()]
    items: list[str] = []
    for ln in lines:
        ln = re.sub(r"^[\-\*\d\s\.\)\]]+", "", ln).strip()
        if ln and ln not in items:
            items.append(ln)
        if len(items) >= max_items:
            return items

    # Fallback: split by punctuation.
    for seg in re.split(r"[。！？!?；;]+", t):
        seg = seg.strip()
        if not seg:
            continue
        if seg not in items:
            items.append(seg)
        if len(items) >= max_items:
            break

    return items


def _tokenize_keywords(text: str, *, limit: int = 40) -> list[str]:
    if not text:
        return []

    tokens: list[str] = []
    tokens.extend(w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{2,}", text))
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seq) < 2:
            continue
        if len(seq) == 2:
            tokens.append(seq)
        else:
            tokens.extend(seq[i : i + 2] for i in range(0, len(seq) - 1))

    # de-dup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break

    return out


def build_sft_dataset(
    *,
    job_id: str,
    questions: list[QAItem],
    retrieve_fn: Callable[[str, int], list[tuple[int, float]]],
    chunks: list[str],
    chunk_meta: list[dict[str, Any]] | None,
    teacher: Any,
    top_k: int,
    meta: dict[str, Any] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[SFTQuestionItemV1]:
    """Build a schema-friendly SFT dataset from a teacher provider."""

    out_rows: list[SFTQuestionItemV1] = []
    base_meta = dict(meta or {})
    base_meta.update({"job_id": job_id, "top_k": int(top_k)})

    total = len(questions)
    for idx, it in enumerate(questions, start=1):
        if should_stop is not None and should_stop():
            break
        if progress_cb is not None:
            try:
                progress_cb(int(idx), int(total))
            except Exception:
                pass
        teacher_out: TeacherOutputV1 = teacher.generate(question=it.question, top_k=top_k)

        ranked = retrieve_fn(it.question, top_k)
        retrieved: list[RetrievedChunkV1] = []
        for chunk_id, score in ranked:
            if 0 <= chunk_id < len(chunks):
                source_path = None
                extra: dict[str, Any] = {}
                if chunk_meta is not None and 0 <= chunk_id < len(chunk_meta):
                    cm = dict(chunk_meta[chunk_id] or {})
                    source_path = cm.get("source_path")
                    # Avoid passing duplicate keyword args (source_path) via **extra.
                    for k in ("chunk_id", "score", "source_path", "text"):
                        cm.pop(k, None)
                    extra = cm
                retrieved.append(
                    RetrievedChunkV1(
                        chunk_id=int(chunk_id),
                        score=float(score),
                        source_path=source_path,
                        text=str(chunks[chunk_id] or ""),
                        **extra,
                    )
                )

        keypoints = _split_keypoints(teacher_out.completion, max_items=6)
        if not keypoints:
            keypoints = [teacher_out.completion.strip()[:200] or "(empty)"]

        must_include = [MustIncludeGroupV1(name="teacher_completion", items=keypoints)]

        # Build per-item sources: map each keypoint to at least one chunk source.
        sources: list[ItemSourceV1] = []

        # Prefer explicit citations from the teacher.
        citation_chunks: list[Citation] = list(teacher_out.citations or [])
        by_chunk_id: dict[int, RetrievedChunkV1] = {c.chunk_id: c for c in retrieved}

        def _source_for_chunk(chunk_id: int) -> SourceRefV1:
            if chunk_id in by_chunk_id:
                file = by_chunk_id[chunk_id].source_path or "unknown"
            else:
                file = "unknown"
            return SourceRefV1(file=file, page=1, anchors=[f"chunk_id:{int(chunk_id)}"])

        # Fallback to top-1 retrieved chunk.
        fallback_chunk_id = int(retrieved[0].chunk_id) if retrieved else 0

        for item_text in keypoints:
            refs: list[SourceRefV1] = []
            if citation_chunks:
                # pick first citation that exists; otherwise first citation anyway.
                chosen = citation_chunks[0]
                refs.append(_source_for_chunk(int(chosen.chunk_id)))
            else:
                refs.append(_source_for_chunk(fallback_chunk_id))
            sources.append(ItemSourceV1(item=item_text, sources=refs))

        expected = ExpectedV1(
            must_include=must_include,
            keywords={
                "core": _tokenize_keywords(it.question, limit=30),
                "completion": _tokenize_keywords(teacher_out.completion, limit=40),
            },
            answer_style=AnswerStyleV1(format=["grounded", "json"], max_length_chars=1200),
            sources=sources,
        )

        qid_str = f"Q{int(it.qid) + 1:04d}"

        row = SFTQuestionItemV1(
            id=qid_str,
            question=it.question,
            expected=expected,
            grading=GradingV1(),
            # Extra fields for SFT/RAG traceability
            prompt=teacher_out.prompt,
            completion=teacher_out.completion,
            citations=[c.model_dump() for c in (teacher_out.citations or [])],
            teacher=teacher_out.teacher,
            retrieved=[c.model_dump() for c in retrieved],
            context_text="\n\n".join([c.text for c in retrieved[: int(top_k)]]),
            meta={
                **base_meta,
                "dataset_fingerprint": _sha256_text(it.question + "\n" + teacher_out.completion),
            },
        )

        out_rows.append(row)

    return out_rows


def write_dataset_json(*, report_dir: Path, job_id: str, rows: list[SFTQuestionItemV1]) -> SFTDatasetArtifacts:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"sft_dataset_{job_id}.json"
    payload = [r.model_dump() for r in rows]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return SFTDatasetArtifacts(dataset_path=str(path), sha256=sha256_file(str(path)), count=len(rows))
