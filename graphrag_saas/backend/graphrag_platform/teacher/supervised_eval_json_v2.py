from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .schema import TeacherOutputV1


def _norm_anchors(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out: list[str] = []
        for x in val:
            s = str(x or "").strip()
            if s and s not in out:
                out.append(s)
        return out
    s = str(val or "").strip()
    if not s:
        return []
    # v1 anchors sometimes store a space-delimited string.
    parts = [p.strip() for p in s.replace("\t", " ").split(" ") if p.strip()]
    return parts[:6] if parts else [s]


def _completion_obj_from_expected(expected: dict[str, Any]) -> dict[str, Any]:
    exp = expected or {}
    ans = exp.get("answer_example") if isinstance(exp.get("answer_example"), dict) else {}
    sm = exp.get("source_map") if isinstance(exp.get("source_map"), list) else []

    source_map: list[dict[str, Any]] = []
    for entry in sm:
        if not isinstance(entry, dict):
            continue
        part = str(entry.get("part") or "").strip()
        refs_in = entry.get("refs") if isinstance(entry.get("refs"), list) else []
        refs: list[dict[str, Any]] = []
        for r in refs_in:
            if not isinstance(r, dict):
                continue
            refs.append(
                {
                    "file": str(r.get("file") or ""),
                    "page": int(r.get("page") or 1),
                    "anchors": _norm_anchors(r.get("anchors")),
                }
            )
        if part:
            source_map.append({"part": part, "refs": refs or [{"file": "", "page": 1, "anchors": []}]})

    return {
        "target_audience": str(ans.get("target_audience") or ""),
        "main_topic": str(ans.get("main_topic") or ""),
        "sub_topic": str(ans.get("sub_topic") or ""),
        "detailed_description": list(ans.get("detailed_description") or []),
        "original_evidence": str(ans.get("original_evidence") or ""),
        "source_map": source_map,
        "predicted_questions": list(ans.get("predicted_questions") or []),
    }


@dataclass(frozen=True)
class SupervisedEvalJsonV2Teacher:
    """Teacher backed by labeled ExpectedV1/V2 question sets.

    Produces (prompt, completion) pairs aligned with docs/prompts/eval_json_v2.md output schema.
    """

    system_text: str
    user_template: str
    expected_by_question: dict[str, dict[str, Any]]
    retrieve_fn: Callable[[str, int], list[tuple[int, float]]]
    chunks: list[str]
    chunk_meta: list[dict[str, Any]] | None = None
    context_max_chars_per_chunk: int = 400
    context_max_total_chars: int = 4000

    def generate(self, *, question: str, top_k: int) -> TeacherOutputV1:
        ranked = self.retrieve_fn(question, top_k)
        ctx: list[dict[str, Any]] = []
        used = 0
        for cid, score in ranked:
            if not (0 <= cid < len(self.chunks)):
                continue
            sp = None
            if self.chunk_meta is not None and 0 <= cid < len(self.chunk_meta):
                try:
                    sp = self.chunk_meta[cid].get("source_path")
                except Exception:
                    sp = None
            txt = str(self.chunks[cid] or "")
            if self.context_max_chars_per_chunk > 0 and len(txt) > self.context_max_chars_per_chunk:
                txt = txt[: int(self.context_max_chars_per_chunk)]
            if self.context_max_total_chars > 0 and (used + len(txt)) > self.context_max_total_chars and ctx:
                break
            used += len(txt)
            ctx.append({"chunk_id": int(cid), "score": float(score), "source_path": sp, "text": txt})

        # Keep the same placeholders as the eval prompt asset. The trainer may later
        # wrap these into the model's chat template for better instruction tuning.
        user_text = (
            self.user_template.replace("{{question}}", (question or "").strip()).replace(
                "{{context_chunks_json}}", json.dumps(ctx, ensure_ascii=False)
            )
        )
        prompt = user_text

        expected = self.expected_by_question.get(question) or {}
        completion_obj = _completion_obj_from_expected(expected)
        completion_json = json.dumps(completion_obj, ensure_ascii=False)
        completion_text = " " + completion_json + "\n"

        return TeacherOutputV1.model_validate(
            {
                "schema_version": "v1",
                "question": question,
                "prompt": prompt,
                "completion": completion_text,
                "citations": [],
                "teacher": {
                    "provider": "supervised",
                    "mode": "eval_json_v2",
                    "system_text": (self.system_text or "").strip(),
                    "user_text": user_text,
                    "completion_json": completion_json,
                },
            }
        )
