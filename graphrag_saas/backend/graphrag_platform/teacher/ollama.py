from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .schema import TeacherOutputV1


def _extract_json_object(text: str) -> Any:
    """Best-effort extraction of a JSON object from model output."""

    text = (text or "").strip()
    if not text:
        raise ValueError("Empty response")

    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Response is not valid JSON")


def _normalize_teacher_obj(obj: Any, *, model_name: str) -> dict[str, Any]:
    """Best-effort normalization to satisfy TeacherOutputV1 strict contract.

    Some models occasionally:
    - emit teacher as a string ("teacher") instead of an object
    - use `source` instead of `source_path` in citations
    - add extra keys inside citations
    We normalize these to reduce brittle failures while keeping determinism.
    """

    if not isinstance(obj, dict):
        raise ValueError("Teacher output must be a JSON object.")

    out = dict(obj)

    # Ensure required top-level keys exist
    out.setdefault("schema_version", "v1")
    out.setdefault("citations", [])
    out.setdefault("teacher", {"provider": "ollama", "model": model_name})

    # teacher must be an object
    if not isinstance(out.get("teacher"), dict):
        out["teacher"] = {"provider": "ollama", "model": model_name}

    # Normalize citations
    cits = out.get("citations")
    if not isinstance(cits, list):
        cits = []
    norm_cits: list[dict[str, Any]] = []
    for c in cits:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if "source" in cc and "source_path" not in cc:
            cc["source_path"] = cc.get("source")
        # Drop unknown keys (TeacherOutputV1 extra=forbid)
        norm_cits.append(
            {
                "chunk_id": cc.get("chunk_id"),
                "source_path": cc.get("source_path"),
                "score": cc.get("score"),
            }
        )
    out["citations"] = norm_cits

    # Prompt/Completion shaping for stable concatenation.
    prompt = str(out.get("prompt") or "")
    if "Answer:" not in prompt:
        # Keep original question if present
        q = str(out.get("question") or "").strip()
        prompt = f"Question: {q}\nAnswer:"
    if not prompt.rstrip().endswith("Answer:"):
        # Ensure exact suffix
        prompt = prompt.rstrip()
        if not prompt.endswith("Answer:"):
            prompt = prompt + "\nAnswer:"
    out["prompt"] = prompt

    completion = str(out.get("completion") or "")
    if completion and not completion.startswith(" "):
        completion = " " + completion.lstrip()
    if completion and not completion.endswith("\n"):
        completion = completion.rstrip() + "\n"
    out["completion"] = completion

    return out


@dataclass(frozen=True)
class OllamaTeacher:
    """Teacher provider backed by Ollama Server API.

    Requires an Ollama server (default: http://localhost:11434).

    We request JSON output and then validate it strictly against TeacherOutputV1.
    Any extra keys cause validation failure.
    """

    base_url: str
    model: str
    retrieve_fn: Callable[[str, int], list[tuple[int, float]]]
    chunks: list[str]
    chunk_meta: list[dict[str, Any]] | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 3
    context_max_chars_per_chunk: int = 1200
    context_max_total_chars: int = 8000

    def generate(self, *, question: str, top_k: int) -> TeacherOutputV1:
        import httpx  # pyright: ignore[reportMissingImports]

        ranked = self.retrieve_fn(question, top_k)

        context: list[dict[str, Any]] = []
        used_chars = 0
        for chunk_id, score in ranked:
            if 0 <= chunk_id < len(self.chunks):
                source_path = None
                if self.chunk_meta is not None and 0 <= chunk_id < len(self.chunk_meta):
                    source_path = self.chunk_meta[chunk_id].get("source_path")
                raw_text = self.chunks[chunk_id]
                txt = raw_text
                try:
                    limit = max(0, int(self.context_max_chars_per_chunk))
                except Exception:
                    limit = 1200
                if limit and len(txt) > limit:
                    txt = txt[:limit]
                # Enforce a total context budget to avoid prompt explosions.
                try:
                    total_limit = max(0, int(self.context_max_total_chars))
                except Exception:
                    total_limit = 8000
                if total_limit and (used_chars + len(txt)) > total_limit and context:
                    break
                used_chars += len(txt)
                context.append(
                    {
                        "chunk_id": int(chunk_id),
                        "score": float(score),
                        "source_path": source_path,
                        "text": txt,
                    }
                )

        last_err: Exception | None = None
        last_raw: str = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = self._request_payload(question, context=context)
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(f"{self.base_url.rstrip('/')}/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()

                content = ""
                if isinstance(data, dict):
                    msg = data.get("message")
                    if isinstance(msg, dict):
                        content = str(msg.get("content") or "")
                        # Some Ollama builds/models may emit the primary payload under `thinking`.
                        if not content.strip():
                            content = str(msg.get("thinking") or "")
                last_raw = content or ""

                obj = _extract_json_object(content)
                norm = _normalize_teacher_obj(obj, model_name=self.model)
                out = TeacherOutputV1.model_validate(norm)

                # Enrich traceability fields deterministically.
                out.teacher = {"provider": "ollama", "model": self.model}
                return out
            except Exception as e:
                last_err = e
                # Small backoff for transient failures
                time.sleep(0.25 * attempt)

        # Fallback: if we still got a non-empty natural language answer, wrap it into TeacherOutputV1.
        # This keeps training unblocked when a model cannot reliably emit strict JSON.
        if (last_raw or "").strip():
            prompt = f"Question: {question.strip()}\nAnswer:"
            completion = " " + last_raw.strip() + "\n"
            return TeacherOutputV1.model_validate(
                {
                    "schema_version": "v1",
                    "question": question,
                    "prompt": prompt,
                    "completion": completion,
                    "citations": [],
                    "teacher": {"provider": "ollama", "model": self.model, "mode": "plain_text_fallback"},
                }
            )

        raise RuntimeError(f"OllamaTeacher failed after {self.max_retries} retries: {last_err}")

    def _request_payload(self, question: str, *, context: list[dict[str, Any]]) -> dict[str, Any]:
        # Strong instruction: output JSON only (no Markdown, no extra text).
        # We still validate strictly server-side.
        schema_hint = {
            "schema_version": "v1",
            "question": "...",
            "prompt": "Question: ...\\nAnswer:",
            "completion": " ...\\n",
            "citations": [
                {"chunk_id": 0, "source_path": "path/to/file", "score": 0.0}
            ],
            "teacher": {"provider": "ollama", "model": self.model},
        }

        system = (
            "You are a teacher model that creates supervised fine-tuning examples. "
            "Return ONLY a single JSON object and NOTHING else. "
            "All keys must exactly match the schema (no extra keys). "
            "Use schema_version='v1'. "
            "prompt must end with 'Answer:' and completion must start with a leading space and end with a newline."
        )

        user = (
            "Generate a TeacherOutputV1 JSON for this question using ONLY the provided context. "
            "If the context is insufficient, still answer but keep it conservative. "
            "citations (if any) MUST reference only chunk_id values from the context list below.\n\n"
            f"Question: {question}\n\n"
            f"Context (top-k retrieved chunks): {json.dumps(context, ensure_ascii=False)}\n\n"
            f"Schema example (for reference only): {json.dumps(schema_hint, ensure_ascii=False)}"
        )

        # Ollama supports `format` as either "json" or a JSON Schema. Prefer a schema to reduce
        # non-JSON outputs on some models.
        teacher_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "question", "prompt", "completion", "citations", "teacher"],
            "properties": {
                "schema_version": {"type": "string"},
                "question": {"type": "string"},
                "prompt": {"type": "string"},
                "completion": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["chunk_id"],
                        "properties": {
                            "chunk_id": {"type": "integer"},
                            "source_path": {"type": ["string", "null"]},
                            "score": {"type": ["number", "null"]},
                        },
                    },
                },
                "teacher": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["provider", "model"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                    },
                },
            },
        }

        return {
            "model": self.model,
            "stream": False,
            # Ask Ollama to enforce JSON via schema if supported.
            "format": teacher_schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                # keep outputs stable-ish
                "temperature": 0.2,
                # Bound output length for stability and avoid empty / runaway generations.
                "num_predict": 300,
            },
        }
