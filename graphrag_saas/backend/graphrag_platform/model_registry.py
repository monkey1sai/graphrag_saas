from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Stage = Literal["SFT", "RL", "IRS"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        raw = path.read_text(encoding="utf-8")
        if not raw.strip():
            return default
        return json.loads(raw)
    except Exception:
        return default


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _container_path_to_workspace_hint(p: str | None) -> str | None:
    """Best-effort hint for host/workspace relative paths.

    In docker-compose.yml we mount:
      /app/reports -> graphrag_saas/backend/reports
      /app/data    -> graphrag_saas/backend/data
      /app/docs    -> graphrag_saas/backend/docs
    """

    if not p:
        return None
    s = str(p)
    s = s.replace("\\", "/")
    if s.startswith("/app/reports/"):
        return "graphrag_saas/backend/reports/" + s.removeprefix("/app/reports/")
    if s.startswith("/app/data/"):
        return "graphrag_saas/backend/data/" + s.removeprefix("/app/data/")
    if s.startswith("/app/docs/"):
        return "graphrag_saas/backend/docs/" + s.removeprefix("/app/docs/")
    return None


@dataclass(frozen=True)
class ModelEntry:
    id: str
    stage: Stage
    job_id: str | None
    created_at: str
    base_model: str | None
    artifacts: dict[str, Any]
    metrics: dict[str, Any]
    runtime: dict[str, Any]
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "runtime": self.runtime,
            "tags": list(self.tags),
        }


class ModelRegistry:
    """Simple on-disk registry for train artifacts + active pointers (per stage).

    Thread-safe within a single process via a lock and atomic file replacement.
    """

    def __init__(self, *, registry_path: Path, active_path: Path) -> None:
        self._lock = threading.Lock()
        self._registry_path = registry_path
        self._active_path = active_path
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_path.parent.mkdir(parents=True, exist_ok=True)

    def list(self, *, stage: Stage | None = None) -> list[dict[str, Any]]:
        with self._lock:
            data = _read_json(self._registry_path, default={"models": []})
            models = data.get("models", [])
            if not isinstance(models, list):
                return []
            out: list[dict[str, Any]] = []
            for m in models:
                if not isinstance(m, dict):
                    continue
                if stage and str(m.get("stage")) != stage:
                    continue
                out.append(dict(m))
            return out

    def get(self, model_id: str) -> dict[str, Any] | None:
        model_id = (model_id or "").strip()
        if not model_id:
            return None
        with self._lock:
            data = _read_json(self._registry_path, default={"models": []})
            models = data.get("models", [])
            if not isinstance(models, list):
                return None
            for m in models:
                if isinstance(m, dict) and str(m.get("id")) == model_id:
                    return dict(m)
            return None

    def register(
        self,
        *,
        stage: Stage,
        base_model: str | None,
        job_id: str | None,
        artifacts: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> ModelEntry:
        entry = ModelEntry(
            id=str(uuid.uuid4()),
            stage=stage,
            job_id=(job_id or None),
            created_at=_utc_now_iso(),
            base_model=(base_model or None),
            artifacts=dict(artifacts or {}),
            metrics=dict(metrics or {}),
            runtime=dict(runtime or {}),
            tags=list(tags or []),
        )

        # Add workspace hints for common artifact paths.
        try:
            for k, v in list(entry.artifacts.items()):
                if isinstance(v, str):
                    hint = _container_path_to_workspace_hint(v)
                    if hint:
                        entry.artifacts.setdefault(f"{k}_workspace_hint", hint)
        except Exception:
            pass

        with self._lock:
            data = _read_json(self._registry_path, default={"models": []})
            models = data.get("models")
            if not isinstance(models, list):
                models = []
                data["models"] = models
            models.append(entry.to_dict())
            _atomic_write_json(self._registry_path, data)

        return entry

    def get_active_ids(self) -> dict[Stage, str | None]:
        with self._lock:
            data = _read_json(self._active_path, default={})
            out: dict[Stage, str | None] = {"SFT": None, "RL": None, "IRS": None}
            if isinstance(data, dict):
                for stage in ["SFT", "RL", "IRS"]:
                    v = data.get(stage)
                    out[stage] = str(v) if isinstance(v, str) and v.strip() else None
            return out

    def set_active(self, *, stage: Stage, model_id: str) -> dict[Stage, str | None]:
        model_id = (model_id or "").strip()
        if not model_id:
            raise ValueError("model_id is required")

        with self._lock:
            reg = _read_json(self._registry_path, default={"models": []})
            models = reg.get("models", [])
            if not isinstance(models, list):
                models = []

            existing: dict[str, Any] | None = None
            for m in models:
                if isinstance(m, dict) and str(m.get("id")) == model_id:
                    existing = m
                    break

            if not existing:
                raise ValueError(f"model not found: {model_id}")
            if str(existing.get("stage")) != stage:
                raise ValueError(f"model stage mismatch: expected {stage}, got {existing.get('stage')!r}")

            data = _read_json(self._active_path, default={})
            if not isinstance(data, dict):
                data = {}
            data[str(stage)] = model_id
            _atomic_write_json(self._active_path, data)

        return self.get_active_ids()

    def annotate(
        self,
        *,
        model_id: str,
        metrics_patch: dict[str, Any] | None = None,
        artifacts_patch: dict[str, Any] | None = None,
        runtime_patch: dict[str, Any] | None = None,
        tags_add: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Patch an existing registry entry in-place (atomic write).

        This is used to attach evaluation results (score/pass rate/prompt hashes)
        after a model has been registered by training.
        """

        model_id = (model_id or "").strip()
        if not model_id:
            return None

        with self._lock:
            data = _read_json(self._registry_path, default={"models": []})
            models = data.get("models", [])
            if not isinstance(models, list):
                return None

            updated: dict[str, Any] | None = None
            for m in models:
                if not isinstance(m, dict) or str(m.get("id")) != model_id:
                    continue

                if metrics_patch:
                    base = m.get("metrics")
                    if not isinstance(base, dict):
                        base = {}
                    base.update(dict(metrics_patch))
                    m["metrics"] = base

                if artifacts_patch:
                    base = m.get("artifacts")
                    if not isinstance(base, dict):
                        base = {}
                    base.update(dict(artifacts_patch))
                    # Add workspace hints for common artifact paths.
                    try:
                        for k, v in list(base.items()):
                            if isinstance(v, str):
                                hint = _container_path_to_workspace_hint(v)
                                if hint:
                                    base.setdefault(f"{k}_workspace_hint", hint)
                    except Exception:
                        pass
                    m["artifacts"] = base

                if runtime_patch:
                    base = m.get("runtime")
                    if not isinstance(base, dict):
                        base = {}
                    base.update(dict(runtime_patch))
                    m["runtime"] = base

                if tags_add:
                    base_tags = m.get("tags")
                    if not isinstance(base_tags, list):
                        base_tags = []
                    for t in tags_add:
                        ts = str(t).strip()
                        if ts and ts not in base_tags:
                            base_tags.append(ts)
                    m["tags"] = base_tags

                updated = dict(m)
                break

            if updated is None:
                return None

            _atomic_write_json(self._registry_path, data)
            return updated
