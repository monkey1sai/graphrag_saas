from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class JobStore:
    """Persists background job states to disk.

    This is intentionally simple: a single JSON file on a local volume.
    It enables `/jobs/{id}` and `/api/train/status/{id}` to survive container restarts.
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read_unlocked()
            jobs = data.get("jobs", {})
            if isinstance(jobs, dict):
                return [dict(v) for v in jobs.values() if isinstance(v, dict)]
            return []

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read_unlocked()
            jobs = data.get("jobs", {})
            if isinstance(jobs, dict):
                value = jobs.get(job_id)
                if isinstance(value, dict):
                    return dict(value)
            return None

    def upsert(self, record: dict[str, Any]) -> None:
        job_id = str(record.get("job_id") or "")
        if not job_id:
            return

        with self._lock:
            data = self._read_unlocked()
            jobs = data.get("jobs")
            if not isinstance(jobs, dict):
                jobs = {}
                data["jobs"] = jobs

            jobs[job_id] = record
            self._write_unlocked(data)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"jobs": {}}
        try:
            raw = self._path.read_text(encoding="utf-8")
            if not raw.strip():
                return {"jobs": {}}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            # If file is corrupted, do not crash the service.
            return {"jobs": {}}
        return {"jobs": {}}

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)
