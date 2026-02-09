from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

REQUEST_ID_HEADER = "X-Request-Id"
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return str(uuid.uuid4())


def set_request_id(rid: str | None) -> None:
    request_id_var.set((rid or "").strip() or None)


def get_request_id() -> str | None:
    rid = request_id_var.get()
    return (rid or "").strip() or None


def _utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": _utc_now_iso_z(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        rid = get_request_id()
        if rid:
            base["request_id"] = rid

        if record.exc_info:
            try:
                base["exc"] = self.formatException(record.exc_info)
            except Exception:
                base["exc"] = "exception"

        # Optional extras via logger.*(..., extra={"fields": {...}})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for k, v in fields.items():
                if k not in base:
                    base[k] = v

        return json.dumps(base, ensure_ascii=False)


def configure_json_logging(*, level: int = logging.INFO) -> None:
    """Configure process-wide JSON logging.

    Keep it simple: a single StreamHandler with JsonFormatter.
    """

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        lg = logging.getLogger(name)
        lg.handlers = [handler]
        lg.setLevel(level)
        lg.propagate = False


def log_http_request(
    *,
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client: str | None,
) -> None:
    logger.info(
        "http_request",
        extra={
            "fields": {
                "method": method,
                "path": path,
                "status_code": int(status_code),
                "duration_ms": float(duration_ms),
                "client": client,
            }
        },
    )

