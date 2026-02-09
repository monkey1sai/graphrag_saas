from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from prometheus_client import Counter


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() in {"1", "true", "True", "yes", "YES"}


@dataclass
class Metrics:
    registry: CollectorRegistry
    errors_total: Counter
    train_job_state: Gauge
    train_job_step: Gauge
    train_job_loss: Gauge
    train_job_avg_reward: Gauge

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


_metrics_singleton: Metrics | None = None


def get_metrics() -> Metrics:
    global _metrics_singleton
    if _metrics_singleton is not None:
        return _metrics_singleton

    # Use a dedicated registry to avoid duplicate collectors if uvicorn reload is used.
    registry = CollectorRegistry(auto_describe=True)

    errors_total = Counter(
        "graphrag_errors_total",
        "Total classified errors",
        ["stage", "code"],
        registry=registry,
    )

    train_job_state = Gauge(
        "graphrag_train_job_state",
        "Train job state (running=1, succeeded=2, failed=-1, stopped=0)",
        ["stage", "job_id"],
        registry=registry,
    )
    train_job_step = Gauge(
        "graphrag_train_job_step",
        "Train job step",
        ["stage", "job_id"],
        registry=registry,
    )
    train_job_loss = Gauge(
        "graphrag_train_job_loss",
        "Train job loss (if available)",
        ["stage", "job_id"],
        registry=registry,
    )
    train_job_avg_reward = Gauge(
        "graphrag_train_job_avg_reward",
        "Train job average reward (if available)",
        ["stage", "job_id"],
        registry=registry,
    )

    _metrics_singleton = Metrics(
        registry=registry,
        errors_total=errors_total,
        train_job_state=train_job_state,
        train_job_step=train_job_step,
        train_job_loss=train_job_loss,
        train_job_avg_reward=train_job_avg_reward,
    )
    return _metrics_singleton


def inc_error(*, stage: str, code: str) -> None:
    try:
        m = get_metrics()
        m.errors_total.labels(stage=str(stage), code=str(code)).inc()
    except Exception:
        return


STATE_MAP: dict[str, int] = {
    "running": 1,
    "succeeded": 2,
    "failed": -1,
    "stopped": 0,
    "stopping": 0,
}


def update_train_job_metrics(*, job: Any) -> None:
    """Best-effort update of gauges from a JobState-like object."""

    try:
        stage = str(getattr(job, "kind", "") or "train")
        detail = getattr(job, "detail", {}) or {}
        if isinstance(detail, dict) and detail.get("stage"):
            stage = str(detail.get("stage"))
        job_id = str(getattr(job, "job_id"))
        state = str(getattr(job, "state", ""))
        step = detail.get("step")
        loss = detail.get("loss")
        avg_rewards = detail.get("avg_rewards")
    except Exception:
        return

    m = get_metrics()
    try:
        m.train_job_state.labels(stage=stage, job_id=job_id).set(float(STATE_MAP.get(state, 0)))
        if step is not None:
            m.train_job_step.labels(stage=stage, job_id=job_id).set(float(step))
        if loss is not None:
            m.train_job_loss.labels(stage=stage, job_id=job_id).set(float(loss))
        if avg_rewards is not None:
            m.train_job_avg_reward.labels(stage=stage, job_id=job_id).set(float(avg_rewards))
    except Exception:
        return
