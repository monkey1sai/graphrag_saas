"""GraphRAG SaaS backend.

Upgrades the original demo server to FastAPI and adds:
- Dataset ingestion from a folder (including OCR for images)
- Persistent index on disk (chunks + metadata)
- Query endpoint using HierarchicalRetriever + WeightedIntegrator (Plan B)
- Benchmark endpoint to generate 1000 questions and evaluate retrieval latency/recall
- Lightweight "train" endpoint for Plan B weight tuning (no GPU required)

This aligns with the "Deep GraphRAG DW-GRPO" plan's platform integration section,
while keeping the implementation dependency-light.
"""

from __future__ import annotations

import os
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphrag_platform.benchmark import (
    QAItem,
    evaluate_retriever,
    generate_questions_stratified,
    sha256_file,
    write_metrics_csv,
    write_questions_jsonl,
    write_report,
)
from graphrag_platform.hierarchical_retriever import HierarchicalRetriever
from graphrag_platform.index_store import IndexStore
from graphrag_platform.ingest import ingest_dataset
from graphrag_platform.job_store import JobStore
from graphrag_platform.rl.dwgrpo_scheduler import DWGRPOWeightScheduler
from graphrag_platform.rl.ppo_runner import PPORunner
from graphrag_platform.rl.rewards import compute_reward
from graphrag_platform.rl.types import PromptContext, RewardWeights
from graphrag_platform.sft_trainer import SFTTrainer
from graphrag_platform.teacher import LocalTeacher, OllamaTeacher
from graphrag_platform.weighted_integrator import WeightedIntegrator
from graphrag_platform.model_registry import ModelRegistry
from graphrag_platform.logging_utils import (
    REQUEST_ID_HEADER,
    configure_json_logging,
    get_request_id,
    log_http_request,
    new_request_id,
    set_request_id,
)
from graphrag_platform.metrics import get_metrics, update_train_job_metrics
from graphrag_platform.metrics import inc_error
from graphrag_platform.error_codes import classify_error

# IRS (Iterative Rejection Sampling) 相關導入
# 延遲 import 避免 Unsloth 預編譯問題
from graphrag_platform.irs import IS_IRS_UNSLOTH
IRS_AVAILABLE = IS_IRS_UNSLOTH  # 只有在 IRS_UNSLOTH 模式才可用
IRSRunner = None  # type: ignore
IRSConfig = None  # type: ignore
DWScorerConfig = None  # type: ignore

def _get_irs_imports():
    """延遲導入 IRS 相關模組"""
    global IRSRunner, IRSConfig, DWScorerConfig
    if IRSRunner is None:
        from graphrag_platform.irs.irs_runner import IRSRunner as _IRSRunner
        from graphrag_platform.irs.irs_runner import IRSConfig as _IRSConfig
        from graphrag_platform.irs.dw_grpo_scorer import DWGRPOConfig as _DWScorerConfig
        IRSRunner = _IRSRunner
        IRSConfig = _IRSConfig
        DWScorerConfig = _DWScorerConfig
    return IRSRunner, IRSConfig, DWScorerConfig


INDEX_DIR = os.getenv("INDEX_DIR", str(Path(__file__).with_name("data") / "index"))
REPORT_DIR = os.getenv("REPORT_DIR", str(Path(__file__).with_name("reports")))
DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", str(Path(__file__).with_name("docs")))
AUTO_INGEST = os.getenv("AUTO_INGEST", "0") == "1"
JOBS_PATH = os.getenv("JOBS_PATH", str(Path(__file__).with_name("data") / "jobs.json"))
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", str(Path(__file__).with_name("data") / "model_registry.json"))
ACTIVE_MODELS_PATH = os.getenv("ACTIVE_MODELS_PATH", str(Path(__file__).with_name("data") / "active_models.json"))

TEACHER_PROVIDER = os.getenv("TEACHER_PROVIDER", "ollama").strip().lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.20.235:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:latest").strip()
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
OLLAMA_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

SFT_SUPERVISED_QUESTIONS_PATH = os.getenv("SFT_SUPERVISED_QUESTIONS_PATH", "").strip()
SFT_PROMPT_ASSET = os.getenv("SFT_PROMPT_ASSET", "/app/eval_docs/prompts/eval_json_v2.md").strip()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class IngestRequest(BaseModel):
    dataset_path: str | None = None


class BenchmarkRequest(BaseModel):
    dataset_path: str | None = None
    n_questions: int = Field(1000, ge=10, le=5000)
    top_k: int = Field(5, ge=1, le=50)
    seed: int = Field(42, ge=0)
    persist_questions: bool = Field(True)
    persist_metrics_csv: bool = Field(True)


class TrainRequest(BaseModel):
    # Platform training interface (Phase 3). SFT is implemented in a minimal CPU-friendly form.
    stage: str = Field("SFT")
    dataset_path: str | None = None
    # Allow small smoke tests (e.g., IRS Phase 4.7.5 uses n_questions=8)
    n_questions: int = Field(200, ge=8, le=2000)
    top_k: int = Field(5, ge=1, le=50)
    seed: int = Field(42, ge=0)

    # SFT knobs (minimal)
    model_id: str = Field("sshleifer/tiny-gpt2")
    max_steps: int = Field(100, ge=1, le=5000)
    batch_size: int = Field(2, ge=1, le=32)
    lr: float = Field(5e-5, gt=0)
    max_length: int = Field(256, ge=32, le=2048)

    # RL knobs (Phase 4 minimal wiring)
    max_new_tokens: int = Field(120, ge=16, le=512)
    min_response_tokens: int = Field(16, ge=0, le=256)
    use_4bit: bool = Field(True)
    lora_r: int = Field(16, ge=2, le=128)
    lora_alpha: int = Field(32, ge=4, le=256)
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5)
    kl_coef: float = Field(0.05, ge=0.0, le=1.0)
    degradation_drop_ratio: float = Field(0.3, ge=0.0, le=0.95)
    degradation_patience: int = Field(3, ge=1, le=50)
    stop_on_degradation: bool = Field(False)
    dwgrpo: bool = Field(True)
    dwgrpo_window: int = Field(20, ge=4, le=200)
    dwgrpo_temperature: float = Field(0.25, gt=0.0, le=5.0)
    dwgrpo_min_weight: float = Field(0.05, gt=0.0, le=0.3)
    dwgrpo_update_every: int = Field(5, ge=1, le=100)

    # IRS knobs (Phase 4.7 - Iterative Rejection Sampling)
    irs_n_iterations: int = Field(3, ge=1, le=20)
    irs_n_candidates: int = Field(4, ge=2, le=16)
    irs_temperature: float = Field(0.7, gt=0.0, le=2.0)
    irs_dw_temperature: float = Field(2.0, gt=0.0, le=10.0)
    irs_dw_momentum: float = Field(0.8, ge=0.0, le=1.0)
    irs_dw_window: int = Field(5, ge=2, le=50)


class ActivateModelRequest(BaseModel):
    stage: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)


@dataclass
class JobState:
    job_id: str
    kind: str
    state: str
    created_at: str
    updated_at: str
    detail: dict[str, Any]


class JobManager:
    def __init__(self, *, store: JobStore) -> None:
        self._lock = threading.Lock()
        self._store = store
        self._jobs: dict[str, JobState] = {}
        for record in store.load_all():
            try:
                job = JobState(
                    job_id=str(record.get("job_id")),
                    kind=str(record.get("kind")),
                    state=str(record.get("state")),
                    created_at=str(record.get("created_at")),
                    updated_at=str(record.get("updated_at")),
                    detail=dict(record.get("detail") or {}),
                )
                self._jobs[job.job_id] = job
            except Exception:
                # Skip corrupted records
                continue

    def create(self, kind: str, detail: dict[str, Any] | None = None) -> str:
        now = datetime.now(timezone.utc).isoformat()
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        with self._lock:
            job = JobState(
                job_id=job_id,
                kind=kind,
                state="running",
                created_at=now,
                updated_at=now,
                detail=detail or {},
            )
            self._jobs[job_id] = job
            self._store.upsert(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "state": job.state,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "detail": job.detail,
                }
            )
        return job_id

    def update(self, job_id: str, *, state: str | None = None, detail: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if state is not None:
                job.state = state
            if detail is not None:
                job.detail = detail
            job.updated_at = now
            self._store.upsert(
                {
                    "job_id": job.job_id,
                    "kind": job.kind,
                    "state": job.state,
                    "created_at": job.created_at,
                    "updated_at": job.updated_at,
                    "detail": job.detail,
                }
            )

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, *, kind: str | None = None) -> list[JobState]:
        with self._lock:
            values = list(self._jobs.values())
        if kind is None:
            return values
        return [j for j in values if j.kind == kind]


class KnowledgeBase:
    def __init__(self, *, index_dir: str) -> None:
        self._lock = threading.Lock()
        self.index_dir = index_dir
        self.store = IndexStore(Path(index_dir))
        self.chunks: list[str] = []
        self.chunk_meta: list[dict[str, Any]] = []
        self.retriever: HierarchicalRetriever | None = None
        self.integrator = WeightedIntegrator()
        self.meta: dict[str, Any] | None = None

    def is_ready(self) -> bool:
        with self._lock:
            return bool(self.retriever and self.chunks)

    def reload_from_disk(self) -> None:
        with self._lock:
            if not self.store.has_index():
                self.chunks = []
                self.chunk_meta = []
                self.retriever = None
                self.meta = None
                return

            meta = self.store.read_meta()
            chunks = self.store.load_chunks()
            entity_names = self.store.load_entity_names()

            self.chunk_meta = chunks
            self.chunks = [c.get("text", "") for c in chunks]
            self.retriever = HierarchicalRetriever(self.chunks, entity_names)
            self.meta = meta.__dict__ if meta else None

    def answer(self, question: str, *, top_k: int) -> dict[str, Any]:
        with self._lock:
            if not self.retriever:
                raise RuntimeError("Knowledge base not initialised. Run /ingest first.")
            ranked = self.retriever.retrieve(question, top_n=top_k)
            answer = self.integrator.integrate(question, ranked, self.chunks)
            sources: list[dict[str, Any]] = []
            for idx, score in ranked:
                if 0 <= idx < len(self.chunk_meta):
                    sources.append(
                        {
                            "chunk_id": idx,
                            "score": float(score),
                            "source_path": self.chunk_meta[idx].get("source_path"),
                        }
                    )
            return {"answer": answer, "sources": sources}


app = FastAPI(title="GraphRAG SaaS", version="0.1.0")
jobs = JobManager(store=JobStore(Path(JOBS_PATH)))
model_registry = ModelRegistry(registry_path=Path(MODEL_REGISTRY_PATH), active_path=Path(ACTIVE_MODELS_PATH))
kb = KnowledgeBase(index_dir=INDEX_DIR)
logger = logging.getLogger("graphrag_saas")


@app.on_event("startup")
def _startup() -> None:
    configure_json_logging(level=logging.INFO)
    kb.reload_from_disk()
    if AUTO_INGEST and not kb.is_ready():
        try:
            ingest_dataset(DEFAULT_DATASET_PATH, index_dir=INDEX_DIR)
        except Exception:
            # Avoid crashing startup due to missing OCR deps / dataset.
            pass
        kb.reload_from_disk()


@app.middleware("http")
async def request_id_middleware(request, call_next):  # type: ignore[no-untyped-def]
    rid = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
    set_request_id(rid)
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "http_exception",
            extra={
                "fields": {
                    "method": request.method,
                    "path": request.url.path,
                }
            },
        )
        raise
    finally:
        duration_ms = (time.time() - start) * 1000.0
        try:
            client = None
            if getattr(request, "client", None) is not None:
                client = getattr(request.client, "host", None)
            status_code = int(getattr(locals().get("response"), "status_code", 500))
            log_http_request(
                logger=logger,
                method=str(request.method),
                path=str(request.url.path),
                status_code=status_code,
                duration_ms=float(duration_ms),
                client=client,
            )
        except Exception:
            pass

    # echo back for clients & downstream logs
    response.headers[REQUEST_ID_HEADER] = get_request_id() or rid
    return response


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "kb_ready": kb.is_ready(),
        "index_meta": kb.meta,
    }


@app.get("/api/models")
def list_models(stage: str | None = None) -> dict[str, Any]:
    st = (stage or "").strip().upper() or None
    if st is not None and st not in {"SFT", "RL", "IRS"}:
        raise HTTPException(status_code=400, detail="stage must be one of: SFT, RL, IRS")
    models = model_registry.list(stage=st)  # type: ignore[arg-type]
    return {"models": models, "count": len(models)}


@app.get("/api/models/active")
def get_active_models() -> dict[str, Any]:
    active = model_registry.get_active_ids()
    detail: dict[str, Any] = {}
    for st, mid in active.items():
        detail[st] = model_registry.get(mid) if mid else None
    return {"active": active, "detail": detail}


@app.get("/metrics")
def metrics() -> Any:
    body, content_type = get_metrics().render()
    # Return raw bytes to avoid JSON encoding.
    from fastapi.responses import Response

    return Response(content=body, media_type=content_type)


@app.post("/api/models/activate")
def activate_model(req: ActivateModelRequest) -> dict[str, Any]:
    st = (req.stage or "").strip().upper()
    if st not in {"SFT", "RL", "IRS"}:
        raise HTTPException(status_code=400, detail="stage must be one of: SFT, RL, IRS")
    try:
        active = model_registry.set_active(stage=st, model_id=req.model_id)  # type: ignore[arg-type]
    except ValueError as e:
        coded = classify_error(e=e, stage="models")
        inc_error(stage=coded.stage, code=coded.code)
        raise HTTPException(status_code=400, detail={"error_code": coded.code, "message": coded.message})
    return {"active": active}


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    try:
        return kb.answer(req.question, top_k=req.top_k)
    except RuntimeError as e:
        coded = classify_error(e=e, stage="query")
        inc_error(stage=coded.stage, code=coded.code)
        raise HTTPException(status_code=409, detail={"error_code": coded.code, "message": coded.message})


@app.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    dataset_path = req.dataset_path or DEFAULT_DATASET_PATH
    job_id = jobs.create("ingest", detail={"dataset_path": dataset_path})

    def _run() -> None:
        try:
            result = ingest_dataset(dataset_path, index_dir=INDEX_DIR)
            kb.reload_from_disk()
            jobs.update(
                job_id,
                state="succeeded",
                detail={
                    "meta": result.meta.__dict__,
                    "warnings": result.warnings,
                },
            )
        except Exception as e:
            coded = classify_error(e=e, stage="ingest")
            inc_error(stage=coded.stage, code=coded.code)
            jobs.update(job_id, state="failed", detail={"error_code": coded.code, "error": coded.message})

    background_tasks.add_task(_run)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "kind": job.kind,
        "state": job.state,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "detail": job.detail,
    }


@app.post("/benchmark")
def benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    dataset_path = req.dataset_path or DEFAULT_DATASET_PATH
    job_id = jobs.create(
        "benchmark",
        detail={
            "dataset_path": dataset_path,
            "n_questions": req.n_questions,
            "top_k": req.top_k,
            "seed": req.seed,
        },
    )

    def _run() -> None:
        try:
            if not kb.is_ready():
                # Ensure index exists for the requested dataset.
                ingest_dataset(dataset_path, index_dir=INDEX_DIR)
                kb.reload_from_disk()
            if not kb.is_ready():
                raise RuntimeError("Knowledge base not initialised.")

            questions: list[QAItem] = generate_questions_stratified(
                kb.chunks,
                n_questions=req.n_questions,
                seed=req.seed,
            )

            def retrieve_fn(q: str, top_k: int) -> list[tuple[int, float]]:
                assert kb.retriever is not None
                return kb.retriever.retrieve(q, top_n=top_k)

            metrics = evaluate_retriever(questions=questions, retrieve_fn=retrieve_fn, top_k=req.top_k)

            report_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            questions_path = None
            questions_sha256 = None
            if req.persist_questions:
                questions_path = write_questions_jsonl(REPORT_DIR, report_id=report_id, questions=questions)
                questions_sha256 = sha256_file(questions_path)

            metrics_csv_path = None
            metrics_csv_sha256 = None
            if req.persist_metrics_csv:
                metrics_csv_path = write_metrics_csv(REPORT_DIR, report_id=report_id, metrics=metrics)
                metrics_csv_sha256 = sha256_file(metrics_csv_path)

            report_payload: dict[str, Any] = {
                "dataset_path": dataset_path,
                "index_meta": kb.meta,
                "metrics": metrics,
                "question_generation": {
                    "mode": "stratified",
                    "seed": req.seed,
                    "n_questions": req.n_questions,
                },
                "artifacts": {
                    "questions_jsonl": {
                        "path": questions_path,
                        "sha256": questions_sha256,
                    },
                    "metrics_csv": {
                        "path": metrics_csv_path,
                        "sha256": metrics_csv_sha256,
                    },
                },
            }

            report_path = write_report(REPORT_DIR, payload=report_payload, report_id=report_id)
            jobs.update(
                job_id,
                state="succeeded",
                detail={
                    "metrics": metrics,
                    "report_path": report_path,
                    "questions_path": questions_path,
                    "metrics_csv_path": metrics_csv_path,
                },
            )
        except Exception as e:
            coded = classify_error(e=e, stage="benchmark")
            inc_error(stage=coded.stage, code=coded.code)
            jobs.update(job_id, state="failed", detail={"error_code": coded.code, "error": coded.message})

    background_tasks.add_task(_run)
    return {"job_id": job_id}


def _coverage_ratio(query: str, answer: str) -> float:
    import re

    q_tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2}", query)
    if not q_tokens:
        return 0.0
    a = answer or ""
    hit = sum(1 for t in q_tokens if t in a)
    return hit / max(1, len(q_tokens))


@app.post("/api/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    dataset_path = req.dataset_path or DEFAULT_DATASET_PATH
    stage = (req.stage or "").upper()
    job_id = jobs.create(
        "train",
        detail={
            "stage": stage,
            "dataset_path": dataset_path,
            "stop_requested": False,
            "epoch": 0,
            "step": 0,
            "loss": None,
            "avg_rewards": None,
            "gpu_mem": None,
            "eta_seconds": None,
        },
    )

    def _run() -> None:
        try:
            if stage not in {"SFT", "RL", "IRS", "WEIGHT_TUNE", "WT", "PLAN_B"}:
                raise RuntimeError("Unsupported stage. Use SFT, RL, IRS, or WEIGHT_TUNE.")

            if not kb.is_ready():
                ingest_dataset(dataset_path, index_dir=INDEX_DIR)
                kb.reload_from_disk()

            # ======= IRS Stage (Iterative Rejection Sampling) =======
            if stage == "IRS":
                if not IRS_AVAILABLE:
                    raise RuntimeError(
                        "IRS module not available. Run in Unsloth container with "
                        "BACKEND_MODE=IRS_UNSLOTH."
                    )
                if not IS_IRS_UNSLOTH:
                    raise RuntimeError(
                        "IRS requires BACKEND_MODE=IRS_UNSLOTH. Current mode does not support IRS."
                    )
                if not kb.is_ready():
                    raise RuntimeError("Knowledge base not initialised.")

                assert kb.retriever is not None

                # 生成問題
                questions: list[QAItem] = generate_questions_stratified(
                    kb.chunks,
                    n_questions=req.n_questions,
                    seed=req.seed,
                )

                # 準備 IRS 訓練資料
                irs_questions: list[dict[str, str]] = []
                for it in questions:
                    ranked = kb.retriever.retrieve(it.question, top_n=req.top_k)
                    ctx_texts: list[str] = []
                    for idx, _score in ranked:
                        if 0 <= idx < len(kb.chunks):
                            ctx_texts.append(kb.chunks[idx])
                    context_text = "\n\n".join(ctx_texts[: req.top_k])
                    irs_questions.append({"question": it.question, "context": context_text})

                # 延遲導入 IRS 模組（避免 Unsloth 預編譯問題）
                _IRSRunner, _IRSConfig, _DWScorerConfig = _get_irs_imports()

                # DW-GRPO Scorer 配置
                dw_config = _DWScorerConfig(
                    device=os.getenv("REWARD_DEVICE", "cpu"),
                    history_window=req.irs_dw_window,
                    temperature=req.irs_dw_temperature,
                    momentum=req.irs_dw_momentum,
                )

                # IRS Runner 配置
                irs_config = _IRSConfig(
                    base_model=req.model_id,
                    n_candidates=req.irs_n_candidates,
                    n_iterations=req.irs_n_iterations,
                    temperature=req.irs_temperature,
                    max_new_tokens=req.max_new_tokens,
                    lora_r=req.lora_r,
                    lora_alpha=req.lora_alpha,
                    dw_config=dw_config,
                    job_id=job_id,
                    output_dir=str(Path(REPORT_DIR).parent / "data" / "jobs"),
                )

                # 狀態回調
                def status_callback(status: dict) -> None:
                    current_job = jobs.get(job_id)
                    current_detail = dict(current_job.detail) if current_job else {}
                    jobs.update(
                        job_id,
                        detail={
                            **current_detail,
                            "stage": "IRS",
                            "current_iteration": status.get("iteration"),
                            "current_phase": status.get("phase"),
                            "phase_progress": status.get("progress"),
                            "reward_breakdown": status.get("reward_breakdown"),
                            "current_weights": status.get("current_weights"),
                        },
                    )

                runner = _IRSRunner(config=irs_config, status_callback=status_callback)
                summary = runner.run(irs_questions)

                jobs.update(
                    job_id,
                    state="succeeded",
                    detail={
                        "stage": "IRS",
                        "summary": summary,
                        "n_iterations": req.irs_n_iterations,
                        "n_questions": req.n_questions,
                        "final_weights": summary.get("final_weights"),
                        "final_checkpoint": summary.get("final_checkpoint"),
                    },
                )
                try:
                    model_registry.register(
                        stage="IRS",
                        base_model=req.model_id,
                        job_id=job_id,
                        artifacts={
                            "job_dir": str(Path(REPORT_DIR).parent / "data" / "jobs" / f"train_{job_id}"),
                            "summary_json": str(Path(REPORT_DIR).parent / "data" / "jobs" / f"train_{job_id}" / "summary.json"),
                            "final_checkpoint": str(summary.get("final_checkpoint") or ""),
                        },
                        metrics={
                            "n_iterations": int(req.irs_n_iterations),
                            "n_questions": int(req.n_questions),
                            "final_weights": summary.get("final_weights"),
                        },
                        runtime={"backend_mode": os.getenv("BACKEND_MODE", "")},
                        tags=["auto-registered"],
                    )
                except Exception:
                    pass
                return

            if stage == "SFT":
                # Minimal SFT: build prompt->answer pairs using the existing Plan-B integrator
                if not kb.is_ready():
                    raise RuntimeError("Knowledge base not initialised.")

                assert kb.retriever is not None
                questions: list[QAItem]

                def retrieve_fn(q: str, top_k: int) -> list[tuple[int, float]]:
                    assert kb.retriever is not None
                    return kb.retriever.retrieve(q, top_n=top_k)

                def answer_fn(q: str, ranked: list[tuple[int, float]]) -> str:
                    return kb.integrator.integrate(q, ranked, kb.chunks)

                teacher = None
                if SFT_SUPERVISED_QUESTIONS_PATH:
                    import json
                    import random

                    from graphrag_platform.teacher.supervised_eval_json_v2 import SupervisedEvalJsonV2Teacher

                    q_path = Path(SFT_SUPERVISED_QUESTIONS_PATH)
                    if not q_path.exists():
                        raise RuntimeError(f"SFT_SUPERVISED_QUESTIONS_PATH not found: {SFT_SUPERVISED_QUESTIONS_PATH}")
                    items = json.loads(q_path.read_text(encoding="utf-8"))
                    if not isinstance(items, list):
                        raise RuntimeError("Supervised question set must be a JSON array.")

                    rng = random.Random(int(req.seed))
                    rng.shuffle(items)
                    if int(req.n_questions) > 0:
                        items = items[: int(req.n_questions)]

                    expected_by_question: dict[str, dict[str, Any]] = {}
                    q_list: list[QAItem] = []
                    for i, it in enumerate(items):
                        if not isinstance(it, dict):
                            continue
                        q = str(it.get("question") or "").strip()
                        if not q:
                            continue
                        expected = it.get("expected") if isinstance(it.get("expected"), dict) else {}
                        expected_by_question[q] = expected
                        q_list.append(QAItem(qid=i, question=q, gold_chunk_id=0, kind="supervised"))

                    prompt_text = Path(SFT_PROMPT_ASSET).read_text(encoding="utf-8")
                    sys_marker = "# 指令（System）"
                    user_marker = "# 輸入（User）"
                    si = prompt_text.find(sys_marker)
                    ui = prompt_text.find(user_marker)
                    if si >= 0 and ui > si:
                        system_text = prompt_text[si + len(sys_marker) : ui].strip()
                        user_template = prompt_text[ui + len(user_marker) :].strip()
                    else:
                        system_text = ""
                        user_template = prompt_text.strip()
                    teacher = SupervisedEvalJsonV2Teacher(
                        system_text=system_text,
                        user_template=user_template,
                        expected_by_question=expected_by_question,
                        retrieve_fn=retrieve_fn,
                        chunks=kb.chunks,
                        chunk_meta=kb.chunk_meta,
                    )
                    questions = q_list
                else:
                    questions = generate_questions_stratified(
                        kb.chunks,
                        n_questions=req.n_questions,
                        seed=req.seed,
                    )

                if teacher is None:
                    if TEACHER_PROVIDER in {"local", ""}:
                        teacher = LocalTeacher(retrieve_fn=retrieve_fn, answer_fn=answer_fn, chunk_meta=kb.chunk_meta)
                    elif TEACHER_PROVIDER in {"ollama"}:
                        teacher = OllamaTeacher(
                            base_url=OLLAMA_BASE_URL,
                            model=OLLAMA_MODEL,
                            retrieve_fn=retrieve_fn,
                            chunks=kb.chunks,
                            chunk_meta=kb.chunk_meta,
                            timeout_seconds=OLLAMA_TIMEOUT_SECONDS,
                            max_retries=OLLAMA_MAX_RETRIES,
                        )
                    else:
                        raise RuntimeError(
                            "Unsupported TEACHER_PROVIDER. Use 'local' or 'ollama'. "
                            f"Got: {TEACHER_PROVIDER!r}"
                        )

                trainer = SFTTrainer(jobs=jobs, report_dir=Path(REPORT_DIR))
                result = trainer.run(
                    job_id=job_id,
                    stage=stage,
                    model_id=req.model_id,
                    questions=questions,
                    retrieve_fn=retrieve_fn,
                    answer_fn=answer_fn,
                    chunks=kb.chunks,
                    chunk_meta=kb.chunk_meta,
                    teacher=teacher,
                    top_k=req.top_k,
                    seed=req.seed,
                    max_steps=req.max_steps,
                    batch_size=req.batch_size,
                    lr=req.lr,
                    max_length=req.max_length,
                    use_4bit=req.use_4bit,
                    lora_r=req.lora_r,
                    lora_alpha=req.lora_alpha,
                    lora_dropout=req.lora_dropout,
                )
                try:
                    model_registry.register(
                        stage="SFT",
                        base_model=req.model_id,
                        job_id=job_id,
                        artifacts={
                            "sft_summary_path": str(Path(REPORT_DIR) / f"train_{job_id}_summary.json"),
                            "sft_dataset_path": str(Path(REPORT_DIR) / f"sft_dataset_{job_id}.json"),
                            "sft_adapter_path": str(result.artifacts.get("sft_adapter_path") or ""),
                        },
                        metrics={
                            "steps": int(result.steps),
                            "final_loss": result.final_loss,
                            "n_questions": int(req.n_questions),
                            "top_k": int(req.top_k),
                        },
                        runtime=dict(result.artifacts.get("runtime") or {}),
                        tags=["auto-registered"],
                    )
                except Exception:
                    pass
                return

            if stage == "RL":
                if not kb.is_ready():
                    raise RuntimeError("Knowledge base not initialised.")

                assert kb.retriever is not None
                questions: list[QAItem] = generate_questions_stratified(
                    kb.chunks,
                    n_questions=req.n_questions,
                    seed=req.seed,
                )

                items: list[PromptContext] = []
                for it in questions:
                    ranked = kb.retriever.retrieve(it.question, top_n=req.top_k)
                    ctx_texts: list[str] = []
                    scores: list[float] = []
                    for idx, score in ranked:
                        if 0 <= idx < len(kb.chunks):
                            ctx_texts.append(kb.chunks[idx])
                            scores.append(float(score))
                    context_text = "\n\n".join(ctx_texts[: req.top_k])
                    rel_signal = (sum(scores) / len(scores)) if scores else 0.0

                    prompt = (
                        "You are a helpful assistant. Answer the question using only the provided context.\n\n"
                        f"Question: {it.question}\n\n"
                        f"Context:\n{context_text}\n\n"
                        "Answer:"
                    )

                    items.append(
                        PromptContext(
                            question=it.question,
                            prompt=prompt,
                            context_text=context_text,
                            rel_signal=float(rel_signal),
                        )
                    )

                base_weights = RewardWeights(w_rel=0.4, w_faith=0.3, w_conc=0.3)

                scheduler = None
                if req.dwgrpo:
                    scheduler = DWGRPOWeightScheduler(
                        window=int(req.dwgrpo_window),
                        temperature=float(req.dwgrpo_temperature),
                        min_weight=float(req.dwgrpo_min_weight),
                    )

                runner = PPORunner(jobs=jobs, report_dir=Path(REPORT_DIR))
                artifacts = runner.run(
                    job_id=job_id,
                    model_id=req.model_id,
                    items=items,
                    reward_fn=lambda item, response, weights: compute_reward(
                        item=item,
                        response=response,
                        weights=weights,
                    ),
                    base_weights=base_weights,
                    weight_scheduler=scheduler,
                    weight_update_every=int(req.dwgrpo_update_every),
                    steps=req.max_steps,
                    batch_size=req.batch_size,
                    max_new_tokens=req.max_new_tokens,
                    min_response_tokens=req.min_response_tokens,
                    seed=req.seed,
                    use_4bit=req.use_4bit,
                    lora_r=req.lora_r,
                    lora_alpha=req.lora_alpha,
                    lora_dropout=req.lora_dropout,
                    kl_coef=req.kl_coef,
                    degradation_drop_ratio=req.degradation_drop_ratio,
                    degradation_patience=req.degradation_patience,
                    stop_on_degradation=req.stop_on_degradation,
                )
                try:
                    job = jobs.get(job_id)
                    detail = dict(job.detail or {}) if job else {}
                    runtime = dict(detail.get("runtime") or {}) if isinstance(detail.get("runtime"), dict) else {}
                    model_registry.register(
                        stage="RL",
                        base_model=req.model_id,
                        job_id=job_id,
                        artifacts={
                            "rl_summary_path": artifacts.summary_path,
                            "adapter_path": artifacts.adapter_path,
                            "rl_metrics_path": str(Path(REPORT_DIR) / f"train_{job_id}_rl_metrics.jsonl"),
                        },
                        metrics={
                            "final_avg_reward": detail.get("avg_rewards"),
                            "steps": detail.get("step"),
                            "anti_cheat": {
                                "min_response_tokens": int(req.min_response_tokens),
                                "degradation_drop_ratio": float(req.degradation_drop_ratio),
                                "degradation_patience": int(req.degradation_patience),
                                "stop_on_degradation": bool(req.stop_on_degradation),
                            },
                        },
                        runtime=runtime,
                        tags=["auto-registered"],
                    )
                except Exception:
                    pass
                return

            # WEIGHT_TUNE (Plan B)
            questions = generate_questions_stratified(kb.chunks, n_questions=req.n_questions, seed=req.seed)
            assert kb.retriever is not None

            # Small grid search over weights (sum ~= 1.0)
            candidates = [
                (0.6, 0.3, 0.1),
                (0.5, 0.3, 0.2),
                (0.45, 0.35, 0.2),
                (0.4, 0.4, 0.2),
                (0.4, 0.3, 0.3),
                (0.35, 0.45, 0.2),
            ]

            best = None
            for w_rel, w_faith, w_conc in candidates:
                kb.integrator.w_rel = w_rel
                kb.integrator.w_faith = w_faith
                kb.integrator.w_conc = w_conc
                total = 0.0
                for it in questions:
                    ranked = kb.retriever.retrieve(it.question, top_n=req.top_k)
                    ans = kb.integrator.integrate(it.question, ranked, kb.chunks)
                    cov = _coverage_ratio(it.question, ans)
                    length_pen = min(1.0, len(ans) / 200.0)
                    total += (2.0 * cov) - (0.5 * length_pen)
                score = total / max(1, len(questions))
                if best is None or score > best[0]:
                    best = (score, (w_rel, w_faith, w_conc))

            assert best is not None
            _score, (w_rel, w_faith, w_conc) = best
            kb.integrator.w_rel = w_rel
            kb.integrator.w_faith = w_faith
            kb.integrator.w_conc = w_conc

            jobs.update(
                job_id,
                state="succeeded",
                detail={
                    "stage": stage,
                    "selected_weights": {"w_rel": w_rel, "w_faith": w_faith, "w_conc": w_conc},
                    "objective": "2*coverage - 0.5*length_penalty (heuristic)",
                    "epoch": 0,
                    "step": len(questions),
                    "loss": None,
                    "avg_rewards": None,
                    "gpu_mem": None,
                    "eta_seconds": None,
                },
            )
        except Exception as e:
            coded = classify_error(e=e, stage="train")
            inc_error(stage=coded.stage, code=coded.code)
            jobs.update(job_id, state="failed", detail={"error_code": coded.code, "error": coded.message})

    background_tasks.add_task(_run)
    return {"job_id": job_id}


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job or job.kind != "train":
        raise HTTPException(status_code=404, detail="Train job not found")
    update_train_job_metrics(job=job)
    d = job.detail or {}
    return {
        "job_id": job.job_id,
        "stage": d.get("stage"),
        "state": job.state,
        "epoch": d.get("epoch"),
        "step": d.get("step"),
        "loss": d.get("loss"),
        "avg_rewards": d.get("avg_rewards"),
        "gpu_mem": d.get("gpu_mem"),
        "eta_seconds": d.get("eta_seconds"),
        "stop_requested": d.get("stop_requested", False),
        "detail": d,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@app.post("/api/train/stop/{job_id}")
def train_stop(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job or job.kind != "train":
        raise HTTPException(status_code=404, detail="Train job not found")
    detail = dict(job.detail or {})
    detail["stop_requested"] = True
    # If currently running, mark as stopping; trainer will flip to stopped.
    state = job.state
    if state == "running":
        state = "stopping"
    jobs.update(job_id, state=state, detail=detail)
    return {"job_id": job_id, "state": state, "stop_requested": True}


@app.get("/api/train/{job_id}/metrics")
def train_metrics(job_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """取得 IRS 訓練的 metrics.jsonl 內容"""
    import json

    job = jobs.get(job_id)
    if not job or job.kind != "train":
        raise HTTPException(status_code=404, detail="Train job not found")

    # 找到 job 目錄
    job_dir = Path(REPORT_DIR).parent / "data" / "jobs" / f"train_{job_id}"
    metrics_path = job_dir / "metrics.jsonl"

    if not metrics_path.exists():
        return {
            "job_id": job_id,
            "metrics": [],
            "total": 0,
            "message": "No metrics file found (job may not be IRS stage or not started yet)",
        }

    # 讀取 JSONL
    entries: list[dict[str, Any]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total = len(entries)
    paginated = entries[offset : offset + limit]

    return {
        "job_id": job_id,
        "metrics": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/train/{job_id}/samples")
def train_samples(job_id: str, limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """取得 IRS 訓練的 samples.jsonl 內容（最佳答案記錄）"""
    import json

    job = jobs.get(job_id)
    if not job or job.kind != "train":
        raise HTTPException(status_code=404, detail="Train job not found")

    job_dir = Path(REPORT_DIR).parent / "data" / "jobs" / f"train_{job_id}"
    samples_path = job_dir / "samples.jsonl"

    if not samples_path.exists():
        return {
            "job_id": job_id,
            "samples": [],
            "total": 0,
            "message": "No samples file found",
        }

    entries: list[dict[str, Any]] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total = len(entries)
    paginated = entries[offset : offset + limit]

    return {
        "job_id": job_id,
        "samples": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
    }
