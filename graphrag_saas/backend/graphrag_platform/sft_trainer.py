from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .benchmark import QAItem
from .sft_dataset_builder import build_sft_dataset, write_dataset_json
from .teacher.base import Teacher
from .teacher.local import LocalTeacher


@dataclass
class SFTResult:
    steps: int
    final_loss: float | None
    artifacts: dict[str, Any]


class SFTTrainer:
    """Minimal supervised fine-tuning runner.

    - Uses an existing retrieval + integrator as a teacher to build (prompt, response) pairs.
    - Runs a tiny causal LM training loop with PyTorch/Transformers.
    - Periodically updates JobManager status and supports cooperative stopping.

    Notes:
    - This is intended for Phase 3 platform validation, not for production-grade training.
    - If `torch/transformers` are missing, a clear error is raised at runtime.
    """

    def __init__(self, *, jobs: Any, report_dir: Path) -> None:
        self._jobs = jobs
        self._report_dir = report_dir
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        job_id: str,
        stage: str,
        model_id: str,
        questions: list[QAItem],
        retrieve_fn: Callable[[str, int], list[tuple[int, float]]],
        answer_fn: Callable[[str, list[tuple[int, float]]], str],
        chunks: list[str],
        chunk_meta: list[dict[str, Any]] | None = None,
        teacher: Teacher | None = None,
        top_k: int,
        seed: int,
        max_steps: int,
        batch_size: int,
        lr: float,
        max_length: int,
        use_4bit: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> SFTResult:
        try:
            import random

            import torch  # pyright: ignore[reportMissingImports]
            from torch.utils.data import DataLoader  # pyright: ignore[reportMissingImports]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

            try:
                from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
            except Exception:
                BitsAndBytesConfig = None  # type: ignore[assignment]
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SFT requires 'torch' and 'transformers'. Install them in the backend image to enable Phase 3 SFT. "
                f"Original error: {e}"
            )

        random.seed(seed)
        torch.manual_seed(seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_4bit and device.type != "cuda":
            raise RuntimeError("use_4bit is only supported on CUDA devices in this build.")

        def _trust_remote_code(mid: str) -> bool:
            s = (mid or "").lower()
            return ("qwen3" in s) or ("qwen-3" in s)

        # Build a small supervised dataset from teacher outputs (strict schema).
        teacher_impl: Teacher = teacher or LocalTeacher(retrieve_fn=retrieve_fn, answer_fn=answer_fn)

        # Phase 4.1: Persist <Q,K,C> + chunk text/metadata in a schema-friendly JSON file.
        dataset_artifacts = None
        rows = []
        dataset_build_error: str | None = None
        try:
            def _progress(i: int, total: int) -> None:
                # Update progress during teacher generation (can take time).
                if self._should_stop(job_id):
                    raise StopIteration()
                self._jobs.update(
                    job_id,
                    state="running",
                    detail=self._merge_detail(
                        job_id,
                        {
                            "stage": stage,
                            "phase": "build_sft_dataset",
                            "dataset_progress": {"current": int(i), "total": int(total)},
                        },
                    ),
                )

            rows = build_sft_dataset(
                job_id=job_id,
                questions=questions,
                retrieve_fn=retrieve_fn,
                chunks=chunks,
                chunk_meta=chunk_meta,
                teacher=teacher_impl,
                top_k=top_k,
                meta={"stage": stage, "model_id": model_id},
                progress_cb=_progress,
                should_stop=lambda: self._should_stop(job_id),
            )
            dataset_artifacts = write_dataset_json(report_dir=self._report_dir, job_id=job_id, rows=rows)
        except StopIteration:
            self._jobs.update(job_id, state="stopped", detail=self._merge_detail(job_id, {"stop_requested": True}))
            return SFTResult(steps=0, final_loss=None, artifacts={"device": str(device), "stopped": True})
        except Exception as e:
            dataset_artifacts = None
            dataset_build_error = str(e)
            # Keep training unblocked, but record why dataset persistence failed.
            try:
                self._jobs.update(
                    job_id,
                    state="running",
                    detail=self._merge_detail(
                        job_id,
                        {
                            "stage": stage,
                            "phase": "build_sft_dataset_failed",
                            "dataset_build_error": dataset_build_error,
                        },
                    ),
                )
            except Exception:
                pass

        train_records: list[dict[str, Any]] = []
        if dataset_artifacts is not None:
            for r in rows:
                train_records.append(
                    {
                        "prompt": str(r.prompt or ""),
                        "completion": str(r.completion or ""),
                        "teacher": dict(r.teacher or {}),
                    }
                )
        else:
            total = len(questions)
            for i, it in enumerate(questions, start=1):
                if self._should_stop(job_id):
                    self._jobs.update(job_id, state="stopped", detail=self._merge_detail(job_id, {"stop_requested": True}))
                    return SFTResult(steps=0, final_loss=None, artifacts={"device": str(device), "stopped": True})
                try:
                    self._jobs.update(
                        job_id,
                        state="running",
                        detail=self._merge_detail(
                            job_id,
                            {
                                "stage": stage,
                                "phase": "build_sft_dataset_fallback",
                                "dataset_progress": {"current": int(i), "total": int(total)},
                                "dataset_build_error": dataset_build_error,
                            },
                        ),
                    )
                except Exception:
                    pass
                out = teacher_impl.generate(question=it.question, top_k=top_k)
                train_records.append(
                    {
                        "prompt": str(out.prompt or ""),
                        "completion": str(out.completion or ""),
                        "teacher": dict(out.teacher or {}),
                    }
                )

        # Model/tokenizer loading can be slow; set a phase for visibility.
        try:
            self._jobs.update(
                job_id,
                state="running",
                detail=self._merge_detail(job_id, {"stage": stage, "phase": "prepare_model"}),
            )
        except Exception:
            pass

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=_trust_remote_code(model_id))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _to_chat_sample(system_text: str, user_text: str, assistant_text: str) -> str | None:
            if not hasattr(tokenizer, "apply_chat_template"):
                return None
            try:
                messages = [
                    {"role": "system", "content": (system_text or "").strip()},
                    {"role": "user", "content": (user_text or "").strip()},
                    {"role": "assistant", "content": (assistant_text or "").strip()},
                ]
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)  # type: ignore[attr-defined]
            except Exception:
                return None

        samples: list[str] = []
        for rec in train_records:
            prompt = str(rec.get("prompt") or "")
            completion = str(rec.get("completion") or "")
            tmeta = rec.get("teacher") if isinstance(rec.get("teacher"), dict) else {}

            system_text = ""
            user_text = ""
            assistant_text = ""
            if isinstance(tmeta, dict) and ("user_text" in tmeta or "completion_json" in tmeta):
                system_text = str(tmeta.get("system_text") or "")
                user_text = str(tmeta.get("user_text") or "")
                # Prefer a clean JSON string if provided; otherwise strip the leading space.
                assistant_text = str(tmeta.get("completion_json") or "").strip() or completion.strip()
            else:
                # Fall back to the legacy flat format.
                user_text = prompt
                assistant_text = completion

            chat_sample = _to_chat_sample(system_text, user_text, assistant_text)
            if chat_sample is not None:
                samples.append(chat_sample)
            else:
                samples.append(prompt + completion)

        adapter_path: str | None = None
        runtime: dict[str, Any] = {
            "torch": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "use_4bit_requested": bool(use_4bit),
        }

        if use_4bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError("use_4bit requested but BitsAndBytesConfig is unavailable (transformers/bitsandbytes missing).")

            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # pyright: ignore[reportMissingImports]
            except Exception as e:
                raise RuntimeError(f"Missing PEFT dependency: {e}")

            try:
                import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]

                runtime["bitsandbytes"] = getattr(bnb, "__version__", "unknown")
                runtime["bitsandbytes_import_ok"] = True
            except Exception as e:
                runtime["bitsandbytes"] = None
                runtime["bitsandbytes_import_ok"] = False
                runtime["bitsandbytes_error"] = str(e)

            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=qcfg,
                device_map="auto",
                trust_remote_code=_trust_remote_code(model_id),
            )
            model = prepare_model_for_kbit_training(model)
            lora = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            )
            model = get_peft_model(model, lora)
            model.train()

            adapter_path = str(self._report_dir / f"train_{job_id}_sft_adapter")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=_trust_remote_code(model_id))
            model.to(device)
            model.train()

        # Tokenize once up-front to avoid spending time in the loop.
        enc = tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        dataset = list(zip(input_ids, attention_mask, labels, strict=False))

        def collate(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
            ids = torch.stack([b[0] for b in batch])
            mask = torch.stack([b[1] for b in batch])
            lab = torch.stack([b[2] for b in batch])
            return ids, mask, lab

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
        params = [p for p in model.parameters() if getattr(p, "requires_grad", True)]
        optim = torch.optim.AdamW(params, lr=lr)

        started = time.time()
        last_loss: float | None = None

        # Calculate how many batches represent one "epoch" over our synthetic dataset.
        steps_per_epoch = max(1, math.ceil(len(dataset) / max(1, batch_size)))

        step = 0
        loader_it = iter(loader)
        while step < max_steps:
            try:
                ids, mask, lab = next(loader_it)
            except StopIteration:
                loader_it = iter(loader)
                continue

            if step == 0:
                try:
                    self._jobs.update(
                        job_id,
                        state="running",
                        detail=self._merge_detail(job_id, {"stage": stage, "phase": "train_sft"}),
                    )
                except Exception:
                    pass

            if self._should_stop(job_id):
                self._jobs.update(job_id, state="stopped", detail=self._merge_detail(job_id, {"stop_requested": True}))
                return SFTResult(steps=step, final_loss=last_loss, artifacts={"device": str(device)})

            ids = ids.to(device)
            mask = mask.to(device)
            lab = lab.to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=lab)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            step += 1
            last_loss = float(loss.detach().cpu().item())

            # Best-effort status update each step (tiny model, cheap enough).
            elapsed = max(1e-6, time.time() - started)
            qps = step / elapsed
            eta_seconds = int((max_steps - step) / max(1e-6, qps))
            epoch = step // steps_per_epoch

            gpu_mem = None
            if device.type == "cuda":
                try:
                    gpu_mem = int(torch.cuda.max_memory_allocated(device=device))
                except Exception:
                    gpu_mem = None

            self._jobs.update(
                job_id,
                state="running",
                detail=self._merge_detail(
                    job_id,
                    {
                        "stage": stage,
                        "epoch": int(epoch),
                        "step": int(step),
                        "loss": last_loss,
                        "avg_rewards": None,
                        "gpu_mem": gpu_mem,
                        "eta_seconds": eta_seconds,
                        "runtime": runtime,
                    },
                ),
            )

        if adapter_path is not None:
            try:
                model.save_pretrained(adapter_path)  # type: ignore[attr-defined]
            except Exception:
                adapter_path = None

        # Save a minimal training summary (avoid large checkpoints by default).
        summary_path = str(self._report_dir / f"train_{job_id}_summary.json")
        try:
            import json

            Path(summary_path).write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "stage": stage,
                        "model_id": model_id,
                        "device": str(device),
                        "steps": step,
                        "final_loss": last_loss,
                        "runtime": runtime,
                        "artifacts": {
                            "sft_adapter_path": adapter_path,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            summary_path = None

        self._jobs.update(
            job_id,
            state="succeeded",
            detail=self._merge_detail(
                job_id,
                {
                    "stage": stage,
                    "epoch": int(step // steps_per_epoch),
                    "step": int(step),
                    "loss": last_loss,
                    "eta_seconds": 0,
                    "artifacts": {
                        "summary_path": summary_path,
                        "sft_dataset_path": getattr(dataset_artifacts, "dataset_path", None),
                        "sft_dataset_sha256": getattr(dataset_artifacts, "sha256", None),
                        "sft_dataset_count": getattr(dataset_artifacts, "count", None),
                        "sft_adapter_path": adapter_path,
                    },
                },
            ),
        )

        return SFTResult(
            steps=step,
            final_loss=last_loss,
            artifacts={
                "summary_path": summary_path,
                "device": str(device),
                "runtime": runtime,
                "sft_adapter_path": adapter_path,
            },
        )

    def _should_stop(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job:
            return False
        detail = job.detail or {}
        return bool(detail.get("stop_requested")) or job.state == "stopping"

    def _merge_detail(self, job_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        base = dict(job.detail or {}) if job else {}
        base.update(patch)
        return base
