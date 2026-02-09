from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .rewards import RewardResult
from .types import PromptContext, RewardComponents, RewardWeights


class WeightScheduler(Protocol):
    def push(self, comps: RewardComponents) -> None: ...

    def current_weights(self, base: RewardWeights) -> RewardWeights: ...


@dataclass
class PPOArtifacts:
    summary_path: str | None
    adapter_path: str | None


class PPORunner:
    """TRL PPO runner (Phase 4).

    Notes:
    - This runner is designed for small-scale on-device PPO with QLoRA where available.
    - If TRL/PEFT/bitsandbytes are not installed, a clear error is raised.
    """

    def __init__(self, *, jobs: Any, report_dir: Path) -> None:
        self._jobs = jobs
        self._report_dir = report_dir
        self._report_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        job_id: str,
        model_id: str,
        items: list[PromptContext],
        reward_fn: Callable[[PromptContext, str, RewardWeights], RewardResult],
        base_weights: RewardWeights,
        weight_scheduler: WeightScheduler | None = None,
        weight_update_every: int = 1,
        steps: int,
        batch_size: int,
        max_new_tokens: int,
        min_response_tokens: int,
        seed: int,
        use_4bit: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        kl_coef: float,
        degradation_drop_ratio: float,
        degradation_patience: int,
        stop_on_degradation: bool,
    ) -> PPOArtifacts:
        try:
            import json
            import random

            import torch  # pyright: ignore[reportMissingImports]
            from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

            # Optional bitsandbytes config (only if installed)
            try:
                from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
            except Exception:
                BitsAndBytesConfig = None  # type: ignore[assignment]

            try:
                from peft import LoraConfig  # pyright: ignore[reportMissingImports]
            except Exception as e:
                raise RuntimeError(f"Missing PEFT dependency: {e}")

            try:
                from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer  # pyright: ignore[reportMissingImports]
            except Exception as e:
                raise RuntimeError(
                    "RL/PPO requires TRL. Build the backend image with INSTALL_RL_DEPS=1 (see docs) and ensure GPU deps are available. "
                    f"Original error: {e}"
                )
        except Exception as e:
            raise RuntimeError(str(e))

        random.seed(seed)
        torch.manual_seed(seed)

        def _collect_runtime() -> dict[str, Any]:
            runtime: dict[str, Any] = {
                "torch": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda": getattr(getattr(torch, "version", None), "cuda", None),
                "use_4bit_requested": bool(use_4bit),
            }
            if torch.cuda.is_available():
                try:
                    runtime["n_gpu"] = int(torch.cuda.device_count())
                    runtime["gpu_name"] = torch.cuda.get_device_name(0)
                    runtime["gpu_capability"] = list(torch.cuda.get_device_capability(0))
                except Exception:
                    pass

            try:
                import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]

                runtime["bitsandbytes"] = getattr(bnb, "__version__", "unknown")
                runtime["bitsandbytes_import_ok"] = True
            except Exception as e:
                runtime["bitsandbytes"] = None
                runtime["bitsandbytes_import_ok"] = False
                runtime["bitsandbytes_error"] = str(e)

            return runtime

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cpu_flag = device.type != "cuda"
        if device.type != "cuda" and use_4bit:
            raise RuntimeError("use_4bit is only supported on CUDA devices in this build.")

        # Persist runtime diagnostics early so users can quickly confirm GPU/BNB are active.
        self._jobs.update(
            job_id,
            state="running",
            detail=self._merge_detail(
                job_id,
                {
                    "stage": "RL",
                    "runtime": _collect_runtime(),
                },
            ),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quant_config = None
        if use_4bit and device.type == "cuda":
            if BitsAndBytesConfig is None:
                raise RuntimeError("use_4bit requested but BitsAndBytesConfig is unavailable (transformers/bitsandbytes missing).")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Prefer letting HF place layers automatically on GPU.
        # For CPU-only environments, fall back to normal loading.
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            device_map="auto" if device.type == "cuda" else None,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
            quantization_config=quant_config,
        )

        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_id,
            device_map="auto" if device.type == "cuda" else None,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
            quantization_config=quant_config,
        )

        # Apply LoRA via PPOTrainer peft_config if supported.
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # TRL's PPOConfig signature changes across versions; filter kwargs defensively.
        import inspect

        def _filter_kwargs(fn: Any, kwargs_in: dict[str, Any]) -> dict[str, Any]:
            sig_local = inspect.signature(fn)
            # If the function accepts **kwargs, don't filter (keeps generation kwargs like max_new_tokens).
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_local.parameters.values()):
                return kwargs_in
            return {k: v for k, v in kwargs_in.items() if k in sig_local.parameters}

        cfg_kwargs: dict[str, Any] = {
            # Model field differs across versions.
            "model_name": model_id,
            "model_name_or_path": model_id,
            # Ensure CPU-only environments don't trip accelerate mixed-precision checks.
            "use_cpu": use_cpu_flag,
            "bf16": False,
            "fp16": False,
            "learning_rate": 1e-5,
            "log_with": None,
            "batch_size": int(batch_size),
            "mini_batch_size": max(1, int(batch_size)),
            "kl_coef": float(kl_coef),
            "seed": int(seed),
        }
        cfg = PPOConfig(**_filter_kwargs(PPOConfig.__init__, cfg_kwargs))

        # PPOTrainer API differs across versions; use best-effort signature matching.

        kwargs: dict[str, Any] = {}
        sig = inspect.signature(PPOTrainer.__init__)
        if "peft_config" in sig.parameters:
            kwargs["peft_config"] = peft_config

        trainer = PPOTrainer(
            config=cfg,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            **kwargs,
        )

        # Guard against exceeding model position embeddings.
        # Many prompts include retrieved context; we must reserve room for generation.
        try:
            max_pos = int(getattr(getattr(model, "pretrained_model", model).config, "max_position_embeddings"))
        except Exception:
            max_pos = int(getattr(tokenizer, "model_max_length", 1024))
        # Some tokenizers use a huge sentinel value for model_max_length.
        if max_pos <= 0 or max_pos > 1_000_000:
            max_pos = 1024

        max_prompt_length = max_pos - int(max_new_tokens) - 1
        if max_prompt_length < 32:
            max_prompt_length = max(32, max_pos // 2)
        # Keep smoke tests quick and memory-friendly.
        max_prompt_length = int(min(512, max_prompt_length))

        started = time.time()
        weights = base_weights.normalized()

        if weight_update_every <= 0:
            weight_update_every = 1

        last_avg_reward: float | None = None
        best_avg_reward: float | None = None
        degradation_bad_steps = 0

        metrics_path = self._report_dir / f"train_{job_id}_rl_metrics.jsonl"

        def _append_metrics(entry: dict[str, Any]) -> None:
            try:
                import json

                with metrics_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        for step_idx in range(int(steps)):
            if self._should_stop(job_id):
                self._jobs.update(job_id, state="stopped", detail=self._merge_detail(job_id, {"stop_requested": True}))
                break

            batch = [items[(step_idx * batch_size + i) % len(items)] for i in range(int(batch_size))]
            prompts = [b.prompt for b in batch]

            # Tokenize queries
            q = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            )
            query_tensors_batch = q["input_ids"].to(trainer.accelerator.device)
            # TRL PPOTrainer.step (legacy) expects 1D tensors or list of 1D tensors.
            query_tensors_list = [t for t in query_tensors_batch]

            # Generate responses
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": True,
                "top_p": 0.9,
                "temperature": 0.8,
                "pad_token_id": tokenizer.eos_token_id,
                # Some TRL versions accept this and return only the generated continuation.
                "return_prompt": False,
            }
            # TRL PPOTrainer.generate (legacy) expects 1D tensors or list of 1D tensors.
            response_tensors_raw = trainer.generate(query_tensors_list, **_filter_kwargs(trainer.generate, gen_kwargs))

            if isinstance(response_tensors_raw, torch.Tensor):
                response_tensors_list = [t for t in response_tensors_raw]
            else:
                response_tensors_list = list(response_tensors_raw)

            # Decode per sample
            responses: list[str] = []
            response_token_counts: list[int] = []
            for i in range(len(response_tensors_list)):
                # response_tensors may include prompt; best-effort slice
                full = tokenizer.decode(response_tensors_list[i], skip_special_tokens=True)
                # try to strip the prompt prefix
                if i < len(prompts) and full.startswith(prompts[i]):
                    resp = full[len(prompts[i]) :]
                else:
                    resp = full
                responses.append(resp)
                try:
                    response_token_counts.append(int(len(tokenizer.encode(resp))))
                except Exception:
                    response_token_counts.append(int(max(0, len(resp.split()))))

            # Compute rewards
            rewards: list[float] = []
            comps_for_log: list[dict[str, float]] = []
            short_count = 0
            for item, resp in zip(batch, responses, strict=False):
                rr = reward_fn(item, resp, weights)
                r = float(rr.reward)
                if min_response_tokens and min_response_tokens > 0:
                    try:
                        idx = len(rewards)
                        n_tok = response_token_counts[idx] if idx < len(response_token_counts) else 0
                    except Exception:
                        n_tok = 0
                    if n_tok < int(min_response_tokens):
                        short_count += 1
                        scale = max(0.0, float(n_tok) / float(min_response_tokens))
                        r = r * scale
                rewards.append(float(r))
                comps_for_log.append({"rel": rr.components.rel, "faith": rr.components.faith, "conc": rr.components.conc})

                if weight_scheduler is not None:
                    try:
                        weight_scheduler.push(rr.components)
                    except Exception:
                        pass

            if weight_scheduler is not None and ((step_idx + 1) % int(weight_update_every) == 0):
                try:
                    weights = weight_scheduler.current_weights(base_weights)
                except Exception:
                    weights = weights

            # TRL expects `scores` to be a list of torch tensors.
            scores = [torch.tensor(r, device=trainer.accelerator.device, dtype=torch.float32) for r in rewards]
            stats = trainer.step(query_tensors_list, response_tensors_list, scores)

            last_avg_reward = float(sum(rewards) / max(1, len(rewards)))
            best_avg_reward = last_avg_reward if best_avg_reward is None else float(max(best_avg_reward, last_avg_reward))

            degraded = False
            if best_avg_reward is not None and best_avg_reward > 0 and degradation_drop_ratio and degradation_drop_ratio > 0:
                threshold = float(best_avg_reward) * float(max(0.0, 1.0 - float(degradation_drop_ratio)))
                degraded = last_avg_reward < threshold
                degradation_bad_steps = (degradation_bad_steps + 1) if degraded else 0

            degradation_triggered = bool(degradation_patience and degradation_bad_steps >= int(degradation_patience))
            if degradation_triggered and stop_on_degradation:
                self._jobs.update(
                    job_id,
                    state="stopped",
                    detail=self._merge_detail(
                        job_id,
                        {
                            "stop_requested": True,
                            "stop_reason": "policy_degradation",
                        },
                    ),
                )
                break

            _append_metrics(
                {
                    "ts": time.time(),
                    "step": int(step_idx + 1),
                    "avg_reward": float(last_avg_reward),
                    "best_avg_reward": float(best_avg_reward),
                    "short_count": int(short_count),
                    "min_response_tokens": int(min_response_tokens or 0),
                    "degraded": bool(degraded),
                    "degradation_bad_steps": int(degradation_bad_steps),
                    "degradation_drop_ratio": float(degradation_drop_ratio),
                    "degradation_patience": int(degradation_patience),
                }
            )

            elapsed = max(1e-6, time.time() - started)
            qps = (step_idx + 1) / elapsed
            eta_seconds = int((steps - (step_idx + 1)) / max(1e-6, qps))

            self._jobs.update(
                job_id,
                state="running",
                detail=self._merge_detail(
                    job_id,
                    {
                        "stage": "RL",
                        "epoch": 0,
                        "step": int(step_idx + 1),
                        "loss": None,
                        "avg_rewards": last_avg_reward,
                        "eta_seconds": eta_seconds,
                        "rl": {
                            "weights": weights.__dict__,
                            "base_weights": base_weights.normalized().__dict__,
                            "weight_update_every": int(weight_update_every),
                            "batch_rewards": rewards,
                            "batch_components": comps_for_log,
                            "min_response_tokens": int(min_response_tokens or 0),
                            "short_count": int(short_count),
                            "best_avg_reward": float(best_avg_reward) if best_avg_reward is not None else None,
                            "degradation": {
                                "drop_ratio": float(degradation_drop_ratio),
                                "patience": int(degradation_patience),
                                "bad_steps": int(degradation_bad_steps),
                                "triggered": bool(degradation_triggered),
                            },
                            "stats": _safe_json(stats),
                        },
                    },
                ),
            )

        summary_path = str(self._report_dir / f"train_{job_id}_rl_summary.json")
        adapter_path = str(self._report_dir / f"train_{job_id}_adapter")

        try:
            # Save a compact summary; do not save full base weights.
            Path(summary_path).write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "stage": "RL",
                        "model_id": model_id,
                        "steps": int(steps),
                        "final_avg_reward": last_avg_reward,
                        "weights": weights.__dict__,
                        "device": str(device),
                        "runtime": _collect_runtime(),
                        "anti_cheat": {
                            "min_response_tokens": int(min_response_tokens or 0),
                            "degradation_drop_ratio": float(degradation_drop_ratio),
                            "degradation_patience": int(degradation_patience),
                            "stop_on_degradation": bool(stop_on_degradation),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            summary_path = None

        # Best-effort: try saving adapters if PEFT is active
        try:
            trainer.model.save_pretrained(adapter_path)
        except Exception:
            adapter_path = None

        self._jobs.update(
            job_id,
            state="succeeded" if (self._jobs.get(job_id) and self._jobs.get(job_id).state not in {"stopped", "failed"}) else (self._jobs.get(job_id).state if self._jobs.get(job_id) else "succeeded"),
            detail=self._merge_detail(
                job_id,
                {
                    "stage": "RL",
                    "epoch": 0,
                    "eta_seconds": 0,
                    "artifacts": {
                        "rl_summary_path": summary_path,
                        "adapter_path": adapter_path,
                    },
                },
            ),
        )

        return PPOArtifacts(summary_path=summary_path, adapter_path=adapter_path)

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


def _safe_json(obj: Any) -> Any:
    try:
        import json

        json.dumps(obj)
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None
