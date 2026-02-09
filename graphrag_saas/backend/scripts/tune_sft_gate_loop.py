from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    max_steps: int
    max_length: int
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "max_steps": self.max_steps,
            "max_length": self.max_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }


def _http() -> httpx.Client:
    return httpx.Client(timeout=60.0)


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    with _http() as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


def get_json(url: str) -> dict[str, Any]:
    with _http() as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def wait_train_done(base: str, job_id: str, *, poll_s: float = 8.0) -> dict[str, Any]:
    url = f"{base.rstrip('/')}/api/train/status/{job_id}"
    while True:
        st = get_json(url)
        state = str(st.get("state") or "")
        if state in {"succeeded", "failed", "stopped"}:
            return st
        time.sleep(poll_s)


def find_registry_entry_id_for_job(*, registry_path: Path, job_id: str) -> str | None:
    if not registry_path.exists():
        return None
    data = json.loads(registry_path.read_text(encoding="utf-8") or "{}")
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    # prefer latest match
    found: str | None = None
    for m in models:
        if not isinstance(m, dict):
            continue
        if str(m.get("job_id") or "") == job_id and str(m.get("stage") or "") == "SFT":
            found = str(m.get("id") or "").strip() or found
    return found


def run_eval_local(
    *,
    python_bin: str,
    eval_script: str,
    questions: str,
    prompt_asset: str,
    out_dir: str,
    index_dir: str,
    model_id: str,
    adapter_path: str,
    registry_path: str,
    active_path: str,
    registry_model_id: str,
    adapter_id: str,
    metric_prefix: str,
    gate_threshold: float | None = None,
) -> dict[str, Any]:
    import subprocess

    cmd = [
        python_bin,
        eval_script,
        "--provider",
        "local",
        "--questions",
        questions,
        "--prompt-asset",
        prompt_asset,
        "--out-dir",
        out_dir,
        "--index-dir",
        index_dir,
        "--llm-model",
        model_id,
        "--adapter-path",
        adapter_path,
        "--registry-path",
        registry_path,
        "--active-path",
        active_path,
        "--registry-model-id",
        registry_model_id,
        "--adapter-id",
        adapter_id,
        "--metric-prefix",
        metric_prefix,
        "--temperature",
        "0.0",
    ]
    if gate_threshold is not None:
        cmd.extend(["--gate-threshold", str(gate_threshold)])

    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"eval failed ({p.returncode}): {p.stderr.strip() or p.stdout.strip()}")

    metrics_path = Path(out_dir) / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def default_round_configs(round_idx: int) -> list[TrainCfg]:
    # Keep to 2~4 configs per round.
    if round_idx == 0:
        return [TrainCfg(lr=2e-4, max_steps=20, max_length=1024)]
    if round_idx == 1:
        return [
            TrainCfg(lr=2e-4, max_steps=200, max_length=1024),
            TrainCfg(lr=1e-4, max_steps=300, max_length=1024),
            TrainCfg(lr=3e-4, max_steps=150, max_length=1024),
        ]
    # Later rounds: tiny neighborhood search around "reasonable defaults"
    base = TrainCfg(lr=1.5e-4, max_steps=300, max_length=1024)
    return [
        base,
        TrainCfg(lr=base.lr, max_steps=base.max_steps + 100, max_length=base.max_length),
        TrainCfg(lr=base.lr * 1.2, max_steps=base.max_steps, max_length=base.max_length),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune SFT configs with dev gate eval (local provider).")
    ap.add_argument("--api-base", default=os.getenv("BACKEND_URL", "http://localhost:8000"))
    ap.add_argument("--model-id", default=os.getenv("SFT_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--rounds", type=int, default=3)

    ap.add_argument("--train-n-questions", type=int, default=64)
    ap.add_argument("--train-top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1)

    ap.add_argument("--index-dir", default=os.getenv("INDEX_DIR", "/app/data/index"))

    # Eval assets (mounted from repo root docs -> /app/eval_docs in compose)
    ap.add_argument("--gate-questions", default="/app/eval_docs/eval_sets/question_v2_dev100.json")
    ap.add_argument("--full-questions", default="/app/eval_docs/question_v2.json")
    ap.add_argument("--prompt-asset", default="/app/eval_docs/prompts/eval_json_v2.md")

    ap.add_argument("--eval-out-root", default="/app/reports/evals")
    ap.add_argument("--gate-threshold", type=float, default=95.0)

    ap.add_argument("--registry-path", default="/app/data/model_registry.json")
    ap.add_argument("--active-path", default="/app/data/active_models.json")

    args = ap.parse_args()

    api = args.api_base.rstrip("/")
    Path(args.eval_out_root).mkdir(parents=True, exist_ok=True)

    best: dict[str, Any] | None = None
    best_avg = -1.0

    for r in range(int(args.rounds)):
        cfgs = default_round_configs(r)
        print(f"[Round {r}] candidates={len(cfgs)}")
        for cfg in cfgs:
            payload = {
                "stage": "SFT",
                "model_id": args.model_id,
                "n_questions": int(args.train_n_questions),
                "top_k": int(args.train_top_k),
                "seed": 42,
                "max_steps": int(cfg.max_steps),
                "batch_size": int(args.batch_size),
                "lr": float(cfg.lr),
                "max_length": int(cfg.max_length),
                "use_4bit": True,
                "lora_r": int(cfg.lora_r),
                "lora_alpha": int(cfg.lora_alpha),
                "lora_dropout": float(cfg.lora_dropout),
            }

            print(f"  [Train] {cfg.to_dict()}")
            resp = post_json(f"{api}/api/train", payload)
            job_id = str(resp.get("job_id"))
            st = wait_train_done(api, job_id)
            if str(st.get("state")) != "succeeded":
                print(f"    -> train failed: {st.get('detail')}")
                continue

            detail = st.get("detail") or {}
            artifacts = detail.get("artifacts") or {}
            adapter_path = str(artifacts.get("sft_adapter_path") or "")
            if not adapter_path:
                print("    -> missing sft_adapter_path, skip eval")
                continue

            reg_id = find_registry_entry_id_for_job(registry_path=Path(args.registry_path), job_id=job_id) or ""
            adapter_id = reg_id or job_id

            gate_out = f"{args.eval_out_root.rstrip('/')}/gate_{job_id}"
            print(f"  [Eval gate] out={gate_out}")
            gate_metrics = run_eval_local(
                python_bin=os.getenv("PYTHON", "python"),
                eval_script="/app/scripts/eval_json_v2.py",
                questions=args.gate_questions,
                prompt_asset=args.prompt_asset,
                out_dir=gate_out,
                index_dir=args.index_dir,
                model_id=args.model_id,
                adapter_path=adapter_path,
                registry_path=args.registry_path,
                active_path=args.active_path,
                registry_model_id=reg_id,
                adapter_id=adapter_id,
                metric_prefix="gate_",
                gate_threshold=float(args.gate_threshold),
            )

            avg = float(gate_metrics.get("eval", {}).get("eval_score_avg") or 0.0)
            passed = bool(gate_metrics.get("gate", {}).get("passed"))
            print(f"    -> gate avg={avg:.3f} passed={passed}")

            if avg > best_avg:
                best_avg = avg
                best = {
                    "job_id": job_id,
                    "registry_model_id": reg_id,
                    "adapter_path": adapter_path,
                    "train_cfg": cfg.to_dict(),
                    "gate_metrics": gate_metrics,
                }

            if passed:
                full_out = f"{args.eval_out_root.rstrip('/')}/full_{job_id}"
                print(f"  [Eval full] out={full_out}")
                _full_metrics = run_eval_local(
                    python_bin=os.getenv("PYTHON", "python"),
                    eval_script="/app/scripts/eval_json_v2.py",
                    questions=args.full_questions,
                    prompt_asset=args.prompt_asset,
                    out_dir=full_out,
                    index_dir=args.index_dir,
                    model_id=args.model_id,
                    adapter_path=adapter_path,
                    registry_path=args.registry_path,
                    active_path=args.active_path,
                    registry_model_id=reg_id,
                    adapter_id=adapter_id,
                    metric_prefix="full_",
                    gate_threshold=None,
                )
                print("  [DONE] gate passed; full eval finished.")
                print(json.dumps(best, ensure_ascii=False, indent=2))
                return 0

    print("[STOP] No gate pass; best candidate summary:")
    print(json.dumps(best or {"best_avg": best_avg}, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

