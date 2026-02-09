from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    # .../graphrag_saas/backend/scripts/rescore_eval_run.py -> repo root
    return Path(__file__).resolve().parents[3]


def _container_path_to_host(repo_root: Path, p: str) -> Path:
    s = (p or "").strip()
    if s.startswith("/app/eval_docs/"):
        return repo_root / "docs" / s.removeprefix("/app/eval_docs/").lstrip("/")
    if s.startswith("/app/data/"):
        return repo_root / "graphrag_saas" / "backend" / "data" / s.removeprefix("/app/data/").lstrip("/")
    if s.startswith("/app/reports/"):
        return repo_root / "graphrag_saas" / "backend" / "reports" / s.removeprefix("/app/reports/").lstrip("/")
    return (repo_root / s).resolve()


def _load_eval_module(repo_root: Path):
    p = repo_root / "graphrag_saas" / "backend" / "scripts" / "eval_json_v2.py"
    spec = importlib.util.spec_from_file_location("_eval_json_v2", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {p}")
    mod = importlib.util.module_from_spec(spec)
    # Ensure the module is visible in sys.modules for decorators (e.g. dataclasses) during import.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _read_raw_text(run_dir: Path, qid: str, *, prefer_repair: bool) -> str:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        return ""
    cands: list[Path] = []
    if prefer_repair:
        cands.append(raw_dir / f"{qid}.raw2.txt")
    cands.append(raw_dir / f"{qid}.raw1.txt")
    for p in cands:
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                return ""
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rescore an existing eval run using the current eval_json_v2.py scoring (no LLM generation)."
    )
    ap.add_argument("--run-dir", required=True, help="reports/evals/<run> directory")
    args = ap.parse_args()

    repo_root = _repo_root()
    eval_mod = _load_eval_module(repo_root)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    metrics_json = run_dir / "metrics.json"
    metrics_partial = run_dir / "metrics.partial.json"
    metrics_path = metrics_json if metrics_json.exists() else metrics_partial
    metrics = _load_json(metrics_path)
    if not isinstance(metrics, dict):
        raise SystemExit("metrics must be a JSON object.")

    per_item_json = run_dir / "per_item.json"
    per_item_partial = run_dir / "per_item.partial.json"
    per_item_path = per_item_json if per_item_json.exists() else per_item_partial
    per_item = _load_json(per_item_path)
    if not isinstance(per_item, list):
        raise SystemExit("per_item must be a JSON array.")

    prov = metrics.get("provenance") or {}
    if not isinstance(prov, dict):
        prov = {}

    eval_set_host = _container_path_to_host(repo_root, str(prov.get("eval_set") or ""))
    index_dir_host = _container_path_to_host(repo_root, str(prov.get("index_dir") or ""))
    retrieval = prov.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        retrieval = {}
    top_k = int(retrieval.get("top_k") or 5)
    max_context_chars = int(retrieval.get("max_context_chars") or 4000)

    questions = _load_json(eval_set_host)
    if not isinstance(questions, list):
        raise SystemExit("eval_set must be a JSON array.")
    q_by_id: dict[str, dict[str, Any]] = {str(q.get("id")): q for q in questions if isinstance(q, dict) and q.get("id")}

    store = eval_mod.IndexStore(Path(index_dir_host))
    chunk_rows = store.load_chunks()
    if not isinstance(chunk_rows, list):
        raise SystemExit(f"Invalid chunks from index: {index_dir_host}")
    chunks = [str(r.get("text") or "") for r in chunk_rows if isinstance(r, dict)]
    chunk_meta = [r if isinstance(r, dict) else {} for r in chunk_rows]
    entity_names = store.load_entity_names()
    retriever = eval_mod.HierarchicalRetriever(chunks, entity_names)

    rescored: list[dict[str, Any]] = []
    scores: list[float] = []
    schema_ok_count = 0

    for it in per_item:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "")
        if not qid:
            continue
        q = q_by_id.get(qid)
        if not isinstance(q, dict):
            continue

        question_text = str(q.get("question") or "")
        expected = q.get("expected") or {}
        if not isinstance(expected, dict):
            expected = {}

        ctx = eval_mod.build_context(
            question=question_text,
            retriever=retriever,
            chunks=chunks,
            chunk_meta=chunk_meta,
            top_k=top_k,
            max_chars=max_context_chars,
        )

        raw_text = _read_raw_text(run_dir, qid, prefer_repair=bool(it.get("repair_used")))
        if not raw_text:
            raw_text = str(it.get("raw_head") or "")

        got_obj = None
        try:
            got_obj, _meta = eval_mod._parse_json_object(raw_text)
        except Exception:
            got_obj = None

        # Mirror eval runner behavior: normalize source_map refs/anchors against retrieved context.
        repair_fn = getattr(eval_mod, "_repair_source_map_for_grounding", None)
        if callable(repair_fn) and isinstance(got_obj, dict):
            try:
                repair_fn(got_obj, ctx)
            except Exception:
                pass

        scored = eval_mod.score_answer_v2(expected=expected, got=got_obj, context_chunks=ctx)
        score = float(scored.get("score") or 0.0)
        schema_ok = bool(scored.get("schema_ok"))
        if schema_ok:
            schema_ok_count += 1
        scores.append(score)

        out = dict(it)
        out["rescored"] = {
            "score": score,
            "schema_ok": schema_ok,
            "subs": scored.get("subs") or {},
            "notes": scored.get("notes") or [],
            "match_diagnostics": scored.get("match_diagnostics") or {},
        }
        rescored.append(out)

    eval_score_avg = float(statistics.mean(scores)) if scores else 0.0
    schema_pass_rate = (schema_ok_count / max(1, len(scores))) if scores else 0.0

    out_metrics = {
        "rescored_from": str(metrics_path),
        "rescored_per_item_from": str(per_item_path),
        "eval": {
            "question_count": int(len(scores)),
            "schema_pass_rate": float(round(schema_pass_rate, 6)),
            "eval_score_avg": float(round(eval_score_avg, 6)),
        },
        "provenance": prov,
    }

    # Convenience: a "flat" per_item file that mirrors the original per_item schema
    # but replaces score/subs/notes/match_diagnostics with the rescored values.
    rescored_flat: list[dict[str, Any]] = []
    for it in rescored:
        base = dict(it)
        r = base.get("rescored") or {}
        if isinstance(r, dict):
            base["score"] = float(r.get("score") or 0.0)
            base["schema_ok"] = bool(r.get("schema_ok"))
            base["subs"] = r.get("subs") or {}
            base["notes"] = r.get("notes") or []
            base["match_diagnostics"] = r.get("match_diagnostics") or {}
        rescored_flat.append(base)

    (run_dir / "metrics.rescored.json").write_text(
        json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "per_item.rescored.json").write_text(
        json.dumps(rescored, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "per_item.rescored_flat.json").write_text(
        json.dumps(rescored_flat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
