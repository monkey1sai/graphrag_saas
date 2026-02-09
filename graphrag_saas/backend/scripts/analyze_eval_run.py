from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _quantiles(vals: list[float], ps: list[float]) -> dict[str, float]:
    if not vals:
        return {f"p{int(p*100)}": 0.0 for p in ps}
    s = sorted(vals)
    out: dict[str, float] = {}
    for p in ps:
        if p <= 0:
            out[f"p{int(p*100)}"] = float(s[0])
            continue
        if p >= 1:
            out[f"p{int(p*100)}"] = float(s[-1])
            continue
        idx = int(round((len(s) - 1) * p))
        out[f"p{int(p*100)}"] = float(s[max(0, min(len(s) - 1, idx))])
    return out


def _stat_block(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, **_quantiles([], [0.5, 0.9, 0.95])}
    return {
        "mean": float(statistics.mean(vals)),
        "min": float(min(vals)),
        "max": float(max(vals)),
        **_quantiles(vals, [0.5, 0.9, 0.95]),
    }


def _repo_root() -> Path:
    # .../graphrag_saas/backend/scripts/analyze_eval_run.py -> repo root
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


def _parse_output_json(eval_mod, text: str) -> dict[str, Any] | None:
    try:
        obj, _meta = eval_mod._parse_json_object(text or "")
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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
    ap = argparse.ArgumentParser(description="Analyze an eval run (per_item.json) into breakdown.json + report.md.")
    ap.add_argument("--run-dir", required=True, help="reports/evals/<run> directory")
    ap.add_argument("--top-n", type=int, default=15, help="Top-N confusion pairs to show")
    ap.add_argument(
        "--use-rescored",
        type=int,
        default=0,
        help="If 1, prefer metrics.rescored.json + per_item.rescored_flat.json when present.",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    eval_mod = _load_eval_module(repo_root)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")

    metrics_rescored = run_dir / "metrics.rescored.json"
    metrics_json = run_dir / "metrics.json"
    metrics_partial = run_dir / "metrics.partial.json"
    if int(args.use_rescored) == 1 and metrics_rescored.exists():
        metrics_path = metrics_rescored
    else:
        metrics_path = metrics_json if metrics_json.exists() else metrics_partial
    metrics = _load_json(metrics_path)

    per_item_rescored_flat = run_dir / "per_item.rescored_flat.json"
    per_item_json = run_dir / "per_item.json"
    per_item_partial = run_dir / "per_item.partial.json"
    if int(args.use_rescored) == 1 and per_item_rescored_flat.exists():
        per_item_path = per_item_rescored_flat
    else:
        per_item_path = per_item_json if per_item_json.exists() else per_item_partial
    per_item = _load_json(per_item_path)
    if not isinstance(per_item, list):
        raise SystemExit("per_item must be a JSON array.")

    is_partial = per_item_path.name.endswith(".partial.json") or metrics_path.name.endswith(".partial.json")

    prov = (metrics.get("provenance") or {}) if isinstance(metrics, dict) else {}
    eval_set_host = _container_path_to_host(repo_root, str(prov.get("eval_set") or ""))
    index_dir_host = _container_path_to_host(repo_root, str(prov.get("index_dir") or ""))
    retrieval = (prov.get("retrieval") or {}) if isinstance(prov, dict) else {}
    top_k = int(retrieval.get("top_k") or 5)
    max_context_chars = int(retrieval.get("max_context_chars") or 4000)

    questions = _load_json(eval_set_host)
    if not isinstance(questions, list):
        raise SystemExit("eval_set must be a JSON array.")
    q_by_id: dict[str, dict[str, Any]] = {str(q.get("id")): q for q in questions if isinstance(q, dict) and q.get("id")}
    question_count_expected = len(questions)

    # Load index for grounding diagnosis
    store = eval_mod.IndexStore(Path(index_dir_host))
    chunk_rows = store.load_chunks()
    if not isinstance(chunk_rows, list):
        raise SystemExit(f"Invalid chunks from index: {index_dir_host}")
    chunks = [str(r.get("text") or "") for r in chunk_rows if isinstance(r, dict)]
    chunk_meta = [r if isinstance(r, dict) else {} for r in chunk_rows]
    entity_names = store.load_entity_names()
    retriever = eval_mod.HierarchicalRetriever(chunks, entity_names)

    scores: list[float] = []
    subs_series: dict[str, list[float]] = defaultdict(list)
    fail_reason = Counter()
    primary_fail_reason = Counter()
    repair_used = 0

    confusion: dict[str, Counter[tuple[str, str]]] = {
        "target_audience": Counter(),
        "main_topic": Counter(),
        "sub_topic": Counter(),
    }
    hits: dict[str, int] = {"target_audience": 0, "main_topic": 0, "sub_topic": 0}
    totals: dict[str, int] = {"target_audience": 0, "main_topic": 0, "sub_topic": 0}

    grounding_zero_reasons = Counter()

    for it in per_item:
        if not isinstance(it, dict):
            continue
        qid = str(it.get("id") or "")
        if not qid:
            continue
        scores.append(float(it.get("score") or 0.0))

        fr = it.get("fail_reason")
        if fr:
            fail_reason[str(fr)] += 1
        pfr = it.get("primary_fail_reason")
        if pfr:
            primary_fail_reason[str(pfr)] += 1

        if bool(it.get("repair_used")):
            repair_used += 1

        subs = it.get("subs") or {}
        if isinstance(subs, dict):
            for k, v in subs.items():
                try:
                    subs_series[str(k)].append(float(v))
                except Exception:
                    subs_series[str(k)].append(0.0)

        q = q_by_id.get(qid) or {}
        exp = (q.get("expected") or {}) if isinstance(q, dict) else {}
        ae = exp.get("answer_example") if isinstance(exp, dict) else None
        if not isinstance(ae, dict):
            continue

        raw_text = _read_raw_text(run_dir, qid, prefer_repair=bool(it.get("repair_used")))
        if not raw_text:
            raw_text = str(it.get("raw_head") or "")
        got = _parse_output_json(eval_mod, raw_text)

        for key in ("target_audience", "main_topic", "sub_topic"):
            exp_v = str(ae.get(key) or "")
            pred_v = str((got or {}).get(key) or "")
            if exp_v:
                totals[key] += 1
                if pred_v == exp_v:
                    hits[key] += 1
                else:
                    confusion[key][(pred_v or "<missing>", exp_v)] += 1

        # Grounding zero reasons
        try:
            s_ground = float((subs or {}).get("grounding") or 0.0)
        except Exception:
            s_ground = 0.0
        if s_ground > 1e-6:
            continue

        # rebuild context
        question_text = str(q.get("question") or "")
        ctx = eval_mod.build_context(
            question=question_text,
            retriever=retriever,
            chunks=chunks,
            chunk_meta=chunk_meta,
            top_k=top_k,
            max_chars=max_context_chars,
        )
        # Apply the same post-repair grounding normalization used by the eval runner
        # so grounding diagnostics reflect the effective scoring behavior.
        repair_fn = getattr(eval_mod, "_repair_source_map_for_grounding", None)
        if callable(repair_fn) and isinstance(got, dict):
            try:
                repair_fn(got, ctx)
            except Exception:
                pass
        ctx_files = {eval_mod._norm(c.source_path or "") for c in ctx if eval_mod._norm(c.source_path or "")}
        ctx_text = "\n".join(c.text for c in ctx)
        norm_fn = getattr(eval_mod, "normalize_text", None)
        if callable(norm_fn):
            ctx_text_norm = norm_fn(ctx_text)
            def _a_norm(x: str) -> str:
                return str(norm_fn(x))
        else:
            ctx_text_norm = eval_mod._norm(ctx_text)
            def _a_norm(x: str) -> str:
                return str(eval_mod._norm(x))

        sm = (got or {}).get("source_map")
        if not isinstance(sm, list) or not sm:
            grounding_zero_reasons["missing_source_map"] += 1
            continue

        any_ref = False
        any_ok = False
        for ent in sm:
            if not isinstance(ent, dict):
                continue
            refs = ent.get("refs")
            if not isinstance(refs, list):
                continue
            for r in refs:
                if not isinstance(r, dict):
                    continue
                any_ref = True
                f = eval_mod._norm(str(r.get("file") or ""))
                anchors = r.get("anchors") or []
                if not f:
                    grounding_zero_reasons["ref_missing_file"] += 1
                    continue
                if not isinstance(anchors, list) or not anchors:
                    grounding_zero_reasons["ref_missing_anchors"] += 1
                    continue
                if ctx_files and f not in ctx_files:
                    grounding_zero_reasons["file_not_in_retrieved_ctx"] += 1
                    continue
                ok_anchor = any((_a_norm(str(a)) and _a_norm(str(a)) in ctx_text_norm) for a in anchors[:6])
                if not ok_anchor:
                    grounding_zero_reasons["anchors_not_found_in_ctx_text"] += 1
                    continue
                any_ok = True

        if not any_ref:
            grounding_zero_reasons["missing_refs_array"] += 1
        elif not any_ok:
            grounding_zero_reasons["no_ok_ref"] += 1

    # Build breakdown.json
    breakdown: dict[str, Any] = {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "per_item_path": str(per_item_path),
        "is_partial": bool(is_partial),
        "question_count_expected": int(question_count_expected),
        "provenance": prov,
        "overall": {
            "count": len(per_item),
            "score": _stat_block(scores),
            "repair_used": {"count": repair_used, "rate": (repair_used / max(1, len(per_item)))},
            "fail_reason": dict(fail_reason),
            "primary_fail_reason": dict(primary_fail_reason),
        },
        "subs": {k: _stat_block(v) for k, v in sorted(subs_series.items())},
        "label_hits": {
            k: {"hits": hits[k], "total": totals[k], "rate": (hits[k] / max(1, totals[k]))}
            for k in ("target_audience", "main_topic", "sub_topic")
        },
        "grounding_zero_reasons": dict(grounding_zero_reasons),
    }

    (run_dir / "breakdown.json").write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")

    # report.md
    lines: list[str] = []
    lines.append("# Eval Breakdown Report")
    lines.append("")
    lines.append(f"- run_dir: `{run_dir}`")
    lines.append(f"- metrics: `{metrics_path}`")
    lines.append(f"- per_item: `{per_item_path}`")
    lines.append("")

    sc = breakdown["overall"]["score"]
    lines.append("## Overall score")
    lines.append(
        f"- mean={sc['mean']:.3f} p50={sc['p50']:.3f} p90={sc['p90']:.3f} p95={sc['p95']:.3f} min={sc['min']:.3f} max={sc['max']:.3f}"
    )
    ru = breakdown["overall"]["repair_used"]
    lines.append(f"- repair_used: {ru['count']}/{len(per_item)} ({_pct(ru['rate'])})")
    lines.append(f"- primary_fail_reason: `{json.dumps(breakdown['overall']['primary_fail_reason'], ensure_ascii=False)}`")
    lines.append("")

    lines.append("## Subs stats (mean / p50 / p95)")
    for k, st in breakdown["subs"].items():
        lines.append(f"- {k}: mean={st['mean']:.4f} p50={st['p50']:.4f} p95={st['p95']:.4f}")
    lines.append("")

    lines.append("## Label hit-rate")
    for k, d in breakdown["label_hits"].items():
        lines.append(f"- {k}: {d['hits']}/{d['total']} ({_pct(d['rate'])})")
    lines.append("")

    def _conf_section(field: str) -> None:
        lines.append(f"## Confusion Top-{int(args.top_n)}: {field} (pred -> expected)")
        c = confusion[field]
        for (pred, exp), n in c.most_common(int(args.top_n)):
            lines.append(f"- {n}× `{pred}` -> `{exp}`")
        lines.append("")

    _conf_section("target_audience")
    _conf_section("main_topic")
    _conf_section("sub_topic")

    lines.append("## Grounding=0 reasons (diagnosis)")
    if grounding_zero_reasons:
        total0 = sum(grounding_zero_reasons.values())
        for k, n in grounding_zero_reasons.most_common():
            lines.append(f"- {n}× {k} ({_pct(n / max(1, total0))})")
    else:
        lines.append("- (no grounding=0 items or insufficient data)")
    lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
