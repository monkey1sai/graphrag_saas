"""Merge 3 partial eval runs into a single full-1000 result set."""
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # graphrag_saas/backend -> graphrag_saas
EVALS = REPO / "backend" / "reports" / "evals"

PARTS = [
    EVALS / "full1000_run1_repair_p10_tok512" / "per_item.partial.json",
    EVALS / "full1000_run1_repair_p10_tok512_resume" / "per_item.partial.json",
    EVALS / "full1000_run1_repair_p10_tok512_final69" / "per_item.partial.json",
]

OUT_DIR = EVALS / "full1000_merged"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_items: list[dict] = []
    seen_ids: set[str] = set()

    for p in PARTS:
        if not p.exists():
            print(f"WARNING: {p} not found, skipping")
            continue
        items = json.loads(p.read_text(encoding="utf-8"))
        for it in items:
            qid = str(it.get("id", ""))
            if qid and qid not in seen_ids:
                seen_ids.add(qid)
                all_items.append(it)
        print(f"Loaded {len(items)} from {p.name} (parent: {p.parent.name})")

    # Sort by ID
    all_items.sort(key=lambda x: str(x.get("id", "")))

    # Compute aggregate metrics
    schema_ok = sum(1 for x in all_items if x.get("schema_ok"))
    scores = [float(x.get("score") or 0) for x in all_items if x.get("schema_ok")]
    grounding = [float((x.get("subs") or {}).get("grounding") or 0) for x in all_items if x.get("schema_ok")]

    n = len(all_items)
    metrics = {
        "eval": {
            "question_count": n,
            "schema_pass_rate": round(schema_ok / max(1, n), 6),
            "eval_score_avg": round(statistics.mean(scores), 6) if scores else 0.0,
            "grounding_avg": round(statistics.mean(grounding), 6) if grounding else 0.0,
        },
        "fail_counts": {},
        "merge_info": {
            "parts": [str(p) for p in PARTS],
            "part_counts": [],
        },
    }

    # Count fail reasons
    for it in all_items:
        fr = it.get("fail_reason")
        key = fr if fr else "ok"
        metrics["fail_counts"][key] = metrics["fail_counts"].get(key, 0) + 1

    for p in PARTS:
        if p.exists():
            items = json.loads(p.read_text(encoding="utf-8"))
            metrics["merge_info"]["part_counts"].append({"file": p.parent.name, "count": len(items)})

    # Write outputs
    (OUT_DIR / "per_item.json").write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print(f"Merged {n} items → {OUT_DIR}")
    print(f"  schema_pass_rate: {metrics['eval']['schema_pass_rate']}")
    print(f"  eval_score_avg:   {metrics['eval']['eval_score_avg']}")
    print(f"  grounding_avg:    {metrics['eval']['grounding_avg']}")
    print(f"  fail_counts:      {metrics['fail_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
