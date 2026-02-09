from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha1_text(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _norm(s: str) -> str:
    return (s or "").strip()


def _group_key(item: dict[str, Any]) -> tuple[str, str]:
    expected = item.get("expected") or {}
    answer_example = expected.get("answer_example") or {}
    ta = _norm(str(answer_example.get("target_audience") or ""))
    mt = _norm(str(answer_example.get("main_topic") or ""))
    return ta, mt


def _stable_shuffle(items: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def build_subset(
    *,
    items: list[dict[str, Any]],
    subset_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size <= 0:
        return []
    if subset_size >= len(items):
        return list(items)

    # Group by (target_audience, main_topic) to cover distribution.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        groups[_group_key(it)].append(it)

    # Shuffle within each group deterministically.
    for k in list(groups.keys()):
        groups[k] = _stable_shuffle(groups[k], seed=seed + (hash(k) % 10_000))

    group_sizes = {k: len(v) for k, v in groups.items()}
    total = sum(group_sizes.values())
    if total <= 0:
        return []

    # 1) Ensure coverage: at least 1 per group (until we run out).
    chosen: list[dict[str, Any]] = []
    keys = sorted(groups.keys(), key=lambda k: (-group_sizes[k], k[0], k[1]))
    for k in keys:
        if len(chosen) >= subset_size:
            break
        if groups[k]:
            chosen.append(groups[k].pop(0))

    if len(chosen) >= subset_size:
        return chosen[:subset_size]

    # 2) Allocate remaining quota proportionally to group size.
    remaining = subset_size - len(chosen)
    weights = {k: group_sizes[k] for k in keys}
    weight_sum = sum(weights.values()) or 1

    alloc: dict[tuple[str, str], int] = {k: 0 for k in keys}
    for k in keys:
        alloc[k] = int(math.floor(remaining * (weights[k] / weight_sum)))

    # Fix rounding: distribute leftovers to largest groups.
    allocated = sum(alloc.values())
    leftovers = max(0, remaining - allocated)
    for k in keys[:leftovers]:
        alloc[k] += 1

    for k in keys:
        take = min(alloc[k], len(groups[k]))
        if take > 0:
            chosen.extend(groups[k][:take])
            groups[k] = groups[k][take:]

    if len(chosen) >= subset_size:
        return chosen[:subset_size]

    # 3) Fill any still-remaining slots by global deterministic shuffle.
    rest: list[dict[str, Any]] = []
    for k in keys:
        rest.extend(groups[k])
    rest = _stable_shuffle(rest, seed=seed + 99991)
    need = subset_size - len(chosen)
    chosen.extend(rest[:need])
    return chosen[:subset_size]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a stratified eval subset (gate set).")
    ap.add_argument("--in-questions", required=True, help="Path to full questions JSON (array).")
    ap.add_argument("--out", required=True, help="Output path for subset JSON (array).")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.in_questions)
    out_path = Path(args.out)

    items = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input questions must be a JSON array.")

    subset = build_subset(items=items, subset_size=int(args.size), seed=int(args.seed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print a small summary for reproducibility.
    counts = Counter(_group_key(it) for it in subset)
    summary = {
        "in": str(in_path),
        "in_sha1": _sha1_text(in_path.read_text(encoding="utf-8")),
        "out": str(out_path),
        "out_sha1": _sha1_text(out_path.read_text(encoding="utf-8")),
        "size": len(subset),
        "seed": int(args.seed),
        "unique_groups": len(counts),
        "top_groups": [
            {"target_audience": k[0], "main_topic": k[1], "count": int(v)}
            for k, v in counts.most_common(10)
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

