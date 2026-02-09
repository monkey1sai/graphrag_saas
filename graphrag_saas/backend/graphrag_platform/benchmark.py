from __future__ import annotations

import json
import random
import re
import time
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class QAItem:
    qid: int
    question: str
    gold_chunk_id: int
    kind: str = "basic"
    template: str | None = None


def _extract_phrases(text: str) -> list[str]:
    if not text:
        return []

    phrases: list[str] = []

    # Prefer medium-length CJK phrases
    for seq in re.findall(r"[\u4e00-\u9fff]{4,12}", text):
        if seq not in phrases:
            phrases.append(seq)

    # Also include useful alphanum tokens
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{3,}", text):
        w = w.strip()
        if w and w not in phrases:
            phrases.append(w)

    return phrases


def generate_questions(
    chunks: list[str],
    *,
    n_questions: int = 1000,
    seed: int = 42,
) -> list[QAItem]:
    rng = random.Random(seed)

    templates = [
        "請說明「{p}」的意義與重點。",
        "什麼是「{p}」？請用簡短方式說明。",
        "請根據資料描述「{p}」的用途/流程。",
        "「{p}」在系統中扮演什麼角色？",
        "請列出「{p}」相關的要點/注意事項。",
        "針對「{p}」，請整理成 3 點重點。",
    ]

    candidates: list[tuple[int, str]] = []
    for cid, chunk in enumerate(chunks):
        for p in _extract_phrases(chunk):
            candidates.append((cid, p))

    if not candidates:
        # Fallback: generic questions based on chunk prefixes
        candidates = [(i, chunk[:12]) for i, chunk in enumerate(chunks) if chunk]

    rng.shuffle(candidates)

    items: list[QAItem] = []
    seen_questions: set[str] = set()

    i = 0
    while len(items) < n_questions and i < len(candidates):
        cid, phrase = candidates[i]
        i += 1
        phrase = (phrase or "").strip()
        if len(phrase) < 2:
            continue
        tmpl = rng.choice(templates)
        question = tmpl.format(p=phrase)
        if question in seen_questions:
            continue
        seen_questions.add(question)
        items.append(QAItem(qid=len(items), question=question, gold_chunk_id=cid, kind="basic", template=tmpl))

    # If dataset is small, allow repeats by sampling chunks
    while len(items) < n_questions and chunks:
        cid = rng.randrange(0, len(chunks))
        phrase_list = _extract_phrases(chunks[cid])
        phrase = phrase_list[0] if phrase_list else (chunks[cid][:10] or "內容")
        tmpl = rng.choice(templates)
        question = tmpl.format(p=phrase)
        if question in seen_questions:
            continue
        seen_questions.add(question)
        items.append(QAItem(qid=len(items), question=question, gold_chunk_id=cid, kind="fallback", template=tmpl))

    return items


def generate_questions_stratified(
    chunks: list[str],
    *,
    n_questions: int = 1000,
    seed: int = 42,
) -> list[QAItem]:
    """Generate questions with simple stratification.

    Rationale: keep every question anchored to a single gold chunk id (for Recall/MRR),
    while mixing different template "kinds" so the set isn't overly repetitive.
    """
    rng = random.Random(seed)

    templates_by_kind: dict[str, list[str]] = {
        "local": [
            "什麼是「{p}」？請用一句話說明。",
            "請說明「{p}」的意義與重點。",
            "請列出「{p}」的重點/注意事項。",
        ],
        "global": [
            "針對「{p}」，請整理成 3 點重點。",
            "請根據資料描述「{p}」的用途/流程。",
            "「{p}」在系統中扮演什麼角色？",
        ],
        "cross_topic": [
            "「{p}」通常會和哪些服務/流程產生關聯？請用要點說明。",
            "從資料內容看，「{p}」的上下游關係是什麼？",
        ],
    }

    candidates: list[tuple[int, str]] = []
    for cid, chunk in enumerate(chunks):
        for p in _extract_phrases(chunk):
            candidates.append((cid, p))
    if not candidates:
        candidates = [(i, chunk[:12]) for i, chunk in enumerate(chunks) if chunk]

    rng.shuffle(candidates)
    seen_questions: set[str] = set()
    items: list[QAItem] = []

    # Target a mix of kinds.
    kind_targets = {
        "local": int(n_questions * 0.5),
        "global": int(n_questions * 0.35),
        "cross_topic": n_questions - int(n_questions * 0.5) - int(n_questions * 0.35),
    }
    kind_counts = {k: 0 for k in templates_by_kind}
    kind_cycle = ["local", "global", "cross_topic"]
    kind_idx = 0

    i = 0
    while len(items) < n_questions and i < len(candidates):
        cid, phrase = candidates[i]
        i += 1
        phrase = (phrase or "").strip()
        if len(phrase) < 2:
            continue

        # Pick a kind that still needs samples.
        tries = 0
        kind = kind_cycle[kind_idx % len(kind_cycle)]
        while tries < len(kind_cycle) and kind_counts.get(kind, 0) >= kind_targets.get(kind, 0):
            kind_idx += 1
            kind = kind_cycle[kind_idx % len(kind_cycle)]
            tries += 1
        if kind_counts.get(kind, 0) >= kind_targets.get(kind, 0):
            # All filled by target; just use local as default.
            kind = "local"

        tmpl = rng.choice(templates_by_kind[kind])
        q = tmpl.format(p=phrase)
        if q in seen_questions:
            continue
        seen_questions.add(q)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        items.append(QAItem(qid=len(items), question=q, gold_chunk_id=cid, kind=kind, template=tmpl))
        kind_idx += 1

    # If dataset is small, allow repeats by sampling chunks
    while len(items) < n_questions and chunks:
        cid = rng.randrange(0, len(chunks))
        phrase_list = _extract_phrases(chunks[cid])
        phrase = phrase_list[0] if phrase_list else (chunks[cid][:10] or "內容")
        kind = rng.choice(list(templates_by_kind.keys()))
        tmpl = rng.choice(templates_by_kind[kind])
        q = tmpl.format(p=phrase)
        if q in seen_questions:
            continue
        seen_questions.add(q)
        items.append(QAItem(qid=len(items), question=q, gold_chunk_id=cid, kind=f"fallback:{kind}", template=tmpl))

    return items


def evaluate_retriever(
    *,
    questions: list[QAItem],
    retrieve_fn: Callable[[str, int], list[tuple[int, float]]],
    top_k: int = 5,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    hits = 0
    mrr_total = 0.0
    failures = 0

    t_all0 = time.perf_counter()

    for item in questions:
        t0 = time.perf_counter()
        try:
            ranked = retrieve_fn(item.question, top_k)
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)
        except Exception:
            dt = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt)
            failures += 1
            continue

        ids = [idx for idx, _score in ranked]
        if item.gold_chunk_id in ids:
            hits += 1
            rank = ids.index(item.gold_chunk_id) + 1
            mrr_total += 1.0 / rank

    latencies_ms.sort()

    def p(pct: float) -> float:
        if not latencies_ms:
            return 0.0
        k = int(round((pct / 100.0) * (len(latencies_ms) - 1)))
        return float(latencies_ms[max(0, min(k, len(latencies_ms) - 1))])

    t_all = max(1e-9, time.perf_counter() - t_all0)
    n = max(1, len(questions))
    return {
        "n": len(questions),
        "top_k": top_k,
        "recall_at_k": hits / n,
        "mrr": mrr_total / n,
        "failures": failures,
        "error_rate": failures / n,
        "qps": len(questions) / t_all,
        "latency_ms": {
            "p50": p(50),
            "p90": p(90),
            "p95": p(95),
            "p99": p(99),
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        },
    }


def write_report(report_dir: str, *, payload: dict[str, Any], report_id: str | None = None) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_id = report_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(report_dir) / f"benchmark_{report_id}.json"
    payload = dict(payload)
    payload.setdefault("report_id", report_id)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_questions_jsonl(report_dir: str, *, report_id: str, questions: list[QAItem]) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / f"questions_{report_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            rec = {
                "qid": q.qid,
                "question": q.question,
                "gold_chunk_id": q.gold_chunk_id,
                "kind": q.kind,
                "template": q.template,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return str(path)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    latency = metrics.get("latency_ms") or {}
    return {
        "n": metrics.get("n"),
        "top_k": metrics.get("top_k"),
        "recall_at_k": metrics.get("recall_at_k"),
        "mrr": metrics.get("mrr"),
        "failures": metrics.get("failures"),
        "error_rate": metrics.get("error_rate"),
        "qps": metrics.get("qps"),
        "latency_p50_ms": latency.get("p50"),
        "latency_p90_ms": latency.get("p90"),
        "latency_p95_ms": latency.get("p95"),
        "latency_p99_ms": latency.get("p99"),
        "latency_mean_ms": latency.get("mean"),
    }


def write_metrics_csv(report_dir: str, *, report_id: str, metrics: dict[str, Any]) -> str:
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    path = Path(report_dir) / f"benchmark_{report_id}.csv"
    row = _flatten_metrics(metrics)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return str(path)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
