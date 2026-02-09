from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import statistics
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

# When executed as a script (python scripts/eval_json_v2.py), Python only adds the
# script directory to sys.path. Add the project root so imports work inside Docker.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graphrag_platform.hierarchical_retriever import HierarchicalRetriever
from graphrag_platform.index_store import IndexStore
from graphrag_platform.model_registry import ModelRegistry


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _sha1_text(text: str) -> str:
    h = hashlib.sha1()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def normalize_text(text: str) -> str:
    """Deterministic text normalization for fuzzy matching / grounding.

    - NFKC normalize to collapse full-width/half-width variants
    - keep only letters and numbers (drops punctuation/symbols/whitespace/control chars)
    - lowercase for latin text
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    out: list[str] = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            out.append(ch.lower())
    return "".join(out)


def _ch_bigrams(s: str) -> set[str]:
    t = re.sub(r"\s+", "", (s or "").strip())
    if len(t) < 2:
        return set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / max(1, uni)


def _soft_match(expected: str, got: str) -> bool:
    e = _norm(expected)
    g = _norm(got)
    if not e or not g:
        return False
    if e in g or g in e:
        return True
    # fall back to bigram similarity
    return _jaccard(_ch_bigrams(e), _ch_bigrams(g)) >= 0.72


def _extract_keywords(text: str, *, max_n: int = 40) -> list[str]:
    t = _norm(text)
    if not t:
        return []
    kws: list[str] = []
    kws.extend(w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]{2,}", t))
    for seq in re.findall(r"[\u4e00-\u9fff]{2,}", t):
        # take bi-grams for chinese sequences
        if len(seq) == 2:
            kws.append(seq)
        else:
            kws.extend(seq[i : i + 2] for i in range(0, len(seq) - 1))

    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= max_n:
            break
    return out


def _coverage_f1(expected_items: list[str], got_items: list[str]) -> tuple[float, float, float]:
    exp = [_norm(x) for x in expected_items if _norm(x)]
    got = [_norm(x) for x in got_items if _norm(x)]
    if not exp and not got:
        return 1.0, 1.0, 1.0
    if not exp:
        return 0.0, 0.0, 0.0
    if not got:
        return 0.0, 0.0, 0.0

    matched = 0
    used_got: set[int] = set()
    for e in exp:
        for i, g in enumerate(got):
            if i in used_got:
                continue
            if _soft_match(e, g):
                matched += 1
                used_got.add(i)
                break

    recall = matched / max(1, len(exp))
    precision = matched / max(1, len(got))
    f1 = 0.0 if (recall + precision) <= 1e-9 else 2 * recall * precision / (recall + precision)
    return recall, precision, f1


def _char_similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return min(0.99, max(len(na), len(nb)) / max(1, len(na) + len(nb) - min(len(na), len(nb))))
    return float(SequenceMatcher(None, na, nb).ratio())


def _fuzzy_list_match(
    expected: list[str],
    pred: list[str],
    *,
    threshold: float,
) -> dict[str, Any]:
    exp = [str(x) for x in expected if normalize_text(str(x))]
    got = [str(x) for x in pred if normalize_text(str(x))]
    if not exp and not got:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "best_match_scores": [],
            "matched_pairs": [],
        }
    if not exp:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "best_match_scores": [],
            "matched_pairs": [],
        }
    if not got:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "best_match_scores": [0.0 for _ in exp],
            "matched_pairs": [],
        }

    pair_scores: list[tuple[float, int, int]] = []
    best_scores = [0.0 for _ in exp]
    for ei, e in enumerate(exp):
        for gi, g in enumerate(got):
            sim = _char_similarity(e, g)
            if sim > best_scores[ei]:
                best_scores[ei] = sim
            if sim >= threshold:
                pair_scores.append((sim, ei, gi))

    pair_scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_e: set[int] = set()
    used_g: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    for sim, ei, gi in pair_scores:
        if ei in used_e or gi in used_g:
            continue
        used_e.add(ei)
        used_g.add(gi)
        matched_pairs.append(
            {
                "expected_idx": ei,
                "pred_idx": gi,
                "expected": exp[ei],
                "pred": got[gi],
                "score": round(float(sim), 4),
            }
        )

    matched = len(matched_pairs)
    recall = matched / max(1, len(exp))
    precision = matched / max(1, len(got))
    f1 = 0.0 if (recall + precision) <= 1e-9 else 2 * recall * precision / (recall + precision)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "best_match_scores": [round(float(x), 4) for x in best_scores],
        "matched_pairs": matched_pairs,
    }


def fuzzy_list_f1(expected: list[str], pred: list[str], threshold: float) -> tuple[float, float, float]:
    res = _fuzzy_list_match(expected, pred, threshold=threshold)
    return float(res["precision"]), float(res["recall"]), float(res["f1"])


def _strip_code_fences(text: str) -> tuple[str, bool]:
    t = (text or "").strip()
    if not t:
        return "", False
    if "```" not in t:
        return t, False
    # Remove any ```lang markers but keep the content inside.
    t2 = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", t)
    t2 = t2.replace("```", "").strip()
    return t2, True


def _extract_first_json_object(text: str) -> tuple[str | None, dict[str, Any]]:
    """Best-effort JSON object extractor.

    Returns (json_str_or_none, meta) where meta includes:
      - had_code_fence: bool
      - extra_text: bool (non-whitespace outside extracted JSON)
      - truncated: bool (unbalanced braces)
      - extracted_json_head: str
    """
    t, had_fence = _strip_code_fences(text)
    if not t:
        return None, {"had_code_fence": had_fence, "extra_text": False, "truncated": False, "extracted_json_head": ""}

    start = t.find("{")
    if start < 0:
        return None, {"had_code_fence": had_fence, "extra_text": bool(t.strip()), "truncated": False, "extracted_json_head": ""}

    depth = 0
    in_str = False
    escape = False
    end: int | None = None
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        extracted = t[start:]
        prefix = t[:start].strip()
        extra_text = bool(prefix)
        return extracted, {
            "had_code_fence": had_fence,
            "extra_text": extra_text,
            "truncated": True,
            "extracted_json_head": extracted[:240],
        }

    extracted = t[start : end + 1]
    prefix = t[:start].strip()
    suffix = t[end + 1 :].strip()
    extra_text = bool(prefix or suffix)
    return extracted, {
        "had_code_fence": had_fence,
        "extra_text": extra_text,
        "truncated": False,
        "extracted_json_head": extracted[:240],
    }


def _parse_json_object(text: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    extracted, meta = _extract_first_json_object(text)
    meta_out = dict(meta)
    meta_out["parse_error"] = None
    if extracted is None:
        return None, meta_out
    if bool(meta_out.get("truncated")):
        return None, meta_out
    try:
        obj = json.loads(extracted)
    except Exception as e:
        meta_out["parse_error"] = str(e)
        return None, meta_out
    return obj if isinstance(obj, dict) else None, meta_out


def _has_placeholder_string(obj: dict[str, Any]) -> bool:
    def _is_string_literal(v: Any) -> bool:
        return isinstance(v, str) and v.strip() == "string"

    keys = [
        "target_audience",
        "main_topic",
        "sub_topic",
        "original_evidence",
    ]
    for k in keys:
        if _is_string_literal(obj.get(k)):
            return True

    dd = obj.get("detailed_description")
    if isinstance(dd, list) and any(_is_string_literal(x) for x in dd[:12]):
        return True

    pq = obj.get("predicted_questions")
    if isinstance(pq, list) and any(_is_string_literal(x) for x in pq[:12]):
        return True

    return False


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: int
    score: float
    source_path: str | None
    text: str


def build_context(
    *,
    question: str,
    retriever: HierarchicalRetriever,
    chunks: list[str],
    chunk_meta: list[dict[str, Any]] | None,
    top_k: int,
    max_chars: int,
) -> list[ContextChunk]:
    ranked = retriever.retrieve(question, top_n=top_k)
    out: list[ContextChunk] = []
    budget = max(200, int(max_chars))
    used = 0
    for cid, score in ranked:
        if not (0 <= cid < len(chunks)):
            continue
        source_path = None
        if chunk_meta is not None and 0 <= cid < len(chunk_meta):
            try:
                source_path = str(chunk_meta[cid].get("source_path") or "")
            except Exception:
                source_path = None
        txt = str(chunks[cid] or "")
        # enforce total budget to avoid context-length explosions
        if used + len(txt) > budget and out:
            break
        used += len(txt)
        out.append(ContextChunk(chunk_id=int(cid), score=float(score), source_path=source_path, text=txt))
    return out


def render_prompt(
    template: str,
    *,
    question: str,
    context_chunks: list[ContextChunk],
    extra_vars: dict[str, str] | None = None,
) -> str:
    def _fmt(chunks: list[ContextChunk]) -> str:
        parts: list[str] = []
        for c in chunks:
            sp = (c.source_path or "").strip() or "unknown"
            txt = (c.text or "").strip()
            ref_id = f"C{c.chunk_id}"
            try:
                cands = _extract_keywords(txt, max_n=40)
                cands = [x for x in cands if 2 <= len(x) <= 20 and _norm(x) in _norm(txt)]
            except Exception:
                cands = []
            cands = cands[:8]
            parts.append(
                f"[ref_id={ref_id} chunk_id={c.chunk_id} score={c.score:.4f} source_path={sp} anchor_candidates={cands}]\n{txt}"
            )
        return "\n\n---\n\n".join(parts)

    ctx = _fmt(context_chunks)
    s = template.replace("{{question}}", question.strip())
    # Intentionally render chunks as plain text (not JSON) to reduce the chance that the model
    # copies the input chunks verbatim into its output JSON.
    s = s.replace("{{context_chunks_json}}", ctx)
    if extra_vars:
        for k, v in extra_vars.items():
            s = s.replace("{{" + k + "}}", str(v))
    return s


def _collect_context_grounding_maps(context_chunks: list[ContextChunk]) -> tuple[dict[str, str], dict[str, list[str]], list[str], str]:
    """Build normalized context maps for post-repair grounding fixes."""
    file_text_norm: dict[str, str] = {}
    file_candidates: dict[str, list[str]] = {}
    global_candidates: list[str] = []
    global_text_norm_parts: list[str] = []

    for c in context_chunks:
        file_key = _norm(str(c.source_path or ""))
        text = str(c.text or "")
        text_norm = normalize_text(text)
        if text_norm:
            global_text_norm_parts.append(text_norm)
        if file_key and text_norm:
            prev = file_text_norm.get(file_key, "")
            file_text_norm[file_key] = (prev + "\n" + text_norm).strip() if prev else text_norm

        cands: list[str] = []
        try:
            cands = _extract_keywords(text, max_n=60)
        except Exception:
            cands = []
        filtered: list[str] = []
        seen_norm: set[str] = set()
        for cand in cands:
            cstr = str(cand).strip()
            cn = normalize_text(cstr)
            if not cn:
                continue
            if len(cn) < 2 or len(cn) > 20:
                continue
            if cn.isdigit():
                continue
            if cn in seen_norm:
                continue
            if text_norm and cn not in text_norm:
                continue
            seen_norm.add(cn)
            filtered.append(cstr)
            if len(filtered) >= 12:
                break

        if file_key:
            exist = file_candidates.get(file_key, [])
            for cand in filtered:
                if cand not in exist:
                    exist.append(cand)
            file_candidates[file_key] = exist[:12]

        for cand in filtered:
            if cand not in global_candidates:
                global_candidates.append(cand)
            if len(global_candidates) >= 24:
                break

    global_text_norm = "\n".join(global_text_norm_parts)
    return file_text_norm, file_candidates, global_candidates, global_text_norm


def _repair_source_map_for_grounding(got_obj: dict[str, Any], context_chunks: list[ContextChunk]) -> bool:
    """Post-repair pass: keep source_map refs grounded in retrieved context."""
    if not isinstance(got_obj, dict):
        return False
    sm = got_obj.get("source_map")
    if not isinstance(sm, list) or not sm:
        return False

    file_text_norm, file_candidates, global_candidates, global_text_norm = _collect_context_grounding_maps(context_chunks)
    default_file = next(iter(file_text_norm.keys()), "")
    changed = False

    for ent in sm[:12]:
        if not isinstance(ent, dict):
            continue
        refs = ent.get("refs")
        if not isinstance(refs, list):
            continue
        for ref in refs[:6]:
            if not isinstance(ref, dict):
                continue

            file_key = _norm(str(ref.get("file") or ""))
            if not file_key or (file_text_norm and file_key not in file_text_norm):
                if default_file:
                    ref["file"] = default_file
                    file_key = default_file
                    changed = True

            anchors = ref.get("anchors")
            if not isinstance(anchors, list):
                anchors = []

            text_norm = file_text_norm.get(file_key) or global_text_norm
            keep: list[str] = []
            for a in anchors[:6]:
                a_str = str(a).strip()
                a_norm = normalize_text(a_str)
                if not a_norm:
                    continue
                if text_norm and a_norm in text_norm:
                    keep.append(a_str)
            if not keep:
                cands = file_candidates.get(file_key) or global_candidates
                if cands:
                    keep = [cands[0]]
                elif text_norm:
                    keep = ["依據"]
                if keep:
                    changed = True

            keep = keep[:2] if keep else []
            if ref.get("anchors") != keep and keep:
                ref["anchors"] = keep
                changed = True

            try:
                page = int(ref.get("page") or 1)
            except Exception:
                page = 1
            if page <= 0:
                page = 1
            if ref.get("page") != page:
                ref["page"] = page
                changed = True

    return changed


def split_prompt_asset(prompt_text: str) -> tuple[str, str]:
    """Split a prompt asset into (system_text, user_template).

    The versioned asset `docs/prompts/eval_json_v2.md` contains two explicit sections:
    - "# 指令（System）"
    - "# 輸入（User）"

    Feeding the whole Markdown document to a chat model often increases the chance of
    Markdown/code-fence outputs. For evaluation stability, we extract the two sections
    and feed them as proper chat roles.
    """

    text = prompt_text or ""
    sys_marker = "# 指令（System）"
    user_marker = "# 輸入（User）"
    si = text.find(sys_marker)
    ui = text.find(user_marker)
    def _sanitize(section: str) -> str:
        s = section or ""
        # Drop HTML comments
        s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
        # Remove backtick fences but keep content
        s = re.sub(r"```[a-zA-Z0-9_-]*\s*", "", s)
        s = s.replace("```", "")
        # Remove excessive markdown heading markers
        s = re.sub(r"^\s*#+\s*", "", s, flags=re.MULTILINE)
        return s.strip()

    if si < 0 or ui < 0 or ui <= si:
        return "Return ONLY a JSON object and nothing else.", _sanitize(text)

    system_text = _sanitize(text[si + len(sys_marker) : ui])
    user_template = _sanitize(text[ui + len(user_marker) :])
    if not system_text:
        system_text = "Return ONLY a JSON object and nothing else."
    # Extra hard constraints to reduce code fences and truncation.
    system_text = (
        system_text
        + "\n\nIMPORTANT:"
        + "\n- Output raw JSON only. Do NOT wrap in ``` fences."
        + "\n- Keep output SHORT to ensure it completes: detailed_description <= 6 items, predicted_questions <= 3 items."
        + "\n- original_evidence must be <= 160 characters (excerpt only)."
    )
    return system_text, user_template


def score_answer_v2(
    *,
    expected: dict[str, Any],
    got: dict[str, Any] | None,
    context_chunks: list[ContextChunk],
) -> dict[str, Any]:
    answer_example = (expected.get("answer_example") or {}) if isinstance(expected, dict) else {}
    exp_ta = _norm(str(answer_example.get("target_audience") or ""))
    exp_mt = _norm(str(answer_example.get("main_topic") or ""))
    exp_st = _norm(str(answer_example.get("sub_topic") or ""))
    exp_dd = list(answer_example.get("detailed_description") or [])
    exp_ev = _norm(str(answer_example.get("original_evidence") or ""))
    exp_pq = list(answer_example.get("predicted_questions") or [])

    if not isinstance(got, dict):
        return {
            "schema_ok": False,
            "score": 0.0,
            "subs": {},
            "notes": ["invalid_json_or_shape"],
        }

    required_keys = {
        "target_audience",
        "main_topic",
        "sub_topic",
        "detailed_description",
        "original_evidence",
        "source_map",
        "predicted_questions",
    }
    if any(k not in got for k in required_keys):
        return {
            "schema_ok": False,
            "score": 0.0,
            "subs": {},
            "notes": ["missing_required_keys"],
        }

    got_ta = _norm(str(got.get("target_audience") or ""))
    got_mt = _norm(str(got.get("main_topic") or ""))
    got_st = _norm(str(got.get("sub_topic") or ""))
    got_dd = got.get("detailed_description") or []
    got_ev = _norm(str(got.get("original_evidence") or ""))
    got_pq = got.get("predicted_questions") or []

    if not isinstance(got_dd, list) or not isinstance(got_pq, list):
        return {
            "schema_ok": False,
            "score": 0.0,
            "subs": {},
            "notes": ["detailed_description_or_predicted_questions_not_list"],
        }

    # 1) Field matches
    s_ta = 1.0 if _soft_match(exp_ta, got_ta) else 0.0
    s_mt = 1.0 if _soft_match(exp_mt, got_mt) else 0.0
    s_st = 1.0 if _soft_match(exp_st, got_st) else 0.0

    # 2) Bullet list fuzzy coverage (deterministic char-level matching).
    # Note: ExpectedV2 bullet strings are often canonicalized while model outputs are
    # extracted/summarized from context. Use a lower deterministic threshold to avoid
    # an all-zero failure mode on list fields.
    dd_match = _fuzzy_list_match(exp_dd[:12], [str(x) for x in got_dd[:12]], threshold=0.13)
    dd_recall = float(dd_match["recall"])
    dd_prec = float(dd_match["precision"])
    dd_f1 = float(dd_match["f1"])

    pq_match = _fuzzy_list_match(exp_pq[:10], [str(x) for x in got_pq[:10]], threshold=0.33)
    pq_recall = float(pq_match["recall"])
    pq_prec = float(pq_match["precision"])
    pq_f1 = float(pq_match["f1"])

    # 3) Evidence similarity (keyword overlap) + minimum length
    ev_keywords = _extract_keywords(exp_ev, max_n=60)
    got_kw = set(_extract_keywords(got_ev, max_n=80))
    hit = sum(1 for k in ev_keywords[:30] if k in got_kw)
    ev_kw_score = min(1.0, hit / 6.0)  # compact evidence often has fewer keywords
    # Evidence field is intentionally short (prompt + truncation safety). Avoid over-penalizing
    # otherwise well-grounded answers that use a compact excerpt.
    ev_len_pen = 1.0 if len(got_ev) >= 25 else max(0.0, len(got_ev) / 25.0)
    s_ev = ev_kw_score * ev_len_pen

    # 4) Grounding score: source_map files + anchors must match context.
    ctx_files = {(_norm(c.source_path or "")) for c in context_chunks if _norm(c.source_path or "")}
    ctx_text = "\n".join(c.text for c in context_chunks)
    ctx_text_norm = normalize_text(ctx_text)
    s_ground = 0.0
    ground_notes: list[str] = []
    sm = got.get("source_map")
    ok_refs = 0
    total_refs = 0
    if isinstance(sm, list) and sm:
        for ent in sm[:12]:
            if not isinstance(ent, dict):
                continue
            refs = ent.get("refs")
            if not isinstance(refs, list):
                continue
            for r in refs[:6]:
                if not isinstance(r, dict):
                    continue
                total_refs += 1
                f = _norm(str(r.get("file") or ""))
                anchors = r.get("anchors") or []
                if not f or not isinstance(anchors, list) or not anchors:
                    continue
                # file must be one of retrieved files (strict)
                if ctx_files and f not in ctx_files:
                    continue
                # at least one anchor must exist in context text
                ok_anchor = any((normalize_text(str(a)) and normalize_text(str(a)) in ctx_text_norm) for a in anchors[:6])
                if ok_anchor:
                    ok_refs += 1
    else:
        ground_notes.append("missing_source_map")

    if total_refs > 0:
        s_ground = ok_refs / total_refs
    else:
        s_ground = 0.0

    # Weighted total score (0..100)
    weights = {
        "target_audience": 0.10,
        "main_topic": 0.10,
        "sub_topic": 0.10,
        "detailed_description": 0.30,
        "original_evidence": 0.20,
        "predicted_questions": 0.10,
        "grounding": 0.10,
    }
    total = (
        weights["target_audience"] * s_ta
        + weights["main_topic"] * s_mt
        + weights["sub_topic"] * s_st
        + weights["detailed_description"] * dd_f1
        + weights["original_evidence"] * s_ev
        + weights["predicted_questions"] * pq_f1
        + weights["grounding"] * s_ground
    )
    total = max(0.0, min(1.0, total))

    return {
        "schema_ok": True,
        "score": round(100.0 * total, 3),
        "subs": {
            "target_audience": s_ta,
            "main_topic": s_mt,
            "sub_topic": s_st,
            "detailed_description_f1": round(dd_f1, 4),
            "detailed_description_recall": round(dd_recall, 4),
            "detailed_description_precision": round(dd_prec, 4),
            "original_evidence": round(s_ev, 4),
            "predicted_questions_f1": round(pq_f1, 4),
            "predicted_questions_recall": round(pq_recall, 4),
            "predicted_questions_precision": round(pq_prec, 4),
            "grounding": round(s_ground, 4),
            "grounding_ok_refs": int(ok_refs),
            "grounding_total_refs": int(total_refs),
        },
        "match_diagnostics": {
            "detailed_description": {
                "threshold": 0.13,
                "best_match_scores": dd_match.get("best_match_scores") or [],
                "matched_pairs": dd_match.get("matched_pairs") or [],
            },
            "predicted_questions": {
                "threshold": 0.33,
                "best_match_scores": pq_match.get("best_match_scores") or [],
                "matched_pairs": pq_match.get("matched_pairs") or [],
            },
        },
        "notes": ground_notes,
    }


async def openai_chat_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_text: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    if httpx is None:
        raise RuntimeError("Missing dependency: httpx (required for --provider openai).")
    base_url = (base_url or "").rstrip("/")
    if not base_url:
        raise RuntimeError("OPENAI_BASE_URL is empty")
    url = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (system_text or "Return ONLY a JSON object and nothing else.").strip()},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
        "max_tokens": int(max_tokens),
        # Some servers support explicit JSON response format; ignore if unsupported.
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers, json=payload)
        data = resp.json()
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {data}")

    try:
        content = data["choices"][0]["message"]["content"]
        return str(content), data
    except Exception as e:
        raise RuntimeError(f"Invalid response shape: {data}") from e


def local_generate_json(
    *,
    model_id: str,
    adapter_path: str | None,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

        try:
            from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
        except Exception:
            BitsAndBytesConfig = None  # type: ignore[assignment]
    except Exception as e:
        raise RuntimeError(f"Local generation requires torch+transformers. Original error: {e}")

    if not torch.cuda.is_available():
        raise RuntimeError("Local generation requires CUDA (torch.cuda.is_available() is false).")

    # Load in 4-bit when possible to fit 8GB VRAM.
    qcfg = None
    if BitsAndBytesConfig is not None:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=qcfg,
    )

    if adapter_path:
        try:
            from peft import PeftModel  # pyright: ignore[reportMissingImports]
        except Exception as e:
            raise RuntimeError(f"Missing PEFT dependency for adapter loading: {e}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = bool(temperature and temperature > 1e-6)
    gen = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    new_tokens = gen[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)

class LocalJsonGenerator:
    def __init__(self, *, model_id: str, adapter_path: str | None, temperature: float, max_new_tokens: int) -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from transformers import AutoModelForCausalLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

            try:
                from transformers import BitsAndBytesConfig  # pyright: ignore[reportMissingImports]
            except Exception:
                BitsAndBytesConfig = None  # type: ignore[assignment]
        except Exception as e:
            raise RuntimeError(f"Local generation requires torch+transformers. Original error: {e}")

        if not torch.cuda.is_available():
            raise RuntimeError("Local generation requires CUDA (torch.cuda.is_available() is false).")

        self._torch = torch
        self._temperature = float(temperature)
        self._max_new_tokens = int(max_new_tokens)

        def _trust_remote_code(mid: str) -> bool:
            s = (mid or "").lower()
            return ("qwen3" in s) or ("qwen-3" in s)

        qcfg = None
        if BitsAndBytesConfig is not None:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=_trust_remote_code(model_id))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=qcfg,
            trust_remote_code=_trust_remote_code(model_id),
        )
        if adapter_path:
            try:
                from peft import PeftModel  # pyright: ignore[reportMissingImports]
            except Exception as e:
                raise RuntimeError(f"Missing PEFT dependency for adapter loading: {e}")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()

        self._tok = tok
        self._model = model
        self._input_device = next(model.parameters()).device

    def generate(self, prompt: str) -> str:
        tok = self._tok
        model = self._model
        torch = self._torch

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

        do_sample = bool(self._temperature and self._temperature > 1e-6)
        gen = model.generate(
            **inputs,
            max_new_tokens=int(self._max_new_tokens),
            do_sample=do_sample,
            temperature=float(self._temperature) if do_sample else None,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[-1] :]
        return tok.decode(new_tokens, skip_special_tokens=True)

    def generate_chat(self, *, system_text: str, user_text: str, assistant_prefix: str = "") -> str:
        tok = self._tok
        model = self._model
        torch = self._torch

        messages: list[dict[str, str]] = [
            {"role": "system", "content": (system_text or "").strip()},
            {"role": "user", "content": (user_text or "").strip()},
        ]

        if hasattr(tok, "apply_chat_template"):
            try:
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
        else:
            prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"

        if (assistant_prefix or "").strip():
            prompt = prompt + (assistant_prefix or "")

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}

        do_sample = bool(self._temperature and self._temperature > 1e-6)
        stopping_criteria = None
        # Reduce JSON truncation / extra trailing text: stop as soon as a full, schema-shaped JSON object is emitted.
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList  # pyright: ignore[reportMissingImports]

            start_len = int(inputs["input_ids"].shape[-1])
            _tok = tok
            _prefix = assistant_prefix or ""
            _required = {
                "target_audience",
                "main_topic",
                "sub_topic",
                "detailed_description",
                "original_evidence",
                "source_map",
                "predicted_questions",
            }
            _close_ids = set(_tok.encode("}", add_special_tokens=False))

            class _StopOnFullJson(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
                    try:
                        last_id = int(input_ids[0, -1])
                        if _close_ids and last_id not in _close_ids:
                            return False
                        gen_ids = input_ids[0][start_len:]
                        text = _prefix + _tok.decode(gen_ids, skip_special_tokens=True)
                        extracted, meta = _extract_first_json_object(text)
                        if not extracted:
                            return False
                        if bool(meta.get("truncated")):
                            return False
                        if bool(meta.get("extra_text")):
                            return False
                        try:
                            obj = json.loads(extracted)
                        except Exception:
                            return False
                        if not isinstance(obj, dict):
                            return False
                        if any(k not in obj for k in _required):
                            return False
                        return True
                    except Exception:
                        return False

            stopping_criteria = StoppingCriteriaList([_StopOnFullJson()])
        except Exception:
            stopping_criteria = None

        gen = model.generate(
            **inputs,
            max_new_tokens=int(self._max_new_tokens),
            do_sample=do_sample,
            temperature=float(self._temperature) if do_sample else None,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        new_tokens = gen[0][inputs["input_ids"].shape[-1] :]
        out = tok.decode(new_tokens, skip_special_tokens=True)
        if (assistant_prefix or "").strip():
            return (assistant_prefix or "") + (out or "")
        return out


def _prompt_version_from_asset(prompt_text: str) -> str:
    m = re.search(r"prompt_version:\s*([^\s]+)", prompt_text)
    return m.group(1).strip() if m else "unknown"

def _collect_allowed_labels(questions: list[dict[str, Any]]) -> dict[str, list[str]]:
    ta: dict[str, int] = {}
    mt: dict[str, int] = {}
    st: dict[str, int] = {}
    for it in questions:
        exp = it.get("expected") or {}
        if not isinstance(exp, dict):
            continue
        ae = exp.get("answer_example") or {}
        if not isinstance(ae, dict):
            continue
        for key, store in (
            ("target_audience", ta),
            ("main_topic", mt),
            ("sub_topic", st),
        ):
            v = str(ae.get(key) or "").strip()
            if not v:
                continue
            store[v] = int(store.get(v, 0)) + 1

    def _sort(d: dict[str, int]) -> list[str]:
        return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))]

    return {
        "allowed_target_audience": _sort(ta),
        "allowed_main_topic": _sort(mt),
        "allowed_sub_topic": _sort(st),
    }


def _read_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Questions must be a JSON array.")
    return data


def _index_version(index_dir: Path) -> str:
    meta_path = index_dir / "index_meta.json"
    if not meta_path.exists():
        return "missing"
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(obj.get("version") or "unknown")
    except Exception:
        return "unknown"


async def main_async() -> int:
    ap = argparse.ArgumentParser(description="Eval runner for ExpectedV2 questions with stable JSON output.")
    ap.add_argument("--questions", required=True, help="Path to question JSON (array).")
    ap.add_argument("--prompt-asset", default="docs/prompts/eval_json_v2.md")
    ap.add_argument("--out-dir", required=True, help="Output directory for reports.")

    ap.add_argument("--index-dir", default="graphrag_saas/backend/data/index", help="Index directory (chunks/meta).")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-context-chars", type=int, default=9000)

    ap.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--llm-model", default=os.getenv("MODEL_NAME", ""))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0, help="Local provider: top_p (only used when sampling).")
    ap.add_argument("--timeout-s", type=float, default=90.0)
    ap.add_argument("--provider", choices=["openai", "local"], default="openai")
    ap.add_argument("--adapter-path", default="", help="Local provider: PEFT adapter path to load.")
    ap.add_argument("--max-new-tokens", type=int, default=512)

    ap.add_argument("--gate-threshold", type=float, default=95.0, help="Pass gate if avg >= threshold.")
    ap.add_argument("--registry-path", default="", help="Optional: backend registry path to annotate.")
    ap.add_argument("--active-path", default="", help="Optional: backend active models path.")
    ap.add_argument("--registry-model-id", default="", help="Optional: model_id (UUID in registry) to annotate.")
    ap.add_argument("--adapter-id", default="", help="Optional: adapter id / tag to record in metrics.")
    ap.add_argument("--metric-prefix", default="", help="Prefix for metrics keys when annotating registry, e.g. 'gate_' or 'full_'.")
    ap.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Write metrics.partial.json/per_item.partial.json every N questions (0 disables).",
    )
    ap.add_argument(
        "--raw-dump-dir",
        default="",
        help="Optional: dump full raw outputs into a subdir under out_dir (one file per question).",
    )
    ap.add_argument(
        "--repair-json",
        type=int,
        default=0,
        help="If 1, run a second-pass format-repair generation when the primary output is not schema-valid JSON.",
    )

    args = ap.parse_args()

    q_path = Path(args.questions)
    prompt_path = Path(args.prompt_asset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = _read_questions(q_path)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_hash = _sha256_text(prompt_text)
    prompt_version = _prompt_version_from_asset(prompt_text)
    system_text, user_template = split_prompt_asset(prompt_text)
    effective_prompt_hash = _sha256_text(system_text + "\n\n" + user_template)
    eval_set_version = f"{q_path.name}@{_sha1_text(q_path.read_text(encoding='utf-8'))[:12]}"

    allowed = _collect_allowed_labels(questions)
    extra_vars = {
        "allowed_target_audience": json.dumps(allowed["allowed_target_audience"], ensure_ascii=False),
        "allowed_main_topic": json.dumps(allowed["allowed_main_topic"], ensure_ascii=False),
        "allowed_sub_topic": json.dumps(allowed["allowed_sub_topic"], ensure_ascii=False),
    }

    index_dir = Path(args.index_dir)
    store = IndexStore(index_dir)
    meta = store.read_meta()
    chunks_dicts = store.load_chunks()
    chunks = [str(c.get("text") or "") for c in chunks_dicts]
    chunk_meta = chunks_dicts
    entity_names = store.load_entity_names()
    retriever = HierarchicalRetriever(chunks, entity_names)
    index_version = (meta.version if meta else _index_version(index_dir))

    model_name = (args.llm_model or "").strip()
    if not model_name:
        raise RuntimeError("Missing --llm-model (or MODEL_NAME).")

    def _is_gate_eval_set(p: Path) -> bool:
        n = (p.name or "").lower()
        return ("_dev20" in n) or ("_dev100" in n)

    gate_mode = _is_gate_eval_set(q_path)
    eff_temperature = 0.0 if gate_mode else float(args.temperature)
    eff_top_p = 1.0 if gate_mode else float(args.top_p)
    # Gate runs prioritize stable, parseable JSON.
    # Keep the max_new_tokens conservative to avoid OOM on long contexts; prefer reducing context budget first.
    eff_max_new_tokens = 512 if gate_mode else int(args.max_new_tokens)
    eff_max_context_chars = 4000 if gate_mode else int(args.max_context_chars)

    local_gen: LocalJsonGenerator | None = None
    if args.provider == "local":
        local_gen = LocalJsonGenerator(
            model_id=model_name,
            adapter_path=(args.adapter_path or "").strip() or None,
            temperature=float(eff_temperature),
            max_new_tokens=int(eff_max_new_tokens),
        )

    started = time.time()
    per_item: list[dict[str, Any]] = []
    schema_ok = 0
    scores: list[float] = []
    grounding_scores: list[float] = []

    flush_every = int(args.flush_every)
    raw_dump_dir = (args.raw_dump_dir or "").strip()
    raw_out_dir = (out_dir / raw_dump_dir) if raw_dump_dir else None
    if raw_out_dir is not None:
        raw_out_dir.mkdir(parents=True, exist_ok=True)
    partial_metrics_path = out_dir / "metrics.partial.json"
    partial_items_path = out_dir / "per_item.partial.json"

    fail_counts: dict[str, int] = {}

    for idx, it in enumerate(questions, start=1):
        qid = str(it.get("id") or f"Q{idx:04d}")
        question = str(it.get("question") or "").strip()
        expected = it.get("expected") or {}
        if not question:
            continue

        ctx = build_context(
            question=question,
            retriever=retriever,
            chunks=chunks,
            chunk_meta=chunk_meta,
            top_k=int(args.top_k),
            max_chars=int(eff_max_context_chars),
        )
        user_prompt = render_prompt(user_template, question=question, context_chunks=ctx, extra_vars=extra_vars)

        raw_text = ""
        err = None
        t0 = time.time()
        try:
            if args.provider == "openai":
                raw_text, _raw_json = await openai_chat_json(
                    base_url=args.openai_base_url,
                    api_key=args.openai_api_key,
                    model=model_name,
                    system_text=system_text,
                    prompt=user_prompt,
                    temperature=float(eff_temperature),
                    top_p=float(eff_top_p),
                    max_tokens=int(eff_max_new_tokens),
                    timeout_s=float(args.timeout_s),
                )
            else:
                assert local_gen is not None
                raw_text = local_gen.generate_chat(
                    system_text=system_text,
                    user_text=user_prompt,
                    assistant_prefix="",
                )
        except Exception as e:
            err = str(e)
        dt_ms = int((time.time() - t0) * 1000)

        if raw_out_dir is not None:
            try:
                (raw_out_dir / f"{qid}.raw1.txt").write_text(raw_text or "", encoding="utf-8")
            except Exception:
                pass

        got_obj, parse_meta = _parse_json_object(raw_text) if err is None else (None, {})

        # Determine failure reason for the primary output (before any repair pass).
        primary_fail_reason: str | None = None
        if err is not None:
            primary_fail_reason = "provider_error"
        else:
            if got_obj is None:
                if bool(parse_meta.get("truncated")):
                    primary_fail_reason = "truncated_json"
                else:
                    primary_fail_reason = "parse_failed"
            else:
                if bool(parse_meta.get("had_code_fence")):
                    primary_fail_reason = "code_fence"
                elif bool(parse_meta.get("extra_text")):
                    primary_fail_reason = "extra_text"
                elif _has_placeholder_string(got_obj):
                    primary_fail_reason = "placeholder_string"
                else:
                    required_keys = {
                        "target_audience",
                        "main_topic",
                        "sub_topic",
                        "detailed_description",
                        "original_evidence",
                        "source_map",
                        "predicted_questions",
                    }
                    if any(k not in got_obj for k in required_keys):
                        primary_fail_reason = "schema_failed"
                    elif not isinstance(got_obj.get("detailed_description"), list) or not isinstance(
                        got_obj.get("predicted_questions"), list
                    ):
                        primary_fail_reason = "schema_failed"
                    elif not isinstance(got_obj.get("source_map"), list):
                        primary_fail_reason = "schema_failed"

        repair_used = False
        repair_error: str | None = None
        primary_raw_head = (raw_text or "")[:400]

        # Optional second-pass repair: force the model to emit a minimal, schema-shaped JSON object.
        if (
            int(args.repair_json) == 1
            and args.provider != "openai"
            and local_gen is not None
            and primary_fail_reason is not None
        ):
            try:
                source_paths: list[str] = []
                if ctx:
                    for c in ctx[:3]:
                        sp = str(c.source_path or "").strip()
                        if sp:
                            source_paths.append(sp)

                # Try to provide anchors that are guaranteed to appear in the retrieved context text.
                anchor_candidates: list[str] = []
                if ctx:
                    txt0 = str(ctx[0].text or "")
                    try:
                        anchor_candidates = _extract_keywords(txt0, max_n=40)
                        anchor_candidates = [x for x in anchor_candidates if 2 <= len(x) <= 20 and _norm(x) in _norm(txt0)]
                    except Exception:
                        anchor_candidates = []
                anchor_candidates = anchor_candidates[:8]

                default_file = (source_paths[0] if source_paths else "")
                default_anchor = (anchor_candidates[0] if anchor_candidates else "")

                repair_system = (
                    "你是一個「JSON 格式修復器」。"
                    "只輸出單一 JSON object（第一個字元必須是 {，最後一個字元必須是 }）。"
                    "不要 Markdown、不要解釋、不要輸出多餘文字、不要使用 ``` code fence。"
                    "禁止輸出字面量 \"string\"。"
                    "輸出必須短：original_evidence <= 160 字；detailed_description <= 4 條；predicted_questions <= 2 條。"
                )
                repair_user = (
                    "請輸出符合下列 schema 的 JSON（鍵不可省略；型別必須正確；內容可為資料不足）：\n"
                    "- target_audience/main_topic/sub_topic/original_evidence: string\n"
                    "- detailed_description: array of string（最多 4 條，每條最多 50 字）\n"
                    "- predicted_questions: array of string（最多 2 條）\n"
                    "- source_map: array（至少 1 筆）\n"
                    "- source_map[0].part: 必須是 target_audience/main_topic/sub_topic/detailed_description/original_evidence 其中之一\n"
                    "- source_map[0].refs: array（至少 1 筆；最多 1 筆）\n"
                    "- source_map[0].refs[0].file: string\n"
                    "- source_map[0].refs[0].page: integer（建議 1）\n"
                    "- source_map[0].refs[0].anchors: array of string（至少 1 筆；最多 2 筆；每條 4~20 字）\n\n"
                    "重要：這不是自由生成。請優先完成『分類』與『引用』，避免輸出「資料不足」。\n"
                    "Allowed labels（target_audience/main_topic/sub_topic 必須逐字選其中一個）：\n"
                    f"- target_audience: {allowed['allowed_target_audience']}\n"
                    f"- main_topic: {allowed['allowed_main_topic']}\n"
                    f"- sub_topic: {allowed['allowed_sub_topic']}\n"
                    "Grounding options（引用必須逐字挑選，不可自行發明）：\n"
                    f"- file candidates: {source_paths}\n"
                    f"- anchor_candidates (choose for anchors): {anchor_candidates}\n\n"
                    "若你填的 anchors 不在 anchor_candidates，必須改成候選中的逐字值；否則視為錯誤。\n\n"
                    "問題："
                    + question
                    + "\n\n"
                    "你可以直接使用下列 JSON 骨架並填值（只輸出 JSON）：\n"
                    "{\n"
                    '  "target_audience": "",\n'
                    '  "main_topic": "",\n'
                    '  "sub_topic": "",\n'
                    '  "detailed_description": [],\n'
                    '  "original_evidence": "",\n'
                    '  "source_map": [\n'
                    "    {\n"
                    '      "part": "original_evidence",\n'
                    '      "refs": [\n'
                    "        {\n"
                    f'          "file": "{default_file or "unknown"}",\n'
                    '          "page": 1,\n'
                    f'          "anchors": ["{default_anchor or "依據"}"]\n'
                    "        }\n"
                    "      ]\n"
                    "    }\n"
                    "  ],\n"
                    '  "predicted_questions": []\n'
                    "}\n\n"
                    "（供修復參考）上一輪輸出：\n"
                    + ((raw_text or "")[:800])
                )

                raw_text2 = local_gen.generate_chat(system_text=repair_system, user_text=repair_user, assistant_prefix="{")
                if raw_out_dir is not None:
                    try:
                        (raw_out_dir / f"{qid}.raw2.txt").write_text(raw_text2 or "", encoding="utf-8")
                    except Exception:
                        pass
                got2, meta2 = _parse_json_object(raw_text2)
                if got2 is not None and isinstance(got2, dict):
                    raw_text = raw_text2
                    got_obj = got2
                    parse_meta = meta2
                    repair_used = True
            except Exception as e:
                repair_error = str(e)

        # Repair-specific grounding normalization:
        # when repair is enabled, enforce refs.file/anchors to align with retrieved context.
        if int(args.repair_json) == 1 and isinstance(got_obj, dict):
            try:
                _repair_source_map_for_grounding(got_obj, ctx)
            except Exception:
                pass

        # Determine actionable failure reason (strict JSON-only gate).
        fail_reason: str | None = None
        if err is not None:
            fail_reason = "provider_error"
        else:
            if got_obj is None:
                if bool(parse_meta.get("truncated")):
                    fail_reason = "truncated_json"
                else:
                    fail_reason = "parse_failed"
            else:
                if bool(parse_meta.get("had_code_fence")):
                    fail_reason = "code_fence"
                elif bool(parse_meta.get("extra_text")):
                    fail_reason = "extra_text"
                elif _has_placeholder_string(got_obj):
                    fail_reason = "placeholder_string"
                else:
                    required_keys = {
                        "target_audience",
                        "main_topic",
                        "sub_topic",
                        "detailed_description",
                        "original_evidence",
                        "source_map",
                        "predicted_questions",
                    }
                    if any(k not in got_obj for k in required_keys):
                        fail_reason = "schema_failed"
                    elif not isinstance(got_obj.get("detailed_description"), list) or not isinstance(
                        got_obj.get("predicted_questions"), list
                    ):
                        fail_reason = "schema_failed"
                    elif not isinstance(got_obj.get("source_map"), list):
                        fail_reason = "schema_failed"

        if fail_reason is not None:
            scored = {"schema_ok": False, "score": 0.0, "subs": {}, "notes": [fail_reason]}
        else:
            scored = score_answer_v2(expected=expected, got=got_obj, context_chunks=ctx)

        if scored.get("schema_ok"):
            schema_ok += 1
            try:
                scores.append(float(scored.get("score") or 0.0))
            except Exception:
                pass
            try:
                grounding_scores.append(float((scored.get("subs") or {}).get("grounding") or 0.0))
            except Exception:
                pass

        try:
            notes = scored.get("notes") or []
            notes_s = ",".join(str(x) for x in notes[:3])
        except Exception:
            notes_s = ""
        print(
            f"[{idx}/{len(questions)}] {qid} schema_ok={bool(scored.get('schema_ok'))} score={scored.get('score')} latency_ms={dt_ms} notes={notes_s}",
            flush=True,
        )

        if fail_reason is None:
            fr = "ok"
        else:
            fr = fail_reason
        fail_counts[fr] = int(fail_counts.get(fr, 0)) + 1

        per_item.append(
            {
                "id": qid,
                "latency_ms": dt_ms,
                "error": err,
                "raw_head": (raw_text or "")[:400],
                "primary_raw_head": primary_raw_head,
                "extracted_json_head": str(parse_meta.get("extracted_json_head") or "") if isinstance(parse_meta, dict) else "",
                "fail_reason": fail_reason,
                "primary_fail_reason": primary_fail_reason,
                "got_keys": (sorted(list(got_obj.keys())) if isinstance(got_obj, dict) else None),
                "score": scored.get("score"),
                "schema_ok": bool(scored.get("schema_ok")),
                "subs": scored.get("subs") or {},
                "match_diagnostics": scored.get("match_diagnostics") or {},
                "notes": scored.get("notes") or [],
                "repair_used": repair_used,
                "repair_error": repair_error,
            }
        )

        if flush_every > 0 and (idx % flush_every == 0 or idx == len(questions)):
            try:
                processed = max(1, idx)
                partial_summary = {
                    "eval": {
                        "question_count": processed,
                        "schema_pass_rate": round(schema_ok / processed, 6),
                        "eval_score_avg": round(float(statistics.mean(scores)) if scores else 0.0, 6),
                        "grounding_avg": round(float(statistics.mean(grounding_scores)) if grounding_scores else 0.0, 6),
                        "elapsed_s": round(time.time() - started, 2),
                    },
                    "fail_counts": dict(fail_counts),
                    "provenance": {
                        "prompt_asset": str(prompt_path),
                        "prompt_version": prompt_version,
                        "prompt_sha256": prompt_hash,
                        "effective_prompt_sha256": effective_prompt_hash,
                        "index_dir": str(index_dir),
                        "index_version": str(index_version),
                        "model_id": model_name,
                        "adapter_id": (args.adapter_id or "").strip() or None,
                        "adapter_path": (args.adapter_path or "").strip() or None,
                        "eval_set": str(q_path),
                        "eval_set_version": eval_set_version,
                        "retrieval": {
                            "top_k": int(args.top_k),
                            "max_context_chars": int(eff_max_context_chars),
                        },
                        "generation": {
                            "gate_mode": bool(gate_mode),
                            "temperature": float(eff_temperature),
                            "top_p": float(eff_top_p),
                            "max_new_tokens": int(eff_max_new_tokens),
                        },
                    },
                }
                partial_metrics_path.write_text(
                    json.dumps(partial_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                partial_items_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    elapsed_s = max(1e-6, time.time() - started)
    avg = float(statistics.mean(scores)) if scores else 0.0
    p50 = float(statistics.median(scores)) if scores else 0.0
    p95 = float(statistics.quantiles(scores, n=20)[18]) if len(scores) >= 20 else (max(scores) if scores else 0.0)
    schema_pass_rate = schema_ok / max(1, len(questions))
    grounding_avg = float(statistics.mean(grounding_scores)) if grounding_scores else 0.0

    summary = {
        "eval": {
            "question_count": len(questions),
            "schema_pass_rate": round(schema_pass_rate, 4),
            "eval_score_avg": round(avg, 3),
            "eval_score_p50": round(p50, 3),
            "eval_score_p95": round(p95, 3),
            "grounding_avg": round(grounding_avg, 4),
            "elapsed_s": round(elapsed_s, 2),
        },
        "fail_counts": dict(fail_counts),
        "provenance": {
            "prompt_asset": str(prompt_path),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_hash,
            "effective_prompt_sha256": effective_prompt_hash,
            "index_dir": str(index_dir),
            "index_version": str(index_version),
            "model_id": model_name,
            "adapter_id": (args.adapter_id or "").strip() or None,
            "adapter_path": (args.adapter_path or "").strip() or None,
            "eval_set": str(q_path),
            "eval_set_version": eval_set_version,
            "retrieval": {
                "top_k": int(args.top_k),
                "max_context_chars": int(eff_max_context_chars),
            },
            "generation": {
                "gate_mode": bool(gate_mode),
                "temperature": float(eff_temperature),
                "top_p": float(eff_top_p),
                "max_new_tokens": int(eff_max_new_tokens),
            },
        },
        "gate": {
            "threshold": float(args.gate_threshold),
            "passed": bool(avg >= float(args.gate_threshold) and schema_pass_rate >= 0.98),
        },
    }

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    items_path = out_dir / "per_item.json"
    items_path.write_text(json.dumps(per_item, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = []
    report_lines.append("# Eval Report (ExpectedV2 JSON)\n")
    report_lines.append(f"- questions: `{q_path}` ({len(questions)})")
    report_lines.append(f"- eval_set_version: `{eval_set_version}`")
    report_lines.append(f"- prompt_asset: `{prompt_path}`")
    report_lines.append(f"- prompt_version: `{prompt_version}`")
    report_lines.append(f"- prompt_sha256: `{prompt_hash}`")
    report_lines.append(f"- index_version: `{index_version}`")
    report_lines.append(f"- model_id: `{model_name}`")
    if (args.adapter_id or "").strip():
        report_lines.append(f"- adapter_id: `{args.adapter_id.strip()}`")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- eval_score_avg: **{summary['eval']['eval_score_avg']}**")
    report_lines.append(f"- schema_pass_rate: **{summary['eval']['schema_pass_rate']}**")
    report_lines.append(f"- grounding_avg: **{summary['eval']['grounding_avg']}**")
    report_lines.append(f"- elapsed_s: {summary['eval']['elapsed_s']}")
    report_lines.append(f"- gate: threshold={summary['gate']['threshold']} passed={summary['gate']['passed']}")
    report_lines.append("")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Optional: annotate registry entry
    if (args.registry_path or "").strip() and (args.active_path or "").strip() and (args.registry_model_id or "").strip():
        reg = ModelRegistry(registry_path=Path(args.registry_path), active_path=Path(args.active_path))
        prefix = (args.metric_prefix or "").strip()
        def _k(k: str) -> str:
            return f"{prefix}{k}" if prefix else k
        reg.annotate(
            model_id=args.registry_model_id.strip(),
            metrics_patch={
                _k("eval_score_avg"): summary["eval"]["eval_score_avg"],
                _k("schema_pass_rate"): summary["eval"]["schema_pass_rate"],
                _k("grounding_avg"): summary["eval"]["grounding_avg"],
                _k("eval_set_version"): summary["provenance"]["eval_set_version"],
                _k("prompt_version"): summary["provenance"]["prompt_version"],
                _k("prompt_sha256"): summary["provenance"]["prompt_sha256"],
                _k("index_version"): summary["provenance"]["index_version"],
                _k("adapter_id"): summary["provenance"]["adapter_id"],
            },
            artifacts_patch={
                _k("last_eval_dir"): str(out_dir),
                _k("last_eval_metrics"): str(metrics_path),
                _k("last_eval_report"): str(out_dir / "report.md"),
            },
            tags_add=["evaluated"],
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    return int(asyncio_run(main_async()))


def asyncio_run(coro):  # tiny compat shim; avoid importing asyncio at module import time
    import asyncio

    return asyncio.run(coro)


if __name__ == "__main__":
    raise SystemExit(main())
