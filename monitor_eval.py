"""Monitor eval progress by checking partial metrics file every N seconds."""
import json
import sys
import time
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path(r"D:\智能客服助手\graphrag_saas\graphrag_saas\backend\reports\evals\full1000_run1_repair_p10_tok512_final69")
PARTIAL_METRICS = REPORT_DIR / "metrics.partial.json"
PARTIAL_ITEMS = REPORT_DIR / "per_item.partial.json"
TOTAL_EXPECTED = 65
INTERVAL_S = 180  # 3 minutes


def check():
    if not PARTIAL_METRICS.exists():
        return None
    try:
        m = json.loads(PARTIAL_METRICS.read_text(encoding="utf-8"))
        ev = m.get("eval", {})
        return {
            "count": ev.get("question_count", 0),
            "schema_pass": ev.get("schema_pass_rate", 0),
            "score_avg": ev.get("eval_score_avg", 0),
            "grounding": ev.get("grounding_avg", 0),
            "elapsed_s": ev.get("elapsed_s", 0),
        }
    except Exception:
        return None


def main():
    print(f"[Monitor] Watching: {REPORT_DIR}")
    print(f"[Monitor] Expected: {TOTAL_EXPECTED} questions")
    print(f"[Monitor] Interval: {INTERVAL_S}s ({INTERVAL_S // 60} min)")
    print("=" * 60)

    while True:
        now = datetime.now().strftime("%H:%M:%S")
        info = check()
        if info is None:
            print(f"[{now}] Waiting for eval to start...")
        else:
            pct = round(info['count'] / TOTAL_EXPECTED * 100, 1)
            eta_s = 0
            if info['count'] > 0 and info['elapsed_s'] > 0:
                per_q = info['elapsed_s'] / info['count']
                remaining = TOTAL_EXPECTED - info['count']
                eta_s = per_q * remaining
            eta_min = round(eta_s / 60, 1)
            print(
                f"[{now}] {info['count']}/{TOTAL_EXPECTED} ({pct}%) | "
                f"score_avg={info['score_avg']:.1f} | "
                f"schema={info['schema_pass']:.2f} | "
                f"grounding={info['grounding']:.2f} | "
                f"elapsed={info['elapsed_s']:.0f}s | "
                f"ETA≈{eta_min}min"
            )
            if info['count'] >= TOTAL_EXPECTED:
                print(f"\n[{now}] ✅ All {TOTAL_EXPECTED} questions completed!")
                break

        time.sleep(INTERVAL_S)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
