from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from how2everything.mine.io_utils import iter_jsonl


def aggregate_judgments(judgments_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Aggregate failure metrics overall and by topic.

    Examples where the judge output could not be parsed (``parse_failed=True``)
    are excluded from scoring but reported separately so users can investigate.
    """
    total = 0
    n_parse_failed = 0
    with_fail = 0
    sum_failures = 0
    by_topic = defaultdict(
        lambda: {"topic": "", "n_judged": 0, "n_with_failures": 0, "sum_failures": 0, "n_parse_failed": 0}
    )

    for rec in iter_jsonl(judgments_path):
        if not isinstance(rec, dict):
            continue
        total += 1
        topic = rec.get("topic", "")
        topic = topic if isinstance(topic, str) else str(topic)

        bt = by_topic[topic]
        bt["topic"] = topic

        # Exclude parse failures from scoring.
        if bool(rec.get("parse_failed", False)):
            n_parse_failed += 1
            bt["n_parse_failed"] += 1
            continue

        bt["n_judged"] += 1

        has_failure = bool(rec.get("has_failure", False))
        n_failures = rec.get("n_failures", 0)
        try:
            n_failures = int(n_failures)
        except Exception:
            n_failures = 0

        if has_failure:
            with_fail += 1
            bt["n_with_failures"] += 1
        sum_failures += n_failures
        bt["sum_failures"] += n_failures

    n_scored = total - n_parse_failed
    failure_rate = (with_fail / n_scored) if n_scored else 0.0
    how2score = (1.0 - failure_rate) if n_scored else 0.0
    summary: dict[str, Any] = {
        # Primary headline metric: % examples with zero critical failures.
        "how2score": how2score,
        "how2score_percent": 100.0 * how2score,
        # Counts
        "n_examples": n_scored,
        "n_with_failures": with_fail,
        # Diagnostics
        "failure_rate": failure_rate,
        "avg_failures_per_example": (sum_failures / n_scored) if n_scored else 0.0,
    }
    if n_parse_failed:
        summary["n_parse_failed"] = n_parse_failed

    rows: list[dict[str, Any]] = []
    for topic in sorted(by_topic.keys()):
        bt = by_topic[topic]
        n = bt["n_judged"]
        wf = bt["n_with_failures"]
        sf = bt["sum_failures"]
        fr = (wf / n) if n else 0.0
        hs = (1.0 - fr) if n else 0.0
        row: dict[str, Any] = {
            "topic": bt["topic"],
            "n_judged": n,
            "n_with_failures": wf,
            "how2score": hs,
            "how2score_percent": 100.0 * hs,
            "failure_rate": fr,
            "avg_failures_per_example": (sf / n) if n else 0.0,
        }
        if bt["n_parse_failed"]:
            row["n_parse_failed"] = bt["n_parse_failed"]
        rows.append(row)

    if n_parse_failed:
        print(
            f"  warning: {n_parse_failed}/{total} judgment(s) had parse failures "
            f"and were excluded from scoring",
            flush=True,
        )

    return summary, rows


def write_aggregates(*, out_root: Path, judgments_path: Path) -> dict[str, Any]:
    out_dir = out_root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, rows = aggregate_judgments(judgments_path)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = out_dir / "by_topic.csv"
    has_parse_failures = any("n_parse_failed" in r for r in rows)
    fieldnames = [
        "topic",
        "n_judged",
        "n_with_failures",
        "how2score",
        "how2score_percent",
        "failure_rate",
        "avg_failures_per_example",
    ]
    if has_parse_failures:
        fieldnames.append("n_parse_failed")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return summary

