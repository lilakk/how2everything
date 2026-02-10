from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from how2everything.bench.aggregate import aggregate_judgments
from how2everything.mine.io_utils import iter_jsonl


@dataclass(frozen=True)
class LeaderboardRow:
    gen_run: str
    generator_model: str
    judge_dir: str
    judge_id: str
    judge_model: str
    how2score: float
    n_examples: int
    n_parse_failed: int = 0


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _first_generator_model(generations_path: Path) -> str:
    """
    Best-effort extraction of generator model name from generations.jsonl.
    """
    for rec in iter_jsonl(generations_path):
        if not isinstance(rec, dict):
            continue
        gen = rec.get("generator")
        if isinstance(gen, dict):
            m = gen.get("model")
            if isinstance(m, str) and m.strip():
                return m.strip()
        break
    return ""


def _matches_judge_dir(dir_name: str, judge: str | None) -> bool:
    if not judge:
        return True
    j = judge.strip()
    if not j:
        return True
    if dir_name == j:
        return True
    # Common: want to filter by just the hash portion.
    if dir_name.endswith(f"_{j}"):
        return True
    # Fallback: substring match.
    return j in dir_name


def iter_leaderboard_rows(
    *,
    generations_root: Path,
    judge: str | None = None,
) -> Iterable[LeaderboardRow]:
    """
    Iterate a leaderboard over many generation runs for a (possibly filtered) judge.

    Expects a directory structure like:
      generations_root/<gen_run>/generations.jsonl
      generations_root/<gen_run>/judgments/<model>_<judge_id>/aggregate/summary.json
    """
    if not generations_root.exists() or not generations_root.is_dir():
        raise FileNotFoundError(f"generations_root not found or not a directory: {generations_root}")

    for gen_run_dir in sorted([p for p in generations_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        gen_run = gen_run_dir.name
        generations_path = gen_run_dir / "generations.jsonl"
        judgments_root = gen_run_dir / "judgments"

        generator_model = _first_generator_model(generations_path) if generations_path.exists() else ""

        if not judgments_root.exists() or not judgments_root.is_dir():
            continue

        judge_dirs = [p for p in judgments_root.iterdir() if p.is_dir() and _matches_judge_dir(p.name, judge)]
        for jdir in sorted(judge_dirs, key=lambda p: p.name):
            summary_path = jdir / "aggregate" / "summary.json"
            judgments_path = jdir / "judgments.jsonl"
            manifest_path = jdir / "judge_manifest.json"

            summary = _read_json(summary_path) if summary_path.exists() else None
            if summary is None and judgments_path.exists():
                summary, _rows = aggregate_judgments(judgments_path)
            if summary is None:
                continue

            judge_id = ""
            judge_model = ""
            manifest = _read_json(manifest_path) if manifest_path.exists() else None
            if isinstance(manifest, dict):
                jid = manifest.get("judge_id")
                if isinstance(jid, str):
                    judge_id = jid
                j = manifest.get("judge")
                if isinstance(j, dict):
                    jm = j.get("model")
                    if isinstance(jm, str):
                        judge_model = jm

            # Output how2score on a 0-100 scale (percent).
            how2score = 0.0
            try:
                if summary.get("how2score_percent", None) is not None:
                    how2score = float(summary.get("how2score_percent", 0.0))
                else:
                    how2score = 100.0 * float(summary.get("how2score", 0.0))
            except Exception:
                how2score = 0.0
            try:
                n_examples = int(summary.get("n_examples", 0))
            except Exception:
                n_examples = 0
            try:
                n_parse_failed = int(summary.get("n_parse_failed", 0))
            except Exception:
                n_parse_failed = 0

            yield LeaderboardRow(
                gen_run=gen_run,
                generator_model=generator_model,
                judge_dir=jdir.name,
                judge_id=judge_id,
                judge_model=judge_model,
                how2score=how2score,
                n_examples=n_examples,
                n_parse_failed=n_parse_failed,
            )


def _clamp_score(v: float) -> float:
    return max(0.0, min(100.0, v))


def write_leaderboard_csv(
    *,
    generations_root: Path,
    judge: str | None = None,
    out_csv: Path | None = None,
) -> None:
    rows = list(iter_leaderboard_rows(generations_root=generations_root, judge=judge))
    has_parse_failures = any(r.n_parse_failed > 0 for r in rows)
    fieldnames = [
        "gen_run",
        "generator_model",
        "judge_dir",
        "judge_model",
        "how2score",
        "n_examples",
    ]
    if has_parse_failures:
        fieldnames.append("n_parse_failed")
    out_f = out_csv.open("w", encoding="utf-8", newline="") if out_csv is not None else None
    try:
        w = csv.DictWriter(out_f if out_f is not None else __import__("sys").stdout, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row: dict[str, Any] = {
                "gen_run": r.gen_run,
                "generator_model": r.generator_model,
                "judge_dir": r.judge_dir,
                "judge_model": r.judge_model,
                "how2score": f"{_clamp_score(r.how2score):.1f}",
                "n_examples": r.n_examples,
            }
            if has_parse_failures:
                row["n_parse_failed"] = r.n_parse_failed
            w.writerow(row)
    finally:
        if out_f is not None:
            out_f.close()


def print_leaderboard_table(
    *,
    generations_root: Path,
    judge: str | None = None,
) -> None:
    """
    Pretty-print a terminal table for quick inspection.

    Uses the same rows/columns as the CSV output.
    """
    rows = list(iter_leaderboard_rows(generations_root=generations_root, judge=judge))
    has_parse_failures = any(r.n_parse_failed > 0 for r in rows)
    cols = ["gen_run", "generator_model", "judge_dir", "judge_model", "how2score", "n_examples"]
    if has_parse_failures:
        cols.append("n_parse_failed")
    numeric_cols = {"how2score", "n_examples", "n_parse_failed"}

    table_rows: list[dict[str, str]] = []
    for r in rows:
        tr: dict[str, str] = {
            "gen_run": r.gen_run,
            "generator_model": r.generator_model,
            "judge_dir": r.judge_dir,
            "judge_model": r.judge_model,
            "how2score": f"{_clamp_score(r.how2score):.1f}",
            "n_examples": str(r.n_examples),
        }
        if has_parse_failures:
            tr["n_parse_failed"] = str(r.n_parse_failed)
        table_rows.append(tr)

    # Compute widths (include header).
    widths: dict[str, int] = {c: len(c) for c in cols}
    for tr in table_rows:
        for c in cols:
            widths[c] = max(widths[c], len(tr.get(c, "")))

    def _border(left: str, mid: str, right: str, fill: str = "─") -> str:
        segs = [fill * (widths[c] + 2) for c in cols]
        return left + mid.join(segs) + right

    def _fmt(tr: dict[str, str]) -> str:
        parts: list[str] = []
        for c in cols:
            s = tr.get(c, "")
            align = ">" if c in numeric_cols else "<"
            parts.append(f"{s:{align}{widths[c]}}")
        return "│ " + " │ ".join(parts) + " │"

    print(_border("┌", "┬", "┐"))
    print(_fmt({c: c for c in cols}))
    print(_border("├", "┼", "┤"))
    for tr in table_rows:
        print(_fmt(tr))
    print(_border("└", "┴", "┘"))

