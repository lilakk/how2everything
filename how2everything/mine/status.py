from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from how2everything.mine.io_utils import count_nonempty_lines, iter_jsonl, sanitize_topic_name

console = Console()

_STAGE_DIRS: list[tuple[str, str, str]] = [
    # (internal_key, directory_name, display_name)
    ("procedures", "01_procedures", "Extract"),
    ("prefilter", "02_prefilter", "Prefilter"),
    ("filter", "03_filter", "LLM Filter"),
    ("postprocess", "04_postprocess", "Postprocess"),
    ("resources", "05_resources", "Resources"),
    ("final_filter", "06_final_filter", "Final"),
]


def _stage_passed(rec: dict[str, Any], stage: str) -> bool:
    pe = rec.get("processed_example")
    if not isinstance(pe, dict):
        return False
    st = pe.get(stage, {})
    return bool(st.get("stage_passed")) if isinstance(st, dict) else False


def _count_passed(path: Path, stage: str) -> int:
    count = 0
    for rec in iter_jsonl(path):
        if _stage_passed(rec, stage):
            count += 1
    return count


def _topic_file(dir_path: Path, topic: str) -> Path:
    return dir_path / f"{sanitize_topic_name(topic)}.jsonl"


def _load_manifest(root: Path) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_status(out_root: str) -> None:
    root = Path(out_root)
    if not root.exists():
        raise FileNotFoundError(f"out_root not found: {root}")

    # --- Header: manifest info ---
    manifest = _load_manifest(root)
    if manifest:
        status = manifest.get("status", "unknown")
        ts = manifest.get("timestamp_unix")
        ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if ts else "?"
        cfg = manifest.get("config", {})
        targets = cfg.get("targets", {})
        model = cfg.get("llm", {}).get("model", "?")

        header_lines = [
            f"[bold]Status:[/bold] {status}",
            f"[bold]Last run:[/bold] {ts_str}",
            f"[bold]Model:[/bold] {model}",
            f"[bold]candidates_per_topic:[/bold] {targets.get('candidates_per_topic', '?')}",
        ]
        dv = targets.get("desired_valid_per_topic")
        if dv:
            header_lines.append(f"[bold]desired_valid_per_topic:[/bold] {dv}")
        console.print(Panel("\n".join(header_lines), title=f"[bold]{root}[/bold]", expand=False))
    else:
        console.print(Panel(f"[dim]No manifest.json found[/dim]", title=f"[bold]{root}[/bold]", expand=False))

    # --- Discover topics ---
    # Try to get topics from manifest config; fall back to scanning the procedures dir.
    topics: list[str] = []
    if manifest:
        topics = manifest.get("config", {}).get("topics", [])
    if not topics:
        proc_dir = root / "01_procedures"
        if proc_dir.is_dir():
            topics = sorted(
                p.stem.replace("_", " ") for p in proc_dir.glob("*.jsonl") if p.is_file()
            )
    if not topics:
        console.print("\n[dim]No topic data found.[/dim]")
        return

    # --- Per-stage pass counts ---
    # Gather data: stage_data[stage_key][topic] = (total, passed)
    stage_data: dict[str, dict[str, tuple[int, int]]] = {}
    active_stages: list[tuple[str, str, str]] = []
    for key, dirname, display in _STAGE_DIRS:
        d = root / dirname
        if not d.is_dir():
            continue
        topic_counts: dict[str, tuple[int, int]] = {}
        has_data = False
        for topic in topics:
            f = _topic_file(d, topic)
            total = count_nonempty_lines(f)
            passed = _count_passed(f, key) if total > 0 else 0
            topic_counts[topic] = (total, passed)
            if total > 0:
                has_data = True
        if has_data:
            stage_data[key] = topic_counts
            active_stages.append((key, dirname, display))

    if not active_stages:
        console.print("\n[dim]No stage data found.[/dim]")
        return

    # --- Build the main table ---
    table = Table(title="Pipeline Status by Topic", expand=False, show_edge=False, pad_edge=False)
    table.add_column("topic", style="bold")
    for key, _, display in active_stages:
        table.add_column(display, justify="right")
    # Add a yield column at the end (final_filter passed / candidates extracted).
    table.add_column("yield", justify="right", style="green")

    sum_extracted: int = 0
    sum_final_valid: int = 0

    for topic in topics:
        row: list[str] = [topic]
        extracted = 0
        final_valid = 0
        for key, _, _ in active_stages:
            total, passed = stage_data.get(key, {}).get(topic, (0, 0))
            if key == "procedures":
                extracted = total
            if key == "final_filter":
                final_valid = passed
            if total == 0:
                row.append("[dim]-[/dim]")
            else:
                row.append(f"{passed}/{total}")
        # Yield column.
        if extracted > 0:
            pct = 100.0 * final_valid / extracted
            row.append(f"{final_valid}/{extracted} ({pct:.0f}%)")
        else:
            row.append("[dim]-[/dim]")
        sum_extracted += extracted
        sum_final_valid += final_valid
        table.add_row(*row)

    # Totals row.
    totals_row: list[str] = ["[bold]TOTAL[/bold]"]
    for key, _, _ in active_stages:
        total_sum = sum(stage_data.get(key, {}).get(t, (0, 0))[0] for t in topics)
        passed_sum = sum(stage_data.get(key, {}).get(t, (0, 0))[1] for t in topics)
        if total_sum == 0:
            totals_row.append("[dim]-[/dim]")
        else:
            totals_row.append(f"[bold]{passed_sum}/{total_sum}[/bold]")
    if sum_extracted > 0:
        pct = 100.0 * sum_final_valid / sum_extracted
        totals_row.append(f"[bold green]{sum_final_valid}/{sum_extracted} ({pct:.0f}%)[/bold green]")
    else:
        totals_row.append("[dim]-[/dim]")
    table.add_row(*totals_row, end_section=True)

    console.print()
    console.print(table)

    # --- Export stats ---
    data_dir = root / "data"
    if data_dir.is_dir():
        export_files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.stat().st_size > 0)
        if export_files:
            console.print()
            console.print("[bold]Exported files:[/bold]")
            for p in export_files:
                size_mb = p.stat().st_size / (1024 * 1024)
                if p.suffix == ".jsonl" or p.name.endswith(".jsonl.zst"):
                    lines = count_nonempty_lines(p)
                    console.print(f"  {p.name}  [dim]({lines} records, {size_mb:.1f} MB)[/dim]")
                else:
                    console.print(f"  {p.name}  [dim]({size_mb:.1f} MB)[/dim]")

    # --- Legend ---
    console.print()
    console.print("[dim]Values shown as passed/total per stage. Yield = final valid / extracted.[/dim]")
