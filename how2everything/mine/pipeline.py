from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Type

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from how2everything.mine.config import How2MineConfig
from how2everything.mine.document_sources import iter_documents, iter_documents_many, iter_input_paths
from how2everything.mine.io_utils import (
    count_nonempty_lines,
    iter_jsonl,
    load_prompt,
    processed_ids,
    sanitize_topic_name,
    write_jsonl,
)
from how2everything.mine.prefilter import apply_prefilter
from how2everything.mine.schemas import (
    FilterJudgment,
    FinalFilterResult,
    PostprocessResult,
    ProcessExtraction,
    ToolsResult,
)
from how2everything.mine.export_utils import export_all_valid
from how2everything.llm.deluge_client import DelugeClient, DelugeLLMConfig
from how2everything.llm.vllm_client import VLLMClient, VLLMConfig

console = Console()

SCHEMA_VERSION = "how2mine.v1"

_BANNER = r"""
    ⛏  [bold cyan]How2Mine[/bold cyan]  ⛏
   [dim]Mining procedures from the web[/dim]
"""

_STAGE_ORDER: list[str] = [
    "procedures", "prefilter", "filter", "postprocess", "resources", "final_filter", "export",
]

_STAGE_EMOJI: dict[str, str] = {
    "procedures": "🧩",
    "prefilter": "🧹",
    "filter": "🚫",
    "postprocess": "✍️",
    "resources": "🧰",
    "final_filter": "✅",
    "export": "📦",
}

# Display names (decoupled from internal keys for backward compat).
_STAGE_DISPLAY: dict[str, str] = {
    "procedures": "Extract Procedures",
    "prefilter": "Heuristics Filter",
    "filter": "LLM Filter",
    "postprocess": "Postprocess",
    "resources": "Extract Resources",
    "final_filter": "Final Validation",
    "export": "Export",
}

_STAGE_STYLE: dict[str, str] = {
    "procedures": "bold cyan",
    "prefilter": "bold yellow",
    "filter": "bold red",
    "postprocess": "bold magenta",
    "resources": "bold green",
    "final_filter": "bold blue",
    "export": "bold white",
}

_stage_timers: dict[str, float] = {}


def _print_stage_banner(name: str) -> None:
    icon = _STAGE_EMOJI.get(name, "•")
    title = _STAGE_DISPLAY.get(name, name.replace("_", " ").strip().title())
    style = _STAGE_STYLE.get(name, "bold")
    idx = _STAGE_ORDER.index(name) + 1 if name in _STAGE_ORDER else 0
    total = len(_STAGE_ORDER)
    console.print()
    console.print(
        Rule(f"{icon}  {title}  [dim]\\[{idx}/{total}][/dim]", style=style, align="left")
    )
    _stage_timers[name] = time.time()


def _print_stage_elapsed(name: str) -> None:
    start = _stage_timers.pop(name, None)
    if start is not None:
        elapsed = time.time() - start
        if elapsed >= 60:
            m, s = divmod(elapsed, 60)
            console.print(f"  [dim]⏱  {int(m)}m {s:.1f}s[/dim]")
        else:
            console.print(f"  [dim]⏱  {elapsed:.1f}s[/dim]")


def _print_usage(usage: dict[str, float | int]) -> None:
    if usage.get("cost_usd") or usage.get("input_tokens") or usage.get("output_tokens"):
        cost = float(usage.get("cost_usd", 0.0))
        it = int(usage.get("input_tokens", 0))
        ot = int(usage.get("output_tokens", 0))
        console.print(
            f"  [green]💰 ${cost:.4f}[/green]   [dim]🔡 {it:,} tokens in / {ot:,} tokens out[/dim]"
        )


def _print_pipeline_legend() -> None:
    legend = Text.from_markup(
        "[dim]Table columns:[/dim] existing = resumed from prior run, "
        "new = this run, total = existing + new.\n"
        "[dim]Values shown as[/dim] processed/passed."
    )
    console.print()
    console.print(Panel(_BANNER.strip(), border_style="cyan", expand=False))
    console.print(legend)


def _stage_passed(rec: dict[str, Any], stage: str) -> bool:
    pe = _get_processed_example(rec)
    st = pe.get(stage, {}) if isinstance(pe, dict) else {}
    return bool(st.get("stage_passed")) if isinstance(st, dict) else False


def _count_existing(out_path: Path, stage: str, eligible_fn) -> tuple[int, int]:
    """Count already-existing processed/passed items in a stage output file."""
    proc = 0
    passed = 0
    for rec in iter_jsonl(out_path):
        if not eligible_fn(rec):
            continue
        proc += 1
        if _stage_passed(rec, stage):
            passed += 1
    return proc, passed


def _count_stage_passed(out_path: Path, stage: str) -> int:
    c = 0
    for rec in iter_jsonl(out_path):
        if _stage_passed(rec, stage):
            c += 1
    return c


class JSONClient(Protocol):
    last_usage: dict[str, float | int]
    total_usage: dict[str, float | int]

    def complete_json(
        self,
        prompts: Sequence[str],
        *,
        output_schema: Type[BaseModel] | dict[str, Any] | None = None,
        max_parse_retries: int = 2,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> list[dict[str, Any]]: ...


def _sum_usage(clients: list[JSONClient]) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "cost_usd": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    for c in clients:
        u = getattr(c, "total_usage", None)
        if not isinstance(u, dict):
            continue
        out["cost_usd"] = float(out["cost_usd"]) + float(u.get("cost_usd", 0.0))
        out["input_tokens"] = int(out["input_tokens"]) + int(u.get("input_tokens", 0))
        out["output_tokens"] = int(out["output_tokens"]) + int(u.get("output_tokens", 0))
        out["cache_read_tokens"] = int(out["cache_read_tokens"]) + int(u.get("cache_read_tokens", 0))
        out["cache_write_tokens"] = int(out["cache_write_tokens"]) + int(u.get("cache_write_tokens", 0))
    return out


def _print_stage_table(
    *,
    rows: list[dict[str, object]],
    totals: dict[str, object] | None,
    note: str | None = None,
) -> None:
    if note:
        console.print(f"  {note}")
    if not rows:
        if totals is not None:
            console.print(f"  [dim]totals: {totals}[/dim]")
        return

    # Stable column order.
    cols: list[str] = ["topic"]
    for k in rows[0].keys():
        if k != "topic":
            cols.append(k)

    # Hide columns that are all zeros (e.g. "existing" on a fresh run).
    def _is_zero_col(col: str) -> bool:
        all_rows = rows + ([{"topic": "TOTAL", **totals}] if totals else [])
        for r in all_rows:
            v = str(r.get(col, ""))
            stripped = v.replace("/", "").replace(" ", "").replace("0", "").replace(".", "").replace("%", "")
            if stripped:
                return False
        return True

    visible_cols = [c for c in cols if c == "topic" or not _is_zero_col(c)]

    from rich.box import SIMPLE_HEAVY

    table = Table(box=SIMPLE_HEAVY, show_edge=False, pad_edge=True, padding=(0, 1))
    for c in visible_cols:
        justify = "left" if c == "topic" else "right"
        table.add_column(c, justify=justify, style="bold" if c == "topic" else "")

    for r in rows:
        values = [str(r.get(c, "")) for c in visible_cols]
        table.add_row(*values)

    if totals is not None:
        table.add_section()
        totals_vals: list[str] = []
        for c in visible_cols:
            if c == "topic":
                totals_vals.append("[bold cyan]TOTAL[/bold cyan]")
            elif c in totals:
                totals_vals.append(f"[bold]{totals[c]}[/bold]")
            else:
                totals_vals.append("")
        table.add_row(*totals_vals)

    console.print(table)


def _safe_git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def write_manifest(cfg: How2MineConfig, *, out_path: Path, status: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp_unix": time.time(),
        "git_sha": _safe_git_sha(),
        "config": _jsonable(asdict(cfg)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _steps_text(steps: List[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))


def _resources_text(resources: List[str]) -> str:
    return "\n".join(f"- {r}" for r in resources)


def _final_filter_all_yes(final_filter_stage: Any) -> bool:
    """True if every rubric answer is 'yes'."""
    if not isinstance(final_filter_stage, dict):
        return False
    # Accept either the full dict with rubric keys, or a wrapped dict containing them.
    ff = final_filter_stage
    if "final_filter" in final_filter_stage and isinstance(final_filter_stage["final_filter"], dict):
        ff = final_filter_stage["final_filter"]
    answers: list[str] = []
    for v in ff.values():
        if isinstance(v, dict) and isinstance(v.get("answer"), str):
            answers.append(v["answer"])
    return bool(answers) and all(a == "yes" for a in answers)


def _ensure_base_record(*, source_example: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_example": source_example,
        "processed_example": {},
    }


def _get_processed_example(rec: dict[str, Any]) -> dict[str, Any]:
    pe = rec.get("processed_example")
    if isinstance(pe, dict):
        return pe
    return {}


def _set_stage(
    rec: dict[str, Any], stage: str, stage_obj: dict[str, Any]
) -> dict[str, Any]:
    out = dict(rec)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("source_example", rec.get("source_example", {}))
    pe = _get_processed_example(out)
    # Ensure we store it under processed_example
    out["processed_example"] = dict(pe)
    out["processed_example"][stage] = stage_obj
    # Do not emit the legacy `response` field in new outputs.
    if "response" in out:
        out.pop("response", None)
    return out


def stage_paths(out_root: Path) -> dict[str, Path]:
    return {
        "procedures": out_root / "01_procedures",
        "prefilter": out_root / "02_prefilter",
        "filter": out_root / "03_filter",
        "postprocess": out_root / "04_postprocess",
        "resources": out_root / "05_resources",
        "final_filter": out_root / "06_final_filter",
        "data": out_root / "data",
    }


def _topic_file(dir_path: Path, topic: str) -> Path:
    return dir_path / f"{sanitize_topic_name(topic)}.jsonl"


def _iter_input_documents(cfg: How2MineConfig) -> Iterable[dict[str, Any]]:
    if cfg.inputs.kind not in {"jsonl_documents", "documents"}:
        raise ValueError(f"Unsupported inputs.kind for public runner: {cfg.inputs.kind}")
    p = cfg.inputs.path
    if p.is_dir():
        paths = iter_input_paths(p, cfg.inputs.include_globs)
        yield from iter_documents_many(paths=paths, fmt=cfg.inputs.format, compression=cfg.inputs.compression)
        return
    yield from iter_documents(path=p, fmt=cfg.inputs.format, compression=cfg.inputs.compression)


def run_extract(
    cfg: How2MineConfig,
    client: JSONClient,
    *,
    max_new_per_topic: int | None = None,
    skip_topics: set[str] | None = None,
) -> dict[str, int]:
    # Stage is now called "procedures" (function name kept for compatibility).
    _print_stage_banner("procedures")
    paths = stage_paths(cfg.out_root)
    procedures_dir = paths["procedures"]
    template = load_prompt(cfg.prompts.procedures)

    # Stream input docs and only keep up to "need" per topic in memory.
    # This keeps public usage simple while supporting large input formats.
    _skip = skip_topics or set()
    per_topic_already: dict[str, set[str]] = {}
    per_topic_need: dict[str, int] = {}
    for topic in cfg.topics:
        out_path = _topic_file(procedures_dir, topic)
        existing_ids = processed_ids(out_path)
        per_topic_already[topic] = existing_ids
        if topic in _skip:
            per_topic_need[topic] = 0
            continue
        budget_remaining = max(cfg.targets.candidates_per_topic - count_nonempty_lines(out_path), 0)
        per_topic_need[topic] = (
            min(int(max_new_per_topic), budget_remaining)
            if max_new_per_topic is not None
            else budget_remaining
        )

    to_process_by_topic: Dict[str, List[dict[str, Any]]] = {t: [] for t in cfg.topics}
    remaining = sum(per_topic_need.values())
    if remaining > 0:
        for rec in _iter_input_documents(cfg):
            if cfg.use_topics:
                topic_val = rec.get(cfg.inputs.topic_field)
                if not isinstance(topic_val, str) or topic_val not in to_process_by_topic:
                    continue
                routed_topic = topic_val
                raw_topic = topic_val
            else:
                routed_topic = cfg.topics[0]  # pseudo-topic, e.g. "all"
                tv = rec.get(cfg.inputs.topic_field)
                raw_topic = str(tv) if tv is not None else ""

            if per_topic_need.get(routed_topic, 0) <= 0:
                continue
            doc_id = rec.get(cfg.inputs.id_field)
            if doc_id is None:
                continue
            doc_id_str = str(doc_id)
            if doc_id_str in per_topic_already[routed_topic]:
                continue
            # Attach the raw topic (if any) so we can preserve it in source_example in no-topics mode.
            if not cfg.use_topics:
                rec = dict(rec)
                rec["_h2e_raw_topic"] = raw_topic
            to_process_by_topic[routed_topic].append(rec)
            per_topic_already[routed_topic].add(doc_id_str)
            per_topic_need[routed_topic] -= 1
            remaining -= 1
            if remaining <= 0:
                break

    def _extract_eligible(rec: dict[str, Any]) -> bool:
        # If it's in the extract output at all, it was extracted.
        return True

    rows: list[dict[str, object]] = []
    totals = {"existing": "0/0", "new": "0/0", "total": "0/0", "pass%": "  0.0%", "data_gap": 0}
    usage_totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0, "gap": 0}
    for topic in cfg.topics:
        out_path = _topic_file(procedures_dir, topic)
        existing = count_nonempty_lines(out_path)
        budget_remaining = max(cfg.targets.candidates_per_topic - existing, 0)
        need = (
            min(int(max_new_per_topic), budget_remaining)
            if max_new_per_topic is not None
            else budget_remaining
        )
        if need <= 0:
            exist_proc, exist_pass = _count_existing(out_path, "procedures", _extract_eligible)
            rows.append(
                {
                    "topic": topic,
                    "existing": f"{exist_proc}/{exist_pass}",
                    "new": "0/0",
                    "total": f"{exist_proc}/{exist_pass}",
                    "pass%": f"{(100.0 * exist_pass / exist_proc) if exist_proc else 0.0:5.1f}%",
                    "data_gap": 0,
                }
            )
            sums["exist_proc"] += exist_proc
            sums["exist_pass"] += exist_pass
            continue

        to_process: List[dict[str, Any]] = to_process_by_topic.get(topic, [])

        prompts = [
            template.format(document=str(r.get(cfg.inputs.text_field, "")))
            for r in to_process
        ]
        if not prompts:
            # We didn't find enough input documents for this topic in this run.
            shortfall = max(need - len(to_process), 0)
            exist_proc, exist_pass = _count_existing(out_path, "procedures", _extract_eligible)
            rows.append(
                {
                    "topic": topic,
                    "existing": f"{exist_proc}/{exist_pass}",
                    "new": "0/0",
                    "total": f"{exist_proc}/{exist_pass}",
                    "pass%": f"{(100.0 * exist_pass / exist_proc) if exist_proc else 0.0:5.1f}%",
                    "data_gap": shortfall,
                }
            )
            sums["exist_proc"] += exist_proc
            sums["exist_pass"] += exist_pass
            sums["gap"] += shortfall
            continue
        results = client.complete_json(
            prompts,
            output_schema=ProcessExtraction,
            show_progress=True,
            progress_desc=f"procedures/{sanitize_topic_name(topic)}",
        )
        usage_totals["cost_usd"] += float(client.last_usage.get("cost_usd", 0.0))
        usage_totals["input_tokens"] += int(client.last_usage.get("input_tokens", 0))
        usage_totals["output_tokens"] += int(client.last_usage.get("output_tokens", 0))

        records: List[dict[str, Any]] = []
        passed = 0
        for r, parsed in zip(to_process, results):
            doc_id = str(r.get(cfg.inputs.id_field))
            raw_topic = (
                r.get("_h2e_raw_topic")
                if (not cfg.use_topics)
                else topic
            )
            src = {
                "id": doc_id,
                "text": r.get(cfg.inputs.text_field, ""),
                "url": r.get(cfg.inputs.url_field, ""),
                "topic": raw_topic if isinstance(raw_topic, str) else "",
            }
            base = _ensure_base_record(source_example=src)
            steps = parsed.get("steps") if isinstance(parsed, dict) else None
            stage_passed = bool(parsed.get("has_valid_process")) and isinstance(steps, list) and len(steps) > 0
            if stage_passed:
                passed += 1
            extract_stage = {
                "stage_passed": stage_passed,
                "has_valid_process": bool(parsed.get("has_valid_process")),
                "goal": parsed.get("goal", "") if isinstance(parsed.get("goal"), str) else "",
                "steps": steps if isinstance(steps, list) else [],
            }
            record = _set_stage(base, "procedures", extract_stage)
            records.append(record)

        if records:
            write_jsonl(out_path, records, mode="a")

        shortfall = max(need - len(to_process), 0)
        processed = len(prompts)  # new processed
        pass_pct = (100.0 * passed / processed) if processed else 0.0
        exist_proc, exist_pass = _count_existing(out_path, "procedures", _extract_eligible)
        # exist_* includes the newly written records; subtract to get pre-run existing.
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
                "data_gap": shortfall,
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed
        sums["gap"] += shortfall

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
        "data_gap": sums["gap"],
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_usage(usage_totals)
    _print_stage_elapsed("procedures")
    return {"new_processed": int(sums["new_proc"]), "data_gap": int(sums["gap"])}


def run_prefilter(cfg: How2MineConfig) -> None:
    _print_stage_banner("prefilter")
    paths = stage_paths(cfg.out_root)
    procedures_dir = paths["procedures"]
    prefilter_dir = paths["prefilter"]

    def _prefilter_eligible(rec: dict[str, Any]) -> bool:
        pe = _get_processed_example(rec)
        proc = pe.get("procedures", {}) if isinstance(pe, dict) else {}
        steps = proc.get("steps") if isinstance(proc, dict) else None
        return bool(proc.get("stage_passed")) and isinstance(steps, list)

    rows: list[dict[str, object]] = []
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0}
    for topic in cfg.topics:
        in_path = _topic_file(procedures_dir, topic)
        out_path = _topic_file(prefilter_dir, topic)
        already = processed_ids(out_path)

        # Progress total: all *new* records for this topic (regardless of eligibility).
        pending_total = 0
        for rec in iter_jsonl(in_path):
            src = rec.get("source_example", {})
            doc_id = str(src.get("id")) if isinstance(src, dict) else None
            if doc_id and doc_id not in already:
                pending_total += 1

        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn

        new_records: List[dict[str, Any]] = []
        processed = 0
        passed = 0
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        with progress:
            task = progress.add_task(
                f"prefilter/{sanitize_topic_name(topic)}", total=pending_total
            )
            for rec in iter_jsonl(in_path):
                src = rec.get("source_example", {})
                doc_id = str(src.get("id")) if isinstance(src, dict) else None
                if not doc_id or doc_id in already:
                    continue

                pe = _get_processed_example(rec)
                proc_stage = pe.get("procedures", {}) if isinstance(pe, dict) else {}
                steps = proc_stage.get("steps") if isinstance(proc_stage, dict) else None
                extract_passed = bool(proc_stage.get("stage_passed")) if isinstance(proc_stage, dict) else False

                if extract_passed and isinstance(steps, list):
                    processed += 1
                    ok, reason = apply_prefilter(steps, cfg.targets.min_steps, cfg.targets.max_steps)
                else:
                    ok, reason = False, "Not a valid extracted process"

                prefilter_stage = {
                    "stage_passed": ok,
                    "reason": reason,
                    "min_steps": cfg.targets.min_steps,
                    "max_steps": cfg.targets.max_steps,
                }
                if ok:
                    passed += 1
                out_rec = _set_stage(dict(rec), "prefilter", prefilter_stage)
                new_records.append(out_rec)
                progress.advance(task)

        if new_records:
            write_jsonl(out_path, new_records, mode="a")

        exist_proc, exist_pass = _count_existing(out_path, "prefilter", _prefilter_eligible)
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_stage_elapsed("prefilter")


def run_filter(cfg: How2MineConfig, client: JSONClient) -> None:
    _print_stage_banner("filter")
    paths = stage_paths(cfg.out_root)
    prefilter_dir = paths["prefilter"]
    filter_dir = paths["filter"]
    template = load_prompt(cfg.prompts.filter)

    def _filter_eligible(rec: dict[str, Any]) -> bool:
        pe = _get_processed_example(rec)
        ex = pe.get("procedures", {}) if isinstance(pe, dict) else {}
        pf = pe.get("prefilter", {}) if isinstance(pe, dict) else {}
        if not (isinstance(ex, dict) and isinstance(pf, dict)):
            return False
        if not ex.get("stage_passed", False):
            return False
        if not pf.get("stage_passed", False):
            return False
        goal = str(ex.get("goal", "")).strip()
        steps = ex.get("steps", [])
        return bool(goal) and isinstance(steps, list) and len(steps) > 0

    rows: list[dict[str, object]] = []
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0}
    usage_totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    for topic in cfg.topics:
        in_path = _topic_file(prefilter_dir, topic)
        out_path = _topic_file(filter_dir, topic)
        already = processed_ids(out_path)

        candidates: List[dict[str, Any]] = []
        prompts: List[str] = []
        for rec in iter_jsonl(in_path):
            src = rec.get("source_example", {})
            doc_id = str(src.get("id")) if isinstance(src, dict) else None
            if not doc_id or doc_id in already:
                continue

            pe = _get_processed_example(rec)
            extract_stage = pe.get("procedures", {}) if isinstance(pe, dict) else {}
            prefilter_stage = pe.get("prefilter", {}) if isinstance(pe, dict) else {}

            if not (isinstance(extract_stage, dict) and isinstance(prefilter_stage, dict)):
                candidates.append(rec)
                prompts.append("")
                continue

            if not extract_stage.get("stage_passed", False):
                candidates.append(rec)
                prompts.append("")
                continue
            if not prefilter_stage.get("stage_passed", False):
                candidates.append(rec)
                prompts.append("")
                continue

            goal = str(extract_stage.get("goal", "")).strip()
            steps = extract_stage.get("steps", [])
            if not goal or not isinstance(steps, list) or not steps:
                candidates.append(rec)
                prompts.append("")
                continue

            prompt = template.format(goal=goal, steps=_steps_text(steps))
            candidates.append(rec)
            prompts.append(prompt)

        # Only call LLM for non-empty prompts.
        to_call = [(i, p) for i, p in enumerate(prompts) if p]
        judgments: Dict[int, dict[str, Any]] = {}
        if to_call:
            idxs, call_prompts = zip(*to_call)
            results = client.complete_json(
                list(call_prompts),
                output_schema=FilterJudgment,
                show_progress=True,
                progress_desc=f"filter/{sanitize_topic_name(topic)}",
            )
            usage_totals["cost_usd"] += float(client.last_usage.get("cost_usd", 0.0))
            usage_totals["input_tokens"] += int(client.last_usage.get("input_tokens", 0))
            usage_totals["output_tokens"] += int(client.last_usage.get("output_tokens", 0))
            for i, parsed in zip(idxs, results):
                judgments[i] = parsed

        out_records: List[dict[str, Any]] = []
        processed = len(to_call)
        passed = 0
        for i, rec in enumerate(candidates):
            parsed = judgments.get(i)
            if isinstance(parsed, dict) and "judgment" in parsed:
                judgment = parsed.get("judgment")
                stage_passed = (judgment is False)
                filter_stage = {
                    "stage_passed": stage_passed,
                    "judgment": bool(judgment),
                    "reason": parsed.get("reason", "") if isinstance(parsed.get("reason"), str) else "",
                }
            else:
                filter_stage = {
                    "stage_passed": False,
                    "judgment": None,
                    "reason": "Missing filter judgment",
                }
            if i in judgments and filter_stage.get("stage_passed") is True:
                passed += 1
            out_records.append(_set_stage(dict(rec), "filter", filter_stage))

        if out_records:
            write_jsonl(out_path, out_records, mode="a")

        exist_proc, exist_pass = _count_existing(out_path, "filter", _filter_eligible)
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_usage(usage_totals)
    _print_stage_elapsed("filter")


def run_postprocess(cfg: How2MineConfig, client: JSONClient) -> None:
    _print_stage_banner("postprocess")
    paths = stage_paths(cfg.out_root)
    filter_dir = paths["filter"]
    post_dir = paths["postprocess"]
    template = load_prompt(cfg.prompts.postprocess)

    def _postprocess_eligible(rec: dict[str, Any]) -> bool:
        pe = _get_processed_example(rec)
        ex = pe.get("procedures", {}) if isinstance(pe, dict) else {}
        flt = pe.get("filter", {}) if isinstance(pe, dict) else {}
        if not (isinstance(ex, dict) and isinstance(flt, dict)):
            return False
        if not flt.get("stage_passed", False):
            return False
        goal = str(ex.get("goal", "")).strip()
        steps = ex.get("steps", [])
        return bool(goal) and isinstance(steps, list) and len(steps) > 0

    rows: list[dict[str, object]] = []
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0}
    usage_totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    for topic in cfg.topics:
        in_path = _topic_file(filter_dir, topic)
        out_path = _topic_file(post_dir, topic)
        already = processed_ids(out_path)

        candidates: List[dict[str, Any]] = []
        prompts: List[str] = []
        for rec in iter_jsonl(in_path):
            src = rec.get("source_example", {})
            doc_id = str(src.get("id")) if isinstance(src, dict) else None
            if not doc_id or doc_id in already:
                continue

            pe = _get_processed_example(rec)
            extract_stage = pe.get("procedures", {}) if isinstance(pe, dict) else {}
            filter_stage = pe.get("filter", {}) if isinstance(pe, dict) else {}

            if not (isinstance(extract_stage, dict) and isinstance(filter_stage, dict)):
                candidates.append(rec)
                prompts.append("")
                continue

            if not filter_stage.get("stage_passed", False):
                candidates.append(rec)
                prompts.append("")
                continue

            goal = str(extract_stage.get("goal", "")).strip()
            steps = extract_stage.get("steps", [])
            if not goal or not isinstance(steps, list) or not steps:
                candidates.append(rec)
                prompts.append("")
                continue
            prompt = template.format(goal=goal, steps=_steps_text(steps))
            candidates.append(rec)
            prompts.append(prompt)

        to_call = [(i, p) for i, p in enumerate(prompts) if p]
        rewrites: Dict[int, dict[str, Any]] = {}
        if to_call:
            idxs, call_prompts = zip(*to_call)
            results = client.complete_json(
                list(call_prompts),
                output_schema=PostprocessResult,
                show_progress=True,
                progress_desc=f"postprocess/{sanitize_topic_name(topic)}",
            )
            usage_totals["cost_usd"] += float(client.last_usage.get("cost_usd", 0.0))
            usage_totals["input_tokens"] += int(client.last_usage.get("input_tokens", 0))
            usage_totals["output_tokens"] += int(client.last_usage.get("output_tokens", 0))
            for i, parsed in zip(idxs, results):
                rewrites[i] = parsed

        out_records: List[dict[str, Any]] = []
        processed = len(to_call)
        passed = 0
        for i, rec in enumerate(candidates):
            parsed = rewrites.get(i)
            if isinstance(parsed, dict):
                rewritten_goal = parsed.get("rewritten_goal")
                rewritten_steps = parsed.get("rewritten_steps", [])
                stage_passed = isinstance(rewritten_goal, str) and isinstance(rewritten_steps, list) and len(rewritten_steps) > 0
                post_stage = {
                    "stage_passed": stage_passed,
                    "rewritten_goal": rewritten_goal if isinstance(rewritten_goal, str) else "",
                    "rewritten_steps": rewritten_steps if isinstance(rewritten_steps, list) else [],
                }
            else:
                post_stage = {
                    "stage_passed": False,
                    "rewritten_goal": "",
                    "rewritten_steps": [],
                }
            if i in rewrites and post_stage.get("stage_passed") is True:
                passed += 1
            out_records.append(_set_stage(dict(rec), "postprocess", post_stage))

        if out_records:
            write_jsonl(out_path, out_records, mode="a")

        exist_proc, exist_pass = _count_existing(out_path, "postprocess", _postprocess_eligible)
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_usage(usage_totals)
    _print_stage_elapsed("postprocess")


def run_tools(cfg: How2MineConfig, client: JSONClient) -> None:
    # Stage is now called "resources" (function name kept for compatibility).
    _print_stage_banner("resources")
    paths = stage_paths(cfg.out_root)
    post_dir = paths["postprocess"]
    resources_dir = paths["resources"]
    template = load_prompt(cfg.prompts.resources)

    def _resources_eligible(rec: dict[str, Any]) -> bool:
        pe = _get_processed_example(rec)
        post = pe.get("postprocess", {}) if isinstance(pe, dict) else {}
        if not isinstance(post, dict) or not post.get("stage_passed", False):
            return False
        goal = str(post.get("rewritten_goal", "")).strip()
        steps = post.get("rewritten_steps", [])
        return bool(goal) and isinstance(steps, list) and len(steps) > 0

    rows: list[dict[str, object]] = []
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0}
    usage_totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    for topic in cfg.topics:
        in_path = _topic_file(post_dir, topic)
        out_path = _topic_file(resources_dir, topic)
        already = processed_ids(out_path)

        candidates: List[dict[str, Any]] = []
        prompts: List[str] = []
        for rec in iter_jsonl(in_path):
            src = rec.get("source_example", {})
            doc_id = str(src.get("id")) if isinstance(src, dict) else None
            if not doc_id or doc_id in already:
                continue

            pe = _get_processed_example(rec)
            post_stage = pe.get("postprocess", {}) if isinstance(pe, dict) else {}
            if not isinstance(post_stage, dict) or not post_stage.get("stage_passed", False):
                candidates.append(rec)
                prompts.append("")
                continue

            goal = str(post_stage.get("rewritten_goal", "")).strip()
            steps = post_stage.get("rewritten_steps", [])
            if not goal or not isinstance(steps, list) or not steps:
                candidates.append(rec)
                prompts.append("")
                continue
            prompt = template.format(goal=goal, steps=_steps_text(steps))
            candidates.append(rec)
            prompts.append(prompt)

        to_call = [(i, p) for i, p in enumerate(prompts) if p]
        parsed_tools: Dict[int, dict[str, Any]] = {}
        if to_call:
            idxs, call_prompts = zip(*to_call)
            results = client.complete_json(
                list(call_prompts),
                output_schema=ToolsResult,
                show_progress=True,
                progress_desc=f"resources/{sanitize_topic_name(topic)}",
            )
            usage_totals["cost_usd"] += float(client.last_usage.get("cost_usd", 0.0))
            usage_totals["input_tokens"] += int(client.last_usage.get("input_tokens", 0))
            usage_totals["output_tokens"] += int(client.last_usage.get("output_tokens", 0))
            for i, parsed in zip(idxs, results):
                parsed_tools[i] = parsed

        out_records: List[dict[str, Any]] = []
        processed = len(to_call)
        passed = 0
        for i, rec in enumerate(candidates):
            parsed = parsed_tools.get(i)
            if isinstance(parsed, dict):
                resources = parsed.get("resources", [])
                stage_passed = isinstance(resources, list)
                tools_stage = {
                    "stage_passed": stage_passed,
                    "resources": resources if isinstance(resources, list) else [],
                }
            else:
                tools_stage = {"stage_passed": False, "resources": []}
            if i in parsed_tools and tools_stage.get("stage_passed") is True:
                passed += 1
            out_records.append(_set_stage(dict(rec), "resources", tools_stage))

        if out_records:
            write_jsonl(out_path, out_records, mode="a")

        exist_proc, exist_pass = _count_existing(out_path, "resources", _resources_eligible)
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_usage(usage_totals)
    _print_stage_elapsed("resources")


def run_final_filter(cfg: How2MineConfig, client: JSONClient) -> None:
    _print_stage_banner("final_filter")
    paths = stage_paths(cfg.out_root)
    resources_dir = paths["resources"]
    ff_dir = paths["final_filter"]
    template = load_prompt(cfg.prompts.final_filter)

    def _final_filter_eligible(rec: dict[str, Any]) -> bool:
        pe = _get_processed_example(rec)
        post = pe.get("postprocess", {}) if isinstance(pe, dict) else {}
        if not isinstance(post, dict) or not post.get("stage_passed", False):
            return False
        goal = str(post.get("rewritten_goal", "")).strip()
        steps = post.get("rewritten_steps", [])
        return bool(goal) and isinstance(steps, list) and len(steps) > 0

    rows: list[dict[str, object]] = []
    sums = {"exist_proc": 0, "exist_pass": 0, "new_proc": 0, "new_pass": 0}
    usage_totals = {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0}
    for topic in cfg.topics:
        in_path = _topic_file(resources_dir, topic)
        out_path = _topic_file(ff_dir, topic)
        already = processed_ids(out_path)

        candidates: List[dict[str, Any]] = []
        prompts: List[str] = []
        for rec in iter_jsonl(in_path):
            src = rec.get("source_example", {})
            doc_id = str(src.get("id")) if isinstance(src, dict) else None
            if not doc_id or doc_id in already:
                continue

            pe = _get_processed_example(rec)
            post_stage = pe.get("postprocess", {}) if isinstance(pe, dict) else {}
            resources_stage = pe.get("resources", {}) if isinstance(pe, dict) else {}
            if not (isinstance(post_stage, dict) and isinstance(resources_stage, dict)):
                candidates.append(rec)
                prompts.append("")
                continue
            if not post_stage.get("stage_passed", False):
                candidates.append(rec)
                prompts.append("")
                continue

            goal = str(post_stage.get("rewritten_goal", "")).strip()
            steps = post_stage.get("rewritten_steps", [])
            resources = resources_stage.get("resources", [])
            if not goal or not isinstance(steps, list) or not steps:
                candidates.append(rec)
                prompts.append("")
                continue
            prompt = template.format(
                goal=goal,
                steps=_steps_text(steps),
                resources=_resources_text(resources if isinstance(resources, list) else []),
            )
            candidates.append(rec)
            prompts.append(prompt)

        to_call = [(i, p) for i, p in enumerate(prompts) if p]
        parsed_ff: Dict[int, dict[str, Any]] = {}
        if to_call:
            idxs, call_prompts = zip(*to_call)
            results = client.complete_json(
                list(call_prompts),
                output_schema=FinalFilterResult,
                show_progress=True,
                progress_desc=f"final_filter/{sanitize_topic_name(topic)}",
            )
            usage_totals["cost_usd"] += float(client.last_usage.get("cost_usd", 0.0))
            usage_totals["input_tokens"] += int(client.last_usage.get("input_tokens", 0))
            usage_totals["output_tokens"] += int(client.last_usage.get("output_tokens", 0))
            for i, parsed in zip(idxs, results):
                parsed_ff[i] = parsed

        out_records: List[dict[str, Any]] = []
        processed = len(to_call)
        passed = 0
        for i, rec in enumerate(candidates):
            parsed = parsed_ff.get(i)
            if isinstance(parsed, dict):
                stage_passed = _final_filter_all_yes(parsed)
                ff_stage = {
                    "stage_passed": stage_passed,
                    **parsed,
                }
            else:
                ff_stage = {"stage_passed": False}
            if i in parsed_ff and ff_stage.get("stage_passed") is True:
                passed += 1
            out_records.append(_set_stage(dict(rec), "final_filter", ff_stage))

        if out_records:
            write_jsonl(out_path, out_records, mode="a")

        exist_proc, exist_pass = _count_existing(out_path, "final_filter", _final_filter_eligible)
        pre_exist_proc = max(exist_proc - processed, 0)
        pre_exist_pass = max(exist_pass - passed, 0)
        total_proc = pre_exist_proc + processed
        total_pass = pre_exist_pass + passed
        rows.append(
            {
                "topic": topic,
                "existing": f"{pre_exist_proc}/{pre_exist_pass}",
                "new": f"{processed}/{passed}",
                "total": f"{total_proc}/{total_pass}",
                "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
            }
        )
        sums["exist_proc"] += pre_exist_proc
        sums["exist_pass"] += pre_exist_pass
        sums["new_proc"] += processed
        sums["new_pass"] += passed

    total_proc = sums["exist_proc"] + sums["new_proc"]
    total_pass = sums["exist_pass"] + sums["new_pass"]
    totals = {
        "existing": f"{sums['exist_proc']}/{sums['exist_pass']}",
        "new": f"{sums['new_proc']}/{sums['new_pass']}",
        "total": f"{total_proc}/{total_pass}",
        "pass%": f"{(100.0 * total_pass / total_proc) if total_proc else 0.0:5.1f}%",
    }
    _print_stage_table(rows=rows, totals=totals)
    _print_usage(usage_totals)
    _print_stage_elapsed("final_filter")


def run_splits(cfg: How2MineConfig) -> None:
    # Keep the function name for backward compatibility, but the public mining
    # pipeline no longer creates train/test splits. That belongs in a separate
    # training-data packaging step (future: `h2e data ...`).
    _print_stage_banner("export")
    paths = stage_paths(cfg.out_root)
    ff_dir = paths["final_filter"]
    data_dir = paths["data"]
    data_dir.mkdir(parents=True, exist_ok=True)

    all_valid: List[dict[str, Any]] = []
    per_topic_valid: Dict[str, List[dict[str, Any]]] = {}
    for topic in cfg.topics:
        per_topic_valid[topic] = []
        for rec in iter_jsonl(_topic_file(ff_dir, topic)):
            pe = _get_processed_example(rec)
            ff_stage = pe.get("final_filter", {}) if isinstance(pe, dict) else {}
            if isinstance(ff_stage, dict) and ff_stage.get("stage_passed", False):
                rec = dict(rec)
                pe = _get_processed_example(rec)
                post = pe.get("postprocess", {}) if isinstance(pe, dict) else {}
                res = pe.get("resources", {}) if isinstance(pe, dict) else {}
                goal = post.get("rewritten_goal", "") if isinstance(post, dict) else ""
                steps = post.get("rewritten_steps", []) if isinstance(post, dict) else []
                resources = res.get("resources", []) if isinstance(res, dict) else []
                rec["final_procedure"] = {
                    "goal": goal if isinstance(goal, str) else "",
                    "steps": steps if isinstance(steps, list) else [],
                    "resources": resources if isinstance(resources, list) else [],
                }
                per_topic_valid[topic].append(rec)
                all_valid.append(rec)

    export_res = export_all_valid(
        records=all_valid,
        out_dir=data_dir,
        stem="all_valid",
        fmt=cfg.export.format,
        max_file_size=cfg.export.max_file_size,
    )

    # Also export a flat bench-compatible file (ready for h2e bench).
    flat_records: List[dict[str, Any]] = []
    for rec in all_valid:
        src = rec.get("source_example", {})
        fp = rec.get("final_procedure", {})
        flat = {
            "source_example_id": src.get("id", "") if isinstance(src, dict) else "",
            "topic": src.get("topic", "") if isinstance(src, dict) else "",
            "goal": fp.get("goal", "") if isinstance(fp, dict) else "",
            "steps": fp.get("steps", []) if isinstance(fp, dict) else [],
            "resources": fp.get("resources", []) if isinstance(fp, dict) else [],
        }
        flat_records.append(flat)

    bench_export_res = export_all_valid(
        records=flat_records,
        out_dir=data_dir,
        stem="all_valid_flat",
        fmt=cfg.export.format,
        max_file_size=cfg.export.max_file_size,
    )

    # Summary: counts per topic + totals.
    rows: list[dict[str, object]] = []
    for topic in cfg.topics:
        n_valid = len(per_topic_valid.get(topic, []))
        rows.append({"topic": topic, "processed": n_valid, "passed": n_valid})
    if export_res.shard_paths:
        if len(export_res.shard_paths) == 1:
            note = f"📦 exported: data/{export_res.shard_paths[0].name}"
        else:
            note = (
                f"📦 exported: {len(export_res.shard_paths)} shards "
                f"(data/{export_res.shard_paths[0].name} … data/{export_res.shard_paths[-1].name})"
            )
    else:
        note = "📦 exported: (no output)"
    if bench_export_res.shard_paths:
        if len(bench_export_res.shard_paths) == 1:
            note += f"\n  📦 bench:    data/{bench_export_res.shard_paths[0].name}"
        else:
            note += (
                f"\n  📦 bench:    {len(bench_export_res.shard_paths)} shards "
                f"(data/{bench_export_res.shard_paths[0].name} … data/{bench_export_res.shard_paths[-1].name})"
            )
    _print_stage_table(
        rows=rows,
        totals={"valid": len(all_valid)},
        note=note,
    )
    _print_stage_elapsed("export")


def _print_resume_summary(cfg: How2MineConfig) -> None:
    """Print a concise per-topic summary of existing data when resuming."""
    paths = stage_paths(cfg.out_root)
    procedures_dir = paths["procedures"]
    ff_dir = paths["final_filter"]

    rows: list[tuple[str, int, int]] = []
    for topic in cfg.topics:
        extracted = count_nonempty_lines(_topic_file(procedures_dir, topic))
        valid = _count_stage_passed(_topic_file(ff_dir, topic), "final_filter")
        rows.append((topic, extracted, valid))

    if not any(extracted > 0 for _, extracted, _ in rows):
        return  # Fresh run, nothing to show.

    table = Table(title="Resuming from existing data", expand=False, show_edge=False)
    table.add_column("topic", style="dim")
    table.add_column("processed", justify="right")
    table.add_column("final valid", justify="right", style="green")
    if cfg.targets.desired_valid_per_topic:
        table.add_column("target", justify="right", style="dim")
    table.add_column("budget used", justify="right", style="dim")

    for topic, extracted, valid in rows:
        row: list[str] = [
            topic,
            str(extracted),
            str(valid),
        ]
        if cfg.targets.desired_valid_per_topic:
            row.append(f"{valid}/{cfg.targets.desired_valid_per_topic}")
        row.append(f"{extracted}/{cfg.targets.candidates_per_topic}")
        table.add_row(*row)

    console.print()
    console.print(table)


def run_pipeline(cfg: How2MineConfig) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, out_path=cfg.out_root / "manifest.json", status="started")
    _print_pipeline_legend()
    _print_resume_summary(cfg)

    def _make_client(stage: str) -> JSONClient:
        st = cfg.resolve_stage_llm(stage)
        if st.backend == "vllm":
            ov = cfg.stage_llm_overrides.get(stage)
            vllm_overrides = ov.vllm if ov is not None else None
            return VLLMClient(
                VLLMConfig.from_overrides(
                    model=st.model,
                    temperature=st.temperature,
                    max_new_tokens=st.max_new_tokens,
                    overrides=vllm_overrides,
                    base_dir=Path.cwd(),
                )
            )
        return DelugeClient(
            DelugeLLMConfig(
                model=st.model,
                max_requests_per_minute=st.max_requests_per_minute,
                max_tokens_per_minute=st.max_tokens_per_minute,
                json_mode=True,
                temperature=st.temperature,
                max_new_tokens=st.max_new_tokens,
            )
        )

    procedures_client = _make_client("procedures")
    filter_client = _make_client("filter")
    postprocess_client = _make_client("postprocess")
    resources_client = _make_client("resources")
    final_filter_client = _make_client("final_filter")
    all_clients: list[JSONClient] = [
        procedures_client,
        filter_client,
        postprocess_client,
        resources_client,
        final_filter_client,
    ]

    if not cfg.targets.desired_valid_per_topic:
        run_extract(cfg, procedures_client)
        run_prefilter(cfg)
        run_filter(cfg, filter_client)
        run_postprocess(cfg, postprocess_client)
        run_tools(cfg, resources_client)
        run_final_filter(cfg, final_filter_client)
    else:
        # Iterative mode: loop until desired_valid_per_topic is reached or budget exhausted.
        paths = stage_paths(cfg.out_root)
        procedures_dir = paths["procedures"]
        ff_dir = paths["final_filter"]
        desired = int(cfg.targets.desired_valid_per_topic)
        batch = int(cfg.targets.extract_batch_size)
        budget = int(cfg.targets.candidates_per_topic)

        round_idx = 0
        while True:
            round_idx += 1
            valid_counts = {
                t: _count_stage_passed(_topic_file(ff_dir, t), "final_filter") for t in cfg.topics
            }
            if all(v >= desired for v in valid_counts.values()):
                console.print(
                    f"\n[bold green]✅  desired_valid_per_topic reached[/bold green] [dim](desired={desired})[/dim]"
                )
                break

            # Topics that already met the yield target — skip extraction for them.
            satisfied = {t for t, v in valid_counts.items() if v >= desired}

            exhausted = {
                t: (t in satisfied or count_nonempty_lines(_topic_file(procedures_dir, t)) >= budget)
                for t in cfg.topics
            }
            if all(exhausted.values()):
                console.print(
                    f"\n[bold yellow]⚠️  candidates_per_topic budget exhausted[/bold yellow] "
                    f"[dim]before reaching desired_valid_per_topic (desired={desired})[/dim]"
                )
                break

            still_needed = [t for t in cfg.topics if t not in satisfied and not exhausted[t]]
            console.print()
            console.print(
                Rule(
                    f"🔁  Round {round_idx}  [dim](desired_valid_per_topic={desired}, batch={batch}, candidates_per_topic={budget})[/dim]",
                    style="bold cyan",
                )
            )
            if satisfied:
                for t in sorted(satisfied):
                    console.print(f"  [green]✓[/green] [dim]{t}[/dim] — target reached ({valid_counts[t]}/{desired}), skipping")
            budget_exhausted = {t for t in cfg.topics if t not in satisfied and exhausted[t]}
            if budget_exhausted:
                for t in sorted(budget_exhausted):
                    console.print(f"  [yellow]![/yellow] [dim]{t}[/dim] — budget exhausted ({valid_counts[t]}/{desired} valid)")
            if still_needed:
                for t in still_needed:
                    console.print(f"  [cyan]→[/cyan] {t} — {valid_counts[t]}/{desired} valid, continuing")
            res = run_extract(cfg, procedures_client, max_new_per_topic=batch, skip_topics=satisfied)
            run_prefilter(cfg)
            run_filter(cfg, filter_client)
            run_postprocess(cfg, postprocess_client)
            run_tools(cfg, resources_client)
            run_final_filter(cfg, final_filter_client)

            if int(res.get("new_processed", 0)) <= 0:
                console.print("\n[bold yellow]⚠️  No new candidates extracted; stopping early.[/bold yellow]")
                break
    run_splits(cfg)
    # Aggregate usage across the whole run (all rounds/stages).
    u = _sum_usage(all_clients)
    if isinstance(u, dict) and (u.get("cost_usd") or u.get("input_tokens") or u.get("output_tokens")):
        cost = float(u.get("cost_usd", 0.0))
        it = int(u.get("input_tokens", 0))
        ot = int(u.get("output_tokens", 0))
        crt = int(u.get("cache_read_tokens", 0))
        cwt = int(u.get("cache_write_tokens", 0))
        parts = f"[green]💰 ${cost:.4f}[/green]   [dim]🔡 {it:,} tokens in / {ot:,} tokens out[/dim]"
        if crt or cwt:
            parts += f"   [dim]🗄️ {crt:,} cache_read / {cwt:,} cache_write[/dim]"
        console.print()
        console.print(Panel(parts, title="[bold]How2Mine Total[/bold]", border_style="green", expand=False))
    write_manifest(cfg, out_path=cfg.out_root / "manifest.json", status="completed")
