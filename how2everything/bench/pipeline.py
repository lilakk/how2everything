from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from how2everything.bench.aggregate import write_aggregates
from how2everything.bench.config import (
    BenchConfig,
    SuiteConfig,
    model_spec_to_bench_config,
)
from how2everything.bench.generate import is_generation_complete, run_generate
from how2everything.bench.judge import is_judging_complete, run_judge
from how2everything.llm.deluge_client import DelugeClient, DelugeLLMConfig
from how2everything.llm.vllm_client import VLLMClient, VLLMConfig


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


def write_manifest(cfg: BenchConfig, *, out_path: Path, status: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "timestamp_unix": time.time(),
        "git_sha": _safe_git_sha(),
        "config": _jsonable(asdict(cfg)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_client(cfg: BenchConfig, stage: str):
    st = cfg.resolve_stage_llm(stage)
    if st.backend == "vllm":
        return VLLMClient(
            VLLMConfig.from_overrides(
                model=st.model,
                temperature=st.temperature,
                max_new_tokens=st.max_new_tokens,
                overrides=st.vllm_overrides,
                base_dir=Path.cwd(),
            )
        )
    return DelugeClient(
        DelugeLLMConfig(
            model=st.model,
            max_requests_per_minute=st.max_requests_per_minute,
            max_tokens_per_minute=st.max_tokens_per_minute,
            # Only enforce JSON mode for schema-constrained stages (judge).
            json_mode=(stage != "generate"),
            temperature=st.temperature,
            max_new_tokens=st.max_new_tokens,
            reasoning_effort=st.reasoning_effort,
        )
    )


def run_pipeline(cfg: BenchConfig) -> None:
    if cfg.generator is None:
        raise ValueError("Config is missing `generator:`. It is required for `h2e bench run`.")

    cfg.out_root.mkdir(parents=True, exist_ok=True)
    manifest = cfg.out_root / "manifest.json"
    write_manifest(cfg, out_path=manifest, status="started")

    print("\n=== how2bench ===", flush=True)
    print(f"out_root: {cfg.out_root}", flush=True)

    gen_path = cfg.generations_path()
    if is_generation_complete(cfg, out_path=gen_path):
        print("\n▶ how2bench / Generate", flush=True)
        print("  skip: all examples already generated", flush=True)
    else:
        gen_client = make_client(cfg, "generate")
        run_generate(cfg, gen_client, out_path=gen_path)
        del gen_client  # free GPU memory before loading judge

    judge_path = cfg.judgments_path()
    if is_judging_complete(cfg, generations_path=gen_path, out_path=judge_path):
        print("\n▶ how2bench / Judge", flush=True)
        print("  skip: all generations already judged", flush=True)
    else:
        judge_client = make_client(cfg, "judge")
        judge_path = run_judge(
            cfg,
            judge_client,
            generations_path=gen_path,
            out_path=judge_path,
        )
    # Write aggregates alongside the judgments (per judge_id).
    summary = write_aggregates(out_root=judge_path.parent, judgments_path=judge_path)

    # End-of-run summary stats (release-friendly).
    try:
        how2score_percent = float(summary.get("how2score_percent", 0.0)) if isinstance(summary, dict) else 0.0
        n_examples = int(summary.get("n_examples", 0)) if isinstance(summary, dict) else 0
        n_with_fail = int(summary.get("n_with_failures", 0)) if isinstance(summary, dict) else 0
        failure_rate = float(summary.get("failure_rate", 0.0)) if isinstance(summary, dict) else 0.0
        avg_fail = float(summary.get("avg_failures_per_example", 0.0)) if isinstance(summary, dict) else 0.0
    except Exception:
        how2score_percent, n_examples, n_with_fail, failure_rate, avg_fail = 0.0, 0, 0, 0.0, 0.0

    print("\nSummary", flush=True)
    print(f"  how2score: {how2score_percent:.1f}%", flush=True)
    print(f"  n_examples: {n_examples}", flush=True)

    write_manifest(cfg, out_path=manifest, status="completed")


# ---------------------------------------------------------------------------
# Suite-level runners (loop over models)
# ---------------------------------------------------------------------------


def _collect_result(bench_cfg: BenchConfig, spec_model: str, spec_run_name: str, generator_id: str, prompt_style: str) -> dict[str, Any]:
    """Read summary.json for a completed model run and return a result dict."""
    how2score_percent = 0.0
    n_examples = 0
    try:
        judge_dir = bench_cfg.judge_dir_name()
        summary_path = bench_cfg.out_root / "judgments" / judge_dir / "aggregate" / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                how2score_percent = float(summary.get("how2score_percent", 0.0))
                n_examples = int(summary.get("n_examples", 0))
    except Exception:
        how2score_percent, n_examples = 0.0, 0

    return {
        "generator_model": spec_model,
        "run_name": spec_run_name,
        "generator_id": generator_id,
        "prompt_style": prompt_style,
        "how2score": how2score_percent,
        "n_examples": n_examples,
    }


def run_suite(suite: SuiteConfig) -> list[dict[str, Any]]:
    """Run gen -> judge -> aggregate for each model in the suite."""
    if not suite.models:
        raise ValueError("No models configured. Add entries to `models:` in your config.")

    suite.out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for spec in suite.models:
        bench_cfg = model_spec_to_bench_config(suite, spec)
        generator_id = bench_cfg.generator_id()

        print(f"\n=== how2bench: {spec.model} ({spec.prompt_style}) ===", flush=True)
        print(f"out_root: {bench_cfg.out_root}", flush=True)
        run_pipeline(bench_cfg)

        results.append(_collect_result(bench_cfg, spec.model, spec.run_name, generator_id, spec.prompt_style))

    return results


def gen_suite(suite: SuiteConfig) -> None:
    """Run generation only for each model in the suite."""
    if not suite.models:
        raise ValueError("No models configured. Add entries to `models:` in your config.")

    suite.out_root.mkdir(parents=True, exist_ok=True)

    for spec in suite.models:
        bench_cfg = model_spec_to_bench_config(suite, spec)
        print(f"\n=== how2bench gen: {spec.model} ({spec.prompt_style}) ===", flush=True)
        print(f"out_root: {bench_cfg.out_root}", flush=True)
        bench_cfg.out_root.mkdir(parents=True, exist_ok=True)
        gen_client = make_client(bench_cfg, "generate")
        run_generate(bench_cfg, gen_client, out_path=bench_cfg.generations_path())


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a box-drawing summary table of suite results."""
    if not results:
        return

    cols = ["run_name", "generator_model", "prompt_style", "how2score", "n_examples"]
    widths: dict[str, int] = {c: len(c) for c in cols}
    rows: list[dict[str, str]] = []
    for r in results:
        tr = {
            "run_name": str(r.get("run_name", "")),
            "generator_model": str(r.get("generator_model", "")),
            "prompt_style": str(r.get("prompt_style", "")),
            "how2score": f"{float(r.get('how2score', 0.0)):.1f}",
            "n_examples": str(int(r.get("n_examples", 0))),
        }
        rows.append(tr)
        for c in cols:
            widths[c] = max(widths[c], len(tr.get(c, "")))

    def _border(left: str, mid: str, right: str, fill: str = "─") -> str:
        segs = [fill * (widths[c] + 2) for c in cols]
        return left + mid.join(segs) + right

    def _fmt(tr: dict[str, str]) -> str:
        parts: list[str] = []
        for c in cols:
            s = tr.get(c, "")
            align = ">" if c in {"how2score", "n_examples"} else "<"
            parts.append(f"{s:{align}{widths[c]}}")
        return "│ " + " │ ".join(parts) + " │"

    print("\n=== how2bench results ===", flush=True)
    print(_border("┌", "┬", "┐"))
    print(_fmt({c: c for c in cols}))
    print(_border("├", "┼", "┤"))
    for tr in rows:
        print(_fmt(tr))
    print(_border("└", "┴", "┘"))
