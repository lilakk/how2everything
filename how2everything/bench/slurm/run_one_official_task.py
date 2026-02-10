from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from how2everything.bench.config import BenchConfig, load_suite_config, model_spec_to_bench_config, ModelSpec, SuiteConfig
from how2everything.bench.pipeline import run_pipeline


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m how2everything.bench.slurm.run_one_official_task",
        description="Run exactly one model spec from a how2bench suite YAML (for Slurm arrays).",
    )
    p.add_argument("--config", required=True, help="Path to how2bench suite YAML config.")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Model index to run (0-based). If omitted, uses $SLURM_ARRAY_TASK_ID.",
    )
    p.add_argument(
        "--print-num-models",
        action="store_true",
        help="Print the number of model entries in the config and exit.",
    )
    p.add_argument(
        "--print-gpu-groups",
        action="store_true",
        help="Print JSON mapping of gpu_count -> [model_indices] and exit.",
    )
    return p.parse_args(argv)


def _tp_size(vllm_overrides: Mapping[str, Any] | None) -> int:
    """Extract tensor_parallel_size from a vllm overrides dict, defaulting to 1."""
    if not vllm_overrides:
        return 1
    engine_kwargs = vllm_overrides.get("engine_kwargs")
    if not isinstance(engine_kwargs, Mapping):
        return 1
    tp = engine_kwargs.get("tensor_parallel_size", 1)
    return max(int(tp), 1)


def _gpu_groups(suite: SuiteConfig) -> dict[int, list[int]]:
    """Group model indices by the number of GPUs they need.

    GPU count per model = max(generator_tp, evaluator_tp) so the Slurm task
    can run both the generator and the judge on the same node.
    """
    eval_tp = 1
    if suite.evaluator is not None:
        eval_tp = _tp_size(suite.evaluator.vllm)

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i, spec in enumerate(suite.models):
        gen_tp = _tp_size(spec.vllm)
        gpus = max(gen_tp, eval_tp)
        groups[gpus].append(i)

    return dict(sorted(groups.items()))


def _read_index(ns: argparse.Namespace) -> int:
    if ns.index is not None:
        return int(ns.index)
    env = os.environ.get("SLURM_ARRAY_TASK_ID", "")
    if env.strip():
        try:
            return int(env.strip())
        except Exception as e:
            raise ValueError(f"Invalid SLURM_ARRAY_TASK_ID: {env!r}") from e
    raise ValueError("Missing --index and SLURM_ARRAY_TASK_ID is not set.")


def _summary_for(cfg: BenchConfig) -> tuple[float, int]:
    """Best-effort read of how2score summary produced by run_pipeline()."""
    try:
        judge_dir = cfg.judge_dir_name()
        summary_path = cfg.out_root / "judgments" / judge_dir / "aggregate" / "summary.json"
        if not summary_path.exists():
            return 0.0, 0
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return 0.0, 0
        return float(obj.get("how2score_percent", 0.0)), int(obj.get("n_examples", 0))
    except Exception:
        return 0.0, 0


def run_one_official_model(*, config_path: str, index: int) -> int:
    suite = load_suite_config(config_path)
    if index < 0 or index >= len(suite.models):
        raise IndexError(f"index out of range: {index} (expected 0..{len(suite.models)-1})")

    spec = suite.models[index]
    cfg = model_spec_to_bench_config(suite, spec)

    print("\n=== how2bench official (slurm task) ===", flush=True)
    print(f"index: {index}", flush=True)
    print(f"model: {spec.model}", flush=True)
    print(f"prompt_style: {spec.prompt_style}", flush=True)
    print(f"run_name: {spec.run_name}", flush=True)
    print(f"generator_id: {cfg.generator_id()}", flush=True)
    print(f"out_root: {cfg.out_root}", flush=True)

    run_pipeline(cfg)

    how2score_percent, n_examples = _summary_for(cfg)
    print("\nTask summary", flush=True)
    print(f"  how2score: {how2score_percent:.1f}%", flush=True)
    print(f"  n_examples: {n_examples}", flush=True)

    return 0


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    suite = load_suite_config(ns.config)
    if ns.print_num_models:
        print(len(suite.models))
        return 0
    if ns.print_gpu_groups:
        groups = _gpu_groups(suite)
        print(json.dumps({str(k): v for k, v in groups.items()}))
        return 0

    idx = _read_index(ns)
    return run_one_official_model(config_path=ns.config, index=idx)


if __name__ == "__main__":
    raise SystemExit(main())
