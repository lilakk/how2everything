from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from how2everything.bench.config import BenchConfig
from how2everything.bench.dataset import load_bench_examples
from how2everything.bench.io_utils import attempted_example_ids
from how2everything.bench.schemas import BenchExample, GenerationRecord, LLMInfo
from how2everything.mine.io_utils import load_prompt, write_jsonl


def _print_stage_banner(name: str) -> None:
    title = name.replace("_", " ").strip().title()
    print(f"\n▶ how2bench / {title}", flush=True)


def _resources_text(resources: list[str]) -> str:
    if not resources:
        return "[]"
    return json.dumps(resources, ensure_ascii=False)


def _format_generate_prompt(template: str, ex: BenchExample) -> str:
    n = len(ex.steps)
    return template.format(goal=ex.goal, resources=_resources_text(ex.resources), n=n)


def _extract_steps_from_text(text: str) -> list[str]:
    """Best-effort heuristic to turn a free-form model completion into a list of steps.

    We look for lines beginning with an ordered list marker (e.g. "1.", "2)", etc.).
    If none are found, we fall back to splitting on newlines – trimming empty lines.
    """
    import re

    if not isinstance(text, str):
        return []
    t = text.strip()
    if not t:
        return []
    # Strip reasoning/thinking tags (e.g. from chain-of-thought models).
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1].strip()
    if "<answer>" in t and "</answer>" in t:
        t = t.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    steps: list[str] = []
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        # Match patterns like "1.", "2)", "3 -", "4:"
        m = re.match(r"^\s*\d+\s*[).:-]?\s*(.*)", line)
        if m:
            s = (m.group(1) or "").strip()
            if s:
                steps.append(s)
    if not steps:
        # Fallback: treat each non-empty line as a step
        steps = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return steps


def is_generation_complete(cfg: BenchConfig, *, out_path: Path | None = None) -> bool:
    """Return True if all benchmark examples already have generations on disk."""
    out_path = out_path or (cfg.out_root / "generations.jsonl")
    if not out_path.exists():
        return False
    examples = load_bench_examples(cfg.inputs)
    already = attempted_example_ids(out_path, key="source_example_id")
    return all(ex.source_example_id in already for ex in examples)


def run_generate(cfg: BenchConfig, client, *, out_path: Path | None = None) -> Path:
    """
    Generate predicted steps for each benchmark example.

    Writes `generations.jsonl` (append-only, resumable by source_example_id).
    """
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (out_root / "generations.jsonl")

    examples = load_bench_examples(cfg.inputs)
    already = attempted_example_ids(out_path, key="source_example_id")

    _print_stage_banner("generate")
    if already:
        print(f"  existing: {len(already)}/{len(examples)}", flush=True)

    template = load_prompt(
        cfg.prompts.inference_base if cfg.generation.model_type == "base" else cfg.prompts.inference_inst
    )

    st = cfg.resolve_stage_llm("generate")
    gen_info = LLMInfo(backend=st.backend, provider=st.provider, model=st.model)
    generation_prompt_sha = cfg.generator_prompt_sha256()
    generator_id = cfg.generator_id()

    # Write a small manifest next to the generations file for reproducibility.
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = out_path.parent / "generation_manifest.json"
        manifest = {
            "generator_id": generator_id,
            "generation_prompt_sha256": generation_prompt_sha,
            "generation_prompt_path": str(cfg.generator_prompt_path()),
            "generation": {"model_type": cfg.generation.model_type},
            "generator": {
                "backend": st.backend,
                "provider": st.provider,
                "model": st.model,
                "temperature": st.temperature,
                "max_new_tokens": st.max_new_tokens,
                "max_requests_per_minute": st.max_requests_per_minute,
                "max_tokens_per_minute": st.max_tokens_per_minute,
                "reasoning_effort": st.reasoning_effort,
                "vllm_overrides": st.vllm_overrides or {},
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Best-effort: never block generation if we can't write the manifest.
        pass

    to_run: list[BenchExample] = [ex for ex in examples if ex.source_example_id not in already]
    if not to_run:
        print("  skip: all examples already generated", flush=True)
        return out_path

    prompts = [_format_generate_prompt(template, ex) for ex in to_run]
    print(f"  running: {len(to_run)} examples", flush=True)
    completions = client.complete(
        prompts,
        show_progress=True,
        progress_desc="bench/generate",
    )

    records: list[dict[str, Any]] = []
    for ex, completion, prompt in zip(to_run, completions, prompts):
        completion_text = completion if isinstance(completion, str) else str(completion)
        predicted_steps = _extract_steps_from_text(completion)
        rec = GenerationRecord(
            generator_id=generator_id,
            generation_prompt_sha256=generation_prompt_sha,
            source_example_id=ex.source_example_id,
            topic=ex.topic,
            goal=ex.goal,
            steps=ex.steps,
            resources=ex.resources,
            model_completion=completion_text,
            predicted_steps=predicted_steps,
            prompt=prompt,
            generator=gen_info,
        )
        records.append(rec.model_dump())

    if records:
        write_jsonl(out_path, records, mode="a")
    print(f"  wrote: {out_path} (+{len(records)})", flush=True)
    return out_path

