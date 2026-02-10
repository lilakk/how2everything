from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from how2everything.bench.config import BenchConfig
from how2everything.bench.io_utils import attempted_example_ids, read_jsonl_by_id
from how2everything.bench.schemas import JudgeResult, JudgmentRecord, LLMInfo
from how2everything.mine.io_utils import load_prompt, write_jsonl


def _print_stage_banner(name: str) -> None:
    title = name.replace("_", " ").strip().title()
    print(f"\n▶ how2bench / {title}", flush=True)


def _steps_text(steps: list[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))


def is_judging_complete(
    cfg: BenchConfig,
    *,
    generations_path: Path | None = None,
    out_path: Path | None = None,
) -> bool:
    """Return True if all generations already have judgments on disk."""
    out_root = cfg.out_root
    generations_path = generations_path or (out_root / "generations.jsonl")
    out_path = out_path or (out_root / "judgments.jsonl")
    if not generations_path.exists() or not out_path.exists():
        return False
    gen_ids = set(read_jsonl_by_id(generations_path, key="source_example_id").keys())
    if not gen_ids:
        return True  # nothing to judge
    already = attempted_example_ids(out_path, key="source_example_id")
    return gen_ids.issubset(already)


def run_judge(
    cfg: BenchConfig,
    client,
    *,
    generations_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Judge predicted steps against references.

    Reads `generations.jsonl` and writes `judgments.jsonl` (append-only, resumable).
    """
    out_root = cfg.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    generations_path = generations_path or (out_root / "generations.jsonl")
    out_path = out_path or (out_root / "judgments.jsonl")

    gen_by_id = read_jsonl_by_id(generations_path, key="source_example_id")
    if not gen_by_id:
        _print_stage_banner("judge")
        print("  skip: no generations found", flush=True)
        return out_path

    already = attempted_example_ids(out_path, key="source_example_id")
    _print_stage_banner("judge")
    if already:
        print(f"  existing: {len(already)}/{len(gen_by_id)}", flush=True)

    template = load_prompt(cfg.prompts.judge)
    judge_prompt_sha = cfg.judge_prompt_sha256()
    judge_id = cfg.judge_id()

    st = cfg.resolve_stage_llm("judge")
    judge_info = LLMInfo(backend=st.backend, provider=st.provider, model=st.model)

    # Write a small manifest next to the judgments file for reproducibility.
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = cfg.judge_manifest_path()
        manifest = {
            "judge_id": judge_id,
            "judge_prompt_sha256": judge_prompt_sha,
            "judge_prompt_path": str(cfg.prompts.judge),
            "judge": {
                "backend": st.backend,
                "provider": st.provider,
                "model": st.model,
                "temperature": st.temperature,
                "max_new_tokens": st.max_new_tokens,
                "reasoning_effort": st.reasoning_effort,
                "vllm_overrides": st.vllm_overrides or {},
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        # Best-effort: never block judging if we can't write the manifest.
        pass

    # Stable iteration order (read_jsonl_by_id overwrites; dict order is insertion in py3.11,
    # but we want deterministic file-order independent ordering)
    example_ids = sorted(gen_by_id.keys())
    to_run_ids = [eid for eid in example_ids if eid not in already]
    if not to_run_ids:
        print("  skip: all generations already judged", flush=True)
        return out_path

    prompts: list[str] = []
    metas: list[dict[str, Any]] = []
    for eid in to_run_ids:
        rec = gen_by_id[eid]
        goal = rec.get("goal", "")
        ref_steps = rec.get("steps", [])
        resources = rec.get("resources", [])
        pred_steps = rec.get("predicted_steps", [])
        if not isinstance(goal, str):
            goal = str(goal)
        if not isinstance(ref_steps, list):
            ref_steps = []
        if not isinstance(resources, list):
            resources = []
        if not isinstance(pred_steps, list):
            pred_steps = []
        goal = goal.strip()
        ref_steps = [str(s).strip() for s in ref_steps if str(s).strip()]
        resources = [str(s).strip() for s in resources if str(s).strip()]
        pred_steps = [str(s).strip() for s in pred_steps if str(s).strip()]
        if not goal or not ref_steps:
            continue
        prompt = template.format(
            goal=goal,
            reference_steps=_steps_text(ref_steps),
            steps=_steps_text(pred_steps),
        )
        # OpenAI JSON response_format requires the prompt to mention 'json' explicitly.
        prompt += "\n\nReturn only valid json."
        prompts.append(prompt)
        metas.append(
            {
                "source_example_id": eid,
                "topic": rec.get("topic", "") if isinstance(rec.get("topic", ""), str) else "",
                "goal": goal,
                "steps": ref_steps,
                "resources": resources,
                "predicted_steps": pred_steps,
            }
        )

    if not prompts:
        print("  skip: no eligible prompts constructed", flush=True)
        return out_path

    print(f"  running: {len(prompts)} examples", flush=True)
    results = client.complete_json(
        prompts,
        output_schema=JudgeResult,
        show_progress=True,
        progress_desc="bench/judge",
    )

    n_parse_failed = 0
    out_records: list[dict[str, Any]] = []
    for meta, parsed in zip(metas, results):
        # Detect parse failure: complete_json returns {} when it cannot parse.
        parse_failed = not parsed

        reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""
        cfs = parsed.get("critical_failures", []) if isinstance(parsed, dict) else []
        if not isinstance(cfs, list):
            cfs = []
        # Normalize entries
        norm_cfs: list[dict[str, Any]] = []
        for cf in cfs:
            if not isinstance(cf, dict):
                continue
            failure = cf.get("failure", "")
            l1 = cf.get("L1_steps", [])
            l2 = cf.get("L2_steps", [])
            norm_cfs.append(
                {
                    "failure": failure if isinstance(failure, str) else str(failure),
                    "L1_steps": [int(x) for x in l1] if isinstance(l1, list) else [],
                    "L2_steps": [int(x) for x in l2] if isinstance(l2, list) else [],
                }
            )
        has_failure = len(norm_cfs) > 0 or parse_failed
        if parse_failed:
            n_parse_failed += 1
        jr = JudgmentRecord(
            judge_id=judge_id,
            judge_prompt_sha256=judge_prompt_sha,
            source_example_id=meta["source_example_id"],
            topic=meta.get("topic", ""),
            goal=meta.get("goal", ""),
            steps=meta.get("steps", []),
            resources=meta.get("resources", []),
            predicted_steps=meta.get("predicted_steps", []),
            judge=judge_info,
            reasoning=reasoning if isinstance(reasoning, str) else str(reasoning),
            critical_failures=norm_cfs,  # pydantic will coerce
            has_failure=has_failure,
            n_failures=len(norm_cfs),
            parse_failed=parse_failed,
        )
        out_records.append(jr.model_dump())

    if n_parse_failed:
        print(f"  warning: {n_parse_failed} example(s) had unparseable judge output", flush=True)

    if out_records:
        write_jsonl(out_path, out_records, mode="a")
    print(f"  wrote: {out_path} (+{len(out_records)})", flush=True)
    return out_path

