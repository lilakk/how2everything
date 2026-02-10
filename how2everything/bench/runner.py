from __future__ import annotations

import os


def validate_config_and_env(config_path: str) -> None:
    from how2everything.bench.config import load_suite_config, model_spec_to_bench_config, suite_to_judge_config
    from how2everything.mine.document_sources import iter_input_paths
    from how2everything.llm.vllm_client import VLLMConfig as LocalVLLMConfig

    suite = load_suite_config(config_path)

    # Input path checks
    if suite.inputs.path is not None:
        if not suite.inputs.path.exists():
            raise FileNotFoundError(f"Input path not found: {suite.inputs.path}")
        if suite.inputs.path.is_dir():
            if not suite.inputs.include_globs:
                raise ValueError(
                    "When inputs.path is a directory, you must set inputs.include_globs "
                    "(e.g. ['*.jsonl', '*.jsonl.zst', '*.arrow'])."
                )
            matched = iter_input_paths(suite.inputs.path, suite.inputs.include_globs)
            if not matched:
                raise ValueError(f"inputs.include_globs matched 0 files in directory: {suite.inputs.path}")
    else:
        # HF dataset mode: just validate the repo id string.
        if not suite.inputs.hf_repo:
            raise ValueError("inputs.path is omitted; inputs.hf_repo must be set (or omitted to use the default).")
        if suite.inputs.hf_cache_dir is not None:
            suite.inputs.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # Prompt paths
    for p in (suite.prompts.inference_base, suite.prompts.inference_inst, suite.prompts.judge):
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")

    # Backend/provider checks.
    def _check_provider(p: str) -> None:
        provider = (p or "").lower().strip()
        if provider == "openai" and not (os.environ.get("OPENAI_API_KEY")):
            raise EnvironmentError("Missing required env var: OPENAI_API_KEY")
        if provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY")):
            raise EnvironmentError("Missing required env var: ANTHROPIC_API_KEY")
        if provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise EnvironmentError("Missing required env var: GEMINI_API_KEY (or GOOGLE_API_KEY)")

    # Validate each model's backend/provider.
    for spec in suite.models:
        cfg = model_spec_to_bench_config(suite, spec)
        st = cfg.resolve_stage_llm("generate")
        if st.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except Exception as e:
                raise EnvironmentError(
                    f"Model {spec.model} is configured with backend=vllm but vllm is not importable. "
                    "Install optional deps (e.g. `pip install -e '.[vllm]'`)."
                ) from e
            if not st.model:
                raise ValueError(f"Model {spec.model} backend=vllm requires a model name.")
            LocalVLLMConfig.from_overrides(
                model=st.model,
                temperature=st.temperature,
                max_new_tokens=st.max_new_tokens,
                overrides=st.vllm_overrides,
            )
        else:
            _check_provider(st.provider)

    # Validate evaluator once (if explicitly configured).
    if suite.evaluator is not None:
        judge_cfg = suite_to_judge_config(suite)
        st = judge_cfg.resolve_stage_llm("judge")
        if st.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except Exception as e:
                raise EnvironmentError(
                    "Evaluator is configured with backend=vllm but vllm is not importable. "
                    "Install optional deps (e.g. `pip install -e '.[vllm]'`)."
                ) from e
            if not st.model:
                raise ValueError("Evaluator backend=vllm requires a model name.")
            LocalVLLMConfig.from_overrides(
                model=st.model,
                temperature=st.temperature,
                max_new_tokens=st.max_new_tokens,
                overrides=st.vllm_overrides,
            )
        else:
            _check_provider(st.provider)


def run_from_config(config_path: str) -> None:
    from how2everything.bench.config import load_suite_config
    from how2everything.bench.pipeline import run_suite, print_results_table

    suite = load_suite_config(config_path)
    results = run_suite(suite)
    print_results_table(results)
