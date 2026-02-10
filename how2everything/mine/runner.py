from __future__ import annotations

import os


def validate_config_and_env(config_path: str) -> None:
    from how2everything.mine.config import load_config
    from how2everything.mine.document_sources import iter_input_paths
    from how2everything.mine.export_utils import parse_size
    from how2everything.llm.vllm_client import VLLMConfig as LocalVLLMConfig

    cfg = load_config(config_path)

    # Config paths
    if not cfg.inputs.path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.inputs.path}")
    if cfg.inputs.path.is_dir():
        if not cfg.inputs.include_globs:
            raise ValueError(
                "When inputs.path is a directory, you must set inputs.include_globs (e.g. ['*.jsonl', '*.jsonl.zst', '*.arrow'])."
            )
        matched = iter_input_paths(cfg.inputs.path, cfg.inputs.include_globs)
        if not matched:
            raise ValueError(
                f"inputs.include_globs matched 0 files in directory: {cfg.inputs.path}"
            )

    # Target sanity checks
    if cfg.targets.candidates_per_topic <= 0:
        raise ValueError("targets.candidates_per_topic must be > 0.")
    if cfg.targets.extract_batch_size <= 0:
        raise ValueError("targets.extract_batch_size must be > 0.")
    if cfg.targets.desired_valid_per_topic is not None:
        if cfg.targets.desired_valid_per_topic > cfg.targets.candidates_per_topic:
            raise ValueError(
                "targets.desired_valid_per_topic cannot exceed targets.candidates_per_topic. "
                f"Got desired_valid_per_topic={cfg.targets.desired_valid_per_topic} "
                f"candidates_per_topic={cfg.targets.candidates_per_topic}."
            )

    # Export sanity checks
    fmt = (cfg.export.format or "jsonl").strip()
    if fmt not in {"jsonl", "jsonl_zst", "parquet", "arrow_ipc"}:
        raise ValueError(f"Unsupported export.format: {fmt} (expected jsonl|jsonl_zst|parquet|arrow_ipc).")
    if cfg.export.max_file_size:
        parse_size(cfg.export.max_file_size)
    for p in (
        cfg.prompts.procedures,
        cfg.prompts.filter,
        cfg.prompts.postprocess,
        cfg.prompts.resources,
        cfg.prompts.final_filter,
    ):
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")

    # Per-stage backend checks.
    # For deluge stages we check the provider's key; for vLLM stages we check vllm importability.
    def _check_provider(p: str) -> None:
        provider = (p or "").lower().strip()
        if provider == "openai" and not (os.environ.get("OPENAI_API_KEY")):
            raise EnvironmentError("Missing required env var: OPENAI_API_KEY")
        if provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY")):
            raise EnvironmentError("Missing required env var: ANTHROPIC_API_KEY")
        if provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise EnvironmentError("Missing required env var: GEMINI_API_KEY (or GOOGLE_API_KEY)")

    for stage in ("procedures", "filter", "postprocess", "resources", "final_filter"):
        st = cfg.resolve_stage_llm(stage)
        if st.backend == "vllm":
            try:
                import vllm  # noqa: F401
            except Exception as e:
                raise EnvironmentError(
                    f"Stage {stage} is configured with backend=vllm but vllm is not importable. "
                    "Install optional deps (e.g. `pip install -e '.[vllm]'`)."
                ) from e
            if not st.model:
                raise ValueError(f"Stage {stage} backend=vllm requires a model name.")
            # Validate vLLM override block structure early (without initializing the model).
            ov = cfg.stage_llm_overrides.get(stage)
            vllm_overrides = ov.vllm if ov is not None else None
            LocalVLLMConfig.from_overrides(
                model=st.model,
                temperature=st.temperature,
                max_new_tokens=st.max_new_tokens,
                overrides=vllm_overrides,
            )
        else:
            _check_provider(st.provider)


def run_from_config(config_path: str) -> None:
    from how2everything.mine.config import load_config
    from how2everything.mine.pipeline import run_pipeline

    cfg = load_config(config_path)
    run_pipeline(cfg)

