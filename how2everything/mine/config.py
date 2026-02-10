from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


STAGE_NAMES: tuple[str, ...] = (
    "procedures",
    "filter",
    "postprocess",
    "resources",
    "final_filter",
)


@dataclass(frozen=True)
class InputsConfig:
    # Discriminator for the *semantic* shape of the input.
    # Public runner currently only supports document rows, so we default to "documents".
    # Keep "jsonl_documents" as a legacy alias for backward-compatible configs.
    path: Path
    kind: str = "documents"  # "documents" | "jsonl_documents" (legacy alias)
    format: str = "auto"  # auto|jsonl|csv|arrow|parquet
    compression: str = "auto"  # auto|none|zst|gz|bz2|xz
    # If `path` is a directory, include files matching any of these (non-recursive) globs.
    # Examples: ["*.jsonl", "*.jsonl.zst", "*.csv*", "*.arrow", "*.parquet"]
    include_globs: list[str] = field(default_factory=list)
    id_field: str = "id"
    text_field: str = "text"
    url_field: str = "url"
    topic_field: str = "topic"


@dataclass(frozen=True)
class LLMConfig:
    provider: str  # "openai" | "anthropic" | "gemini" (informational; model determines behavior in lm-deluge)
    model: str
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    temperature: float = 0.0
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class StageLLMConfig:
    """
    Optional per-stage LLM config override.

    Missing fields inherit from the top-level `llm` config.
    """

    backend: str = "deluge"  # deluge|vllm
    provider: str | None = None  # used for deluge env validation only
    model: str | None = None
    temperature: float | None = None
    max_new_tokens: int | None = None
    max_requests_per_minute: int | None = None
    max_tokens_per_minute: int | None = None
    # Backend-specific overrides (currently only used when backend == "vllm").
    # Parsed as a raw mapping and validated by the vLLM client config builder.
    vllm: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedStageLLM:
    backend: str  # deluge|vllm
    provider: str
    model: str
    temperature: float
    max_new_tokens: int
    max_requests_per_minute: int
    max_tokens_per_minute: int


@dataclass(frozen=True)
class PromptPaths:
    procedures: Path
    filter: Path
    postprocess: Path
    resources: Path
    final_filter: Path


@dataclass(frozen=True)
class Targets:
    # Max documents to extract per topic (the budget / safety cap).
    candidates_per_topic: int = 100
    # If set, the pipeline loops in batches until this many records pass the final
    # filter per topic (or the budget is exhausted). If None, the pipeline runs once.
    desired_valid_per_topic: int | None = None
    # When desired_valid_per_topic is set, extract up to this many new candidates
    # per topic per round.
    extract_batch_size: int = 50
    min_steps: int = 4
    max_steps: int = 15
    train_per_topic: int = 1
    test_per_topic: int = 1


@dataclass(frozen=True)
class ExportConfig:
    # Output format for final export (`out_root/data/all_valid.*`).
    # Supported: jsonl | jsonl_zst | parquet | arrow_ipc
    format: str = "jsonl"
    # Optional max shard size like "500MB", "2GB". If omitted/empty, write a single file.
    max_file_size: str | None = None


@dataclass(frozen=True)
class How2MineConfig:
    out_root: Path
    inputs: InputsConfig
    topics: list[str]
    # True iff `topics` in config is non-empty and should be used to filter inputs.
    # If False, pipeline routes all records into a single pseudo-topic ("all").
    use_topics: bool
    llm: LLMConfig
    stage_llm_overrides: dict[str, StageLLMConfig]
    prompts: PromptPaths
    targets: Targets
    export: ExportConfig

    def resolve_stage_llm(self, stage: str) -> ResolvedStageLLM:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {stage} (expected one of {list(STAGE_NAMES)})")
        ov = self.stage_llm_overrides.get(stage, StageLLMConfig())
        backend = (ov.backend or "deluge").strip().lower()
        if backend not in {"deluge", "vllm"}:
            raise ValueError(f"Unsupported backend for stage {stage}: {backend}")
        provider = (ov.provider or self.llm.provider).strip()
        model = (ov.model or self.llm.model).strip()
        temperature = float(self.llm.temperature if ov.temperature is None else ov.temperature)
        max_new_tokens = int(self.llm.max_new_tokens if ov.max_new_tokens is None else ov.max_new_tokens)
        max_rpm = int(
            self.llm.max_requests_per_minute
            if ov.max_requests_per_minute is None
            else ov.max_requests_per_minute
        )
        max_tpm = int(
            self.llm.max_tokens_per_minute
            if ov.max_tokens_per_minute is None
            else ov.max_tokens_per_minute
        )
        # For vLLM backends, provider/rate limits are not used, but we still resolve them.
        return ResolvedStageLLM(
            backend=backend,
            provider=provider,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_requests_per_minute=max_rpm,
            max_tokens_per_minute=max_tpm,
        )


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a YAML mapping (dict) at top-level.")
    return data


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    v = raw.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Config must set non-empty string: {key}")
    return v


def _require_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    v = raw.get(key)
    if not isinstance(v, Mapping):
        raise ValueError(f"Config must set mapping: {key}")
    return v


def _require_list_of_str(raw: Mapping[str, Any], key: str) -> list[str]:
    v = raw.get(key)
    if not isinstance(v, Sequence) or isinstance(v, (str, bytes)):
        raise ValueError(f"Config must set list[str]: {key}")
    out: list[str] = []
    for item in v:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Config {key} must contain non-empty strings.")
        out.append(item)
    return out


def load_config(config_path: str) -> How2MineConfig:
    cfg_path = Path(config_path)
    raw = _load_yaml(cfg_path)
    base_dir = Path.cwd()

    out_root = Path(_require_str(raw, "out_root"))
    if not out_root.is_absolute():
        out_root = (base_dir / out_root).resolve()

    inputs_raw = _require_mapping(raw, "inputs")
    inputs_path = Path(_require_str(inputs_raw, "path"))
    if not inputs_path.is_absolute():
        inputs_path = (base_dir / inputs_path).resolve()
    inputs_kind = str(inputs_raw.get("kind", "documents")).strip() or "documents"
    # Backward compatibility: older public docs used kind=jsonl_documents.
    # We keep that accepted, but allow kind=documents going forward.
    inputs_format = str(inputs_raw.get("format", "auto")).strip() or "auto"
    inputs_compression = str(inputs_raw.get("compression", "auto")).strip() or "auto"
    if inputs_kind == "jsonl_documents" and "format" not in inputs_raw:
        # Preserve old behavior explicitly.
        inputs_format = "jsonl"
    if inputs_kind not in {"documents", "jsonl_documents"}:
        raise ValueError(f"Unsupported inputs.kind for public runner: {inputs_kind}")

    include_globs_raw = inputs_raw.get("include_globs", [])
    if include_globs_raw is None:
        include_globs: list[str] = []
    elif isinstance(include_globs_raw, Sequence) and not isinstance(include_globs_raw, (str, bytes)):
        include_globs = [str(x).strip() for x in include_globs_raw if str(x).strip()]
    else:
        raise ValueError("Config inputs.include_globs must be a list of glob strings (or omitted).")

    inputs = InputsConfig(
        kind=inputs_kind,
        path=inputs_path,
        format=inputs_format,
        compression=inputs_compression,
        include_globs=include_globs,
        id_field=str(inputs_raw.get("id_field", "id")),
        text_field=str(inputs_raw.get("text_field", "text")),
        url_field=str(inputs_raw.get("url_field", "url")),
        topic_field=str(inputs_raw.get("topic_field", "topic")),
    )

    topics_val = raw.get("topics", None)
    if topics_val is None:
        topics: list[str] = []
    elif isinstance(topics_val, Sequence) and not isinstance(topics_val, (str, bytes)):
        topics = []
        for item in topics_val:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Config topics must contain non-empty strings.")
            topics.append(item)
    else:
        raise ValueError("Config topics must be a list of strings (or omitted).")

    use_topics = bool(topics)
    if not use_topics:
        topics = ["all"]

    llm_raw = _require_mapping(raw, "llm")
    llm = LLMConfig(
        provider=_require_str(llm_raw, "provider"),
        model=_require_str(llm_raw, "model"),
        max_requests_per_minute=int(llm_raw.get("max_requests_per_minute", 1_000)),
        max_tokens_per_minute=int(llm_raw.get("max_tokens_per_minute", 100_000)),
        temperature=float(llm_raw.get("temperature", 0.0)),
        max_new_tokens=int(llm_raw.get("max_new_tokens", 2048)),
    )

    # Per-stage overrides (preferred key: stage_llm_overrides; legacy alias: stages)
    stage_overrides_raw = None
    if isinstance(raw.get("stage_llm_overrides"), Mapping):
        stage_overrides_raw = raw.get("stage_llm_overrides")
    elif isinstance(raw.get("stages"), Mapping):
        stage_overrides_raw = raw.get("stages")

    stage_llm_overrides: dict[str, StageLLMConfig] = {}
    if stage_overrides_raw:
        for stage_name, v in stage_overrides_raw.items():
            if not isinstance(stage_name, str):
                raise ValueError("stage_llm_overrides keys must be strings.")
            if stage_name not in STAGE_NAMES:
                raise ValueError(
                    f"Unsupported stage override: {stage_name} (expected one of {list(STAGE_NAMES)})"
                )
            if not isinstance(v, Mapping):
                raise ValueError(f"stage_llm_overrides.{stage_name} must be a mapping.")
            backend = str(v.get("backend", "deluge")).strip().lower() or "deluge"
            vllm_block = v.get("vllm", None)
            if vllm_block is not None and not isinstance(vllm_block, Mapping):
                raise ValueError(f"stage_llm_overrides.{stage_name}.vllm must be a mapping (or omitted).")
            stage_llm_overrides[stage_name] = StageLLMConfig(
                backend=backend,
                provider=str(v["provider"]).strip() if "provider" in v and v.get("provider") is not None else None,
                model=str(v["model"]).strip() if "model" in v and v.get("model") is not None else None,
                temperature=float(v["temperature"]) if "temperature" in v and v.get("temperature") is not None else None,
                max_new_tokens=int(v["max_new_tokens"]) if "max_new_tokens" in v and v.get("max_new_tokens") is not None else None,
                max_requests_per_minute=int(v["max_requests_per_minute"])
                if "max_requests_per_minute" in v and v.get("max_requests_per_minute") is not None
                else None,
                max_tokens_per_minute=int(v["max_tokens_per_minute"])
                if "max_tokens_per_minute" in v and v.get("max_tokens_per_minute") is not None
                else None,
                vllm=dict(vllm_block) if isinstance(vllm_block, Mapping) else None,
            )

    prompts_raw = _require_mapping(raw, "prompts")
    def _p(key: str) -> Path:
        p = Path(_require_str(prompts_raw, key))
        return p if p.is_absolute() else (base_dir / p).resolve()

    prompts = PromptPaths(
        procedures=_p("procedures") if "procedures" in prompts_raw else _p("extract"),
        filter=_p("filter"),
        postprocess=_p("postprocess"),
        resources=_p("resources") if "resources" in prompts_raw else _p("tools"),
        final_filter=_p("final_filter"),
    )

    targets_raw = raw.get("targets") if isinstance(raw.get("targets"), Mapping) else {}

    candidates_per_topic = int(targets_raw.get("candidates_per_topic", 100))

    _dv_raw = targets_raw.get("desired_valid_per_topic")
    desired_valid_per_topic: int | None = int(_dv_raw) if _dv_raw is not None else None

    targets = Targets(
        candidates_per_topic=candidates_per_topic,
        desired_valid_per_topic=desired_valid_per_topic,
        extract_batch_size=int(targets_raw.get("extract_batch_size", 50)),
        min_steps=int(targets_raw.get("min_steps", 4)),
        max_steps=int(targets_raw.get("max_steps", 15)),
        train_per_topic=int(targets_raw.get("train_per_topic", 1)),
        test_per_topic=int(targets_raw.get("test_per_topic", 1)),
    )

    export_raw = raw.get("export") if isinstance(raw.get("export"), Mapping) else {}
    export_format = str(export_raw.get("format", "jsonl")).strip() or "jsonl"
    mfs = export_raw.get("max_file_size", None)
    export_max_file_size = None
    if mfs is None:
        export_max_file_size = None
    elif isinstance(mfs, str):
        export_max_file_size = mfs.strip() or None
    else:
        raise ValueError("export.max_file_size must be a string like '500MB' (or omitted).")
    export = ExportConfig(format=export_format, max_file_size=export_max_file_size)

    return How2MineConfig(
        out_root=out_root,
        inputs=inputs,
        topics=topics,
        use_topics=use_topics,
        llm=llm,
        stage_llm_overrides=stage_llm_overrides,
        prompts=prompts,
        targets=targets,
        export=export,
    )

