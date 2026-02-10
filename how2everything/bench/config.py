from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


STAGE_NAMES: tuple[str, ...] = ("generate", "judge")

# Default distilled judge (8B) on HuggingFace.
DEFAULT_HOW2JUDGE_MODEL = "how2everything/how2judge"
DEFAULT_HOW2BENCH_DATASET = "how2everything/how2bench"


@dataclass(frozen=True)
class InputsConfig:
    """
    Inputs for bench.

    - kind=auto: attempt to infer record type (how2mine export vs bench JSONL).
    - kind=how2mine_export: expects records with `final_procedure` and `source_example`.
    - kind=bench: expects records with `source_example_id`, `goal`, and `steps`.
    """

    # Local path input. If omitted, defaults to loading from `hf_repo`.
    path: Path | None = None
    # Hugging Face dataset repo id to read from when `path` is omitted.
    hf_repo: str | None = DEFAULT_HOW2BENCH_DATASET
    hf_split: str = "train"
    hf_streaming: bool = True
    # Optional HF datasets cache dir override (useful on clusters where ~/.cache is on NFS).
    hf_cache_dir: Path | None = None
    kind: str = "auto"  # auto|how2mine_export|bench
    format: str = "auto"  # auto|jsonl|csv|arrow|parquet
    compression: str = "auto"  # auto|none|zst|gz|bz2|xz
    include_globs: list[str] = field(default_factory=list)  # when path is a dir (non-recursive)


@dataclass(frozen=True)
class RoleLLMConfig:
    """
    Explicit LLM role configuration for how2bench.

    Roles:
    - generator: used for stage `generate`
    - evaluator: used for stage `judge` (optional; defaults to local how2judge)

    Notes:
    - If backend=deluge, `provider` is required (for env validation / metadata).
    - If backend=vllm, `provider` is informational only and can be omitted.
    """

    backend: str = "deluge"  # deluge|vllm
    provider: str | None = None  # required when backend=deluge
    model: str = ""
    temperature: float = 0.0
    max_new_tokens: int = 4096
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    reasoning_effort: str | None = None
    vllm: Mapping[str, Any] | None = None  # backend-specific overrides for vLLM


@dataclass(frozen=True)
class ResolvedStageLLM:
    backend: str  # deluge|vllm
    provider: str
    model: str
    temperature: float
    max_new_tokens: int
    max_requests_per_minute: int
    max_tokens_per_minute: int
    reasoning_effort: str | None
    vllm_overrides: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PromptPaths:
    inference_base: Path
    inference_inst: Path
    judge: Path


@dataclass(frozen=True)
class ArtifactPaths:
    """
    Optional overrides for how2bench artifact file locations.

    If omitted, how2bench defaults to writing/reading under `out_root`:
    - generations.jsonl
    - judgments/<model>_<judge_id>/judgments.jsonl
    """

    generations: Path | None = None
    judgments: Path | None = None


@dataclass(frozen=True)
class GenerationConfig:
    model_type: str = "inst"  # base|inst


@dataclass(frozen=True)
class BenchConfig:
    out_root: Path
    inputs: InputsConfig
    # generator is required for `h2e bench gen` and `h2e bench run`.
    generator: RoleLLMConfig | None
    # evaluator is optional; if omitted, judge defaults to local how2judge via vLLM.
    evaluator: RoleLLMConfig | None
    prompts: PromptPaths
    paths: ArtifactPaths
    generation: GenerationConfig

    def generations_path(self) -> Path:
        return self.paths.generations if self.paths.generations is not None else (self.out_root / "generations.jsonl")

    def judgments_path(self) -> Path:
        if self.paths.judgments is not None:
            return self.paths.judgments
        # Default: keep generation as the root, and store judgments under a judge-id subdir.
        return self.out_root / "judgments" / self.judge_dir_name() / "judgments.jsonl"

    def judge_manifest_path(self) -> Path:
        return self.judgments_path().parent / "judge_manifest.json"

    def judge_prompt_sha256(self) -> str:
        txt = self.prompts.judge.read_text(encoding="utf-8")
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()

    def generator_prompt_path(self) -> Path:
        """
        Resolved generation prompt path for the current `generation.model_type`.
        """
        return self.prompts.inference_base if self.generation.model_type == "base" else self.prompts.inference_inst

    def generator_prompt_sha256(self) -> str:
        txt = self.generator_prompt_path().read_text(encoding="utf-8")
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()

    def judge_dir_name(self) -> str:
        """
        Default judgments subdirectory name under `out_root/judgments/`.

        Format: <short_model>_<judge_id>

        short_model is derived from the resolved judge model name by splitting on
        '/' and '_' and taking the last segment, then sanitizing for filesystem
        safety.
        """
        st = self.resolve_stage_llm("judge")
        model = (st.model or "").strip()
        parts = [p for p in re.split(r"[/_]+", model) if p]
        short = parts[-1] if parts else "judge"
        # Sanitize: keep alnum, dot, dash; map others to '-'.
        safe = "".join(ch if (ch.isalnum() or ch in ".-") else "-" for ch in short).strip("-")
        if not safe:
            safe = "judge"
        return f"{safe}_{self.judge_id()}"

    def judge_id(self) -> str:
        """
        Stable short hash identifying the judge run.

        Includes:
        - resolved judge config (model + decoding settings + vLLM overrides)
        - SHA-256 of the judge prompt contents
        """
        st = self.resolve_stage_llm("judge")
        payload = _canonical_json(
            {
                "judge_id_version": "how2bench.judge_id.v1",
                "judge": {
                    "backend": st.backend,
                    "provider": st.provider,
                    "model": st.model,
                    "temperature": st.temperature,
                    "max_new_tokens": st.max_new_tokens,
                    "reasoning_effort": st.reasoning_effort,
                    "vllm_overrides": st.vllm_overrides or {},
                },
                "judge_prompt_sha256": self.judge_prompt_sha256(),
                "judge_prompt_path": str(self.prompts.judge),
            }
        )
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def generator_id(self) -> str:
        """
        Stable short hash identifying the generator configuration used for `generate`.

        Includes:
        - resolved generator config (model + decoding settings + vLLM overrides)
        - generation prompt contents SHA-256 (based on `generation.model_type`)
        - generation model_type (base vs inst)
        """
        st = self.resolve_stage_llm("generate")
        payload = _canonical_json(
            {
                "generator_id_version": "how2bench.generator_id.v1",
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
                "generation": {
                    "model_type": self.generation.model_type,
                },
                "generation_prompt_sha256": self.generator_prompt_sha256(),
            }
        )
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def resolve_stage_llm(self, stage: str) -> ResolvedStageLLM:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage: {stage} (expected one of {list(STAGE_NAMES)})")
        role = self.generator if stage == "generate" else self.evaluator
        # Default evaluator to the distilled 8B how2judge model unless explicitly set.
        if stage == "judge" and role is None:
            return ResolvedStageLLM(
                backend="vllm",
                provider="local",
                model=DEFAULT_HOW2JUDGE_MODEL,
                temperature=0.0,
                max_new_tokens=4096,
                max_requests_per_minute=1_000,
                max_tokens_per_minute=100_000,
                reasoning_effort=None,
                # Default: disable any template-driven "thinking" behavior.
                # This is a no-op unless the configured chat template consults it.
                vllm_overrides={"chat_template_kwargs": {"enable_thinking": False}},
            )

        if role is None:
            raise ValueError(f"Missing required role config for stage: {stage}")

        backend = (role.backend or "deluge").strip().lower()
        if backend not in {"deluge", "vllm"}:
            raise ValueError(f"Unsupported backend for stage {stage}: {backend}")

        model = (role.model or "").strip()
        if not model:
            raise ValueError(f"{stage} model must be a non-empty string.")

        provider = (role.provider or "").strip()
        if backend == "deluge" and not provider:
            raise ValueError(f"{stage} provider is required when backend=deluge.")
        if backend == "vllm" and not provider:
            provider = "local"

        return ResolvedStageLLM(
            backend=backend,
            provider=provider,
            model=model,
            temperature=float(role.temperature),
            max_new_tokens=int(role.max_new_tokens),
            max_requests_per_minute=int(role.max_requests_per_minute),
            max_tokens_per_minute=int(role.max_tokens_per_minute),
            reasoning_effort=role.reasoning_effort,
            vllm_overrides=role.vllm if isinstance(role.vllm, Mapping) else None,
        )


# ---------------------------------------------------------------------------
# ModelSpec: per-model entry in the new unified `models:` config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One entry from `models:` in the suite config."""

    model: str
    run_name: str
    prompt_style: str = "inst"  # base|inst
    backend: str = "deluge"
    provider: str | None = None
    temperature: float = 0.0
    max_new_tokens: int = 4096
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    reasoning_effort: str | None = None
    vllm: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SuiteConfig:
    """
    Unified config for how2bench.

    Replaces both the old single-model ``load_config()`` and
    ``OfficialBenchmarkConfig``.
    """

    out_root: Path
    models: list[ModelSpec]
    evaluator: RoleLLMConfig | None
    inputs: InputsConfig
    prompts: PromptPaths
    paths: ArtifactPaths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Config must be a YAML mapping (dict) at top-level.")
    return data


def _canonical_json(obj: Any) -> Any:
    """Canonicalize nested mappings/lists for stable hashing."""
    if isinstance(obj, dict):
        return {str(k): _canonical_json(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}
    if isinstance(obj, list):
        return [_canonical_json(x) for x in obj]
    return obj


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    v = raw.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Config must set non-empty string: {key}")
    return v


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    v = raw.get(key)
    if v is None:
        return None
    if not isinstance(v, Mapping):
        raise ValueError(f"Config key {key} must be a mapping when provided.")
    return v


def _parse_role_llm_block(block: Mapping[str, Any], *, key_name: str) -> RoleLLMConfig:
    backend = str(block.get("backend", "deluge")).strip().lower() or "deluge"
    vllm_block = block.get("vllm", None)
    if vllm_block is not None and not isinstance(vllm_block, Mapping):
        raise ValueError(f"{key_name}.vllm must be a mapping (or omitted).")
    provider = block.get("provider", None)
    model = block.get("model", None)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{key_name}.model must be a non-empty string.")
    provider_str = str(provider).strip() if isinstance(provider, str) else None
    if backend == "deluge" and not provider_str:
        raise ValueError(f"{key_name}.provider is required when backend=deluge.")
    return RoleLLMConfig(
        backend=backend,
        provider=provider_str,
        model=model.strip(),
        temperature=float(block.get("temperature", 0.0)),
        max_new_tokens=int(block.get("max_new_tokens", 4096)),
        max_requests_per_minute=int(block.get("max_requests_per_minute", 1_000)),
        max_tokens_per_minute=int(block.get("max_tokens_per_minute", 100_000)),
        reasoning_effort=(str(block["reasoning_effort"]).strip() if block.get("reasoning_effort") is not None else None),
        vllm=(dict(vllm_block) if isinstance(vllm_block, Mapping) else None),
    )


def _safe_dirname(s: str) -> str:
    """Filesystem-safe directory name from a model string."""
    out = []
    for ch in s.strip():
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        elif ch in {"/", "\\", " "}:
            out.append("-")
        else:
            out.append("-")
    name = "".join(out).strip("-")
    return name or "model"


def _deep_merge_vllm(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """One-level-deep merge for vllm blocks (engine_kwargs, chat_template_kwargs, etc.)."""
    if base is None and override is None:
        return None
    if base is None:
        return dict(override) if override else None
    if override is None:
        return dict(base)
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), Mapping):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


def _parse_model_entry(
    entry: str | Mapping[str, Any],
    defaults: Mapping[str, Any],
    index: int,
) -> ModelSpec:
    """Parse a single item from ``models:`` into a ``ModelSpec``."""
    if isinstance(entry, str):
        model = entry.strip()
        if not model:
            raise ValueError(f"models[{index}] must be a non-empty string.")
        prompt_style_raw = defaults.get("prompt_style")
        if not prompt_style_raw or str(prompt_style_raw).strip().lower() not in {"base", "inst"}:
            raise ValueError(
                f"models[{index}] ('{model}'): prompt_style is required. "
                "Set it on the model entry or in generator_defaults (must be 'base' or 'inst')."
            )
        return ModelSpec(
            model=model,
            run_name=_safe_dirname(model),
            prompt_style=str(prompt_style_raw).strip().lower(),
            backend=str(defaults.get("backend", "deluge")).strip().lower() or "deluge",
            provider=defaults.get("provider"),
            temperature=float(defaults.get("temperature", 0.0)),
            max_new_tokens=int(defaults.get("max_new_tokens", 4096)),
            max_requests_per_minute=int(defaults.get("max_requests_per_minute", 1_000)),
            max_tokens_per_minute=int(defaults.get("max_tokens_per_minute", 100_000)),
            reasoning_effort=defaults.get("reasoning_effort"),
            vllm=(dict(defaults["vllm"]) if isinstance(defaults.get("vllm"), Mapping) else None),
        )

    if not isinstance(entry, Mapping):
        raise ValueError(f"models[{index}] must be a string or mapping, got: {type(entry).__name__}")

    model = entry.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"models[{index}].model must be a non-empty string.")
    model = model.strip()

    run_name_raw = entry.get("run_name")
    if run_name_raw is None:
        run_name = _safe_dirname(model)
    elif isinstance(run_name_raw, str) and run_name_raw.strip():
        run_name = _safe_dirname(run_name_raw)
    else:
        raise ValueError(f"models[{index}].run_name must be a non-empty string if provided.")

    prompt_style_raw = entry.get("prompt_style", defaults.get("prompt_style"))
    if not prompt_style_raw:
        raise ValueError(
            f"models[{index}] ('{model}'): prompt_style is required. "
            "Set it on the model entry or in generator_defaults (must be 'base' or 'inst')."
        )
    prompt_style = str(prompt_style_raw).strip().lower()
    if prompt_style not in {"base", "inst"}:
        raise ValueError(f"models[{index}].prompt_style must be 'base' or 'inst', got: {prompt_style!r}")

    backend = str(entry.get("backend", defaults.get("backend", "deluge"))).strip().lower() or "deluge"
    if backend not in {"deluge", "vllm"}:
        raise ValueError(f"models[{index}].backend must be 'deluge' or 'vllm', got: {backend!r}")

    provider_raw = entry.get("provider", defaults.get("provider"))
    provider = str(provider_raw).strip() if isinstance(provider_raw, str) and provider_raw.strip() else None
    if backend == "deluge" and not provider:
        raise ValueError(f"models[{index}] backend=deluge requires provider.")

    vllm_entry = entry.get("vllm")
    if vllm_entry is not None and not isinstance(vllm_entry, Mapping):
        raise ValueError(f"models[{index}].vllm must be a mapping if provided.")
    vllm_defaults = defaults.get("vllm")
    if vllm_defaults is not None and not isinstance(vllm_defaults, Mapping):
        vllm_defaults = None
    vllm_merged = _deep_merge_vllm(vllm_defaults, vllm_entry)

    reasoning_raw = entry.get("reasoning_effort", defaults.get("reasoning_effort"))
    reasoning = str(reasoning_raw).strip() if reasoning_raw is not None else None

    return ModelSpec(
        model=model,
        run_name=run_name,
        prompt_style=prompt_style,
        backend=backend,
        provider=provider,
        temperature=float(entry.get("temperature", defaults.get("temperature", 0.0))),
        max_new_tokens=int(entry.get("max_new_tokens", defaults.get("max_new_tokens", 4096))),
        max_requests_per_minute=int(entry.get("max_requests_per_minute", defaults.get("max_requests_per_minute", 1_000))),
        max_tokens_per_minute=int(entry.get("max_tokens_per_minute", defaults.get("max_tokens_per_minute", 100_000))),
        reasoning_effort=reasoning,
        vllm=vllm_merged,
    )


# ---------------------------------------------------------------------------
# Shared parsing helpers for inputs / prompts / paths
# ---------------------------------------------------------------------------


def _parse_inputs(raw: Mapping[str, Any], base_dir: Path) -> InputsConfig:
    inputs_raw_any = raw.get("inputs")
    if inputs_raw_any is None:
        inputs_raw: Mapping[str, Any] = {}
    elif isinstance(inputs_raw_any, Mapping):
        inputs_raw = inputs_raw_any
    else:
        raise ValueError("inputs must be a mapping when provided.")

    inputs_path: Path | None = None
    if inputs_raw.get("path") is not None:
        if not isinstance(inputs_raw.get("path"), str) or not str(inputs_raw.get("path")).strip():
            raise ValueError("inputs.path must be a non-empty string when provided.")
        inputs_path = Path(str(inputs_raw.get("path")).strip())
        if not inputs_path.is_absolute():
            inputs_path = (base_dir / inputs_path).resolve()

    hf_repo_val = inputs_raw.get("hf_repo", DEFAULT_HOW2BENCH_DATASET)
    hf_repo = None
    if inputs_path is None:
        if hf_repo_val is None:
            hf_repo = DEFAULT_HOW2BENCH_DATASET
        elif isinstance(hf_repo_val, str) and hf_repo_val.strip():
            hf_repo = hf_repo_val.strip()
        else:
            raise ValueError("inputs.hf_repo must be a non-empty string (or omitted).")

    hf_split = str(inputs_raw.get("hf_split", "train")).strip() or "train"
    hf_streaming = bool(inputs_raw.get("hf_streaming", True))
    hf_cache_dir_raw = inputs_raw.get("hf_cache_dir", None)
    hf_cache_dir: Path | None = None
    if hf_cache_dir_raw is not None:
        if not isinstance(hf_cache_dir_raw, str) or not hf_cache_dir_raw.strip():
            raise ValueError("inputs.hf_cache_dir must be a non-empty string when provided.")
        p = Path(hf_cache_dir_raw.strip())
        hf_cache_dir = p if p.is_absolute() else (base_dir / p).resolve()

    inputs_kind = str(inputs_raw.get("kind", "auto")).strip().lower() or "auto"
    if inputs_kind not in {"auto", "how2mine_export", "bench"}:
        raise ValueError("inputs.kind must be one of: auto|how2mine_export|bench")
    inputs_format = str(inputs_raw.get("format", "auto")).strip() or "auto"
    inputs_compression = str(inputs_raw.get("compression", "auto")).strip() or "auto"

    include_globs_raw = inputs_raw.get("include_globs", [])
    if include_globs_raw is None:
        include_globs: list[str] = []
    elif isinstance(include_globs_raw, Sequence) and not isinstance(include_globs_raw, (str, bytes)):
        include_globs = [str(x).strip() for x in include_globs_raw if str(x).strip()]
    else:
        raise ValueError("inputs.include_globs must be a list[str] (or omitted).")

    return InputsConfig(
        path=inputs_path,
        hf_repo=hf_repo,
        hf_split=hf_split,
        hf_streaming=hf_streaming,
        hf_cache_dir=hf_cache_dir,
        kind=inputs_kind,
        format=inputs_format,
        compression=inputs_compression,
        include_globs=include_globs,
    )


def _parse_prompts(raw: Mapping[str, Any], base_dir: Path) -> PromptPaths:
    prompts_raw = raw.get("prompts") if isinstance(raw.get("prompts"), Mapping) else {}

    def _p(key: str, default_rel: str) -> Path:
        raw_val = prompts_raw.get(key, default_rel)
        p = Path(str(raw_val))
        return p if p.is_absolute() else (base_dir / p).resolve()

    return PromptPaths(
        inference_base=_p("inference_base", "prompts/inference_base.txt"),
        inference_inst=_p("inference_inst", "prompts/inference_inst.txt"),
        judge=_p("judge", "prompts/judge.txt"),
    )


def _parse_paths(raw: Mapping[str, Any], base_dir: Path) -> ArtifactPaths:
    paths_raw = raw.get("paths")
    if paths_raw is None:
        return ArtifactPaths()
    if not isinstance(paths_raw, Mapping):
        raise ValueError("paths must be a mapping when provided.")

    def _path(val: Any) -> Path | None:
        if val is None:
            return None
        if not isinstance(val, str) or not val.strip():
            raise ValueError("paths entries must be non-empty strings when provided.")
        p = Path(val.strip())
        return p if p.is_absolute() else (base_dir / p).resolve()

    return ArtifactPaths(
        generations=_path(paths_raw.get("generations")),
        judgments=_path(paths_raw.get("judgments")),
    )


# ---------------------------------------------------------------------------
# Unified loader: load_suite_config()
# ---------------------------------------------------------------------------

_LEGACY_KEYS = {"generator", "llm", "generation", "generations_root", "stage_llm_overrides"}


def load_suite_config(config_path: str) -> SuiteConfig:
    """
    Parse the unified how2bench YAML config into a ``SuiteConfig``.

    Accepts the new ``models:`` format. Rejects legacy keys (``generator:``,
    ``llm:``, ``generation:``, ``generations_root:``) with a clear migration
    message.
    """
    cfg_path = Path(config_path)
    raw = _load_yaml(cfg_path)
    base_dir = Path.cwd()

    # Reject legacy keys with helpful error.
    for legacy in _LEGACY_KEYS:
        if legacy in raw:
            raise ValueError(
                f"Legacy key `{legacy}:` is no longer supported.\n"
                "Migrate to the new config format:\n"
                "  - Replace `generator:` / `generation:` with `models:` list + optional `generator_defaults:`\n"
                "  - Replace `generations_root:` with `out_root:`\n"
                "See examples/bench/README.md for the new config format."
            )

    # out_root (required)
    out_root = Path(_require_str(raw, "out_root"))
    if not out_root.is_absolute():
        out_root = (base_dir / out_root).resolve()

    # generator_defaults (optional shared defaults for models)
    gen_defaults_raw = raw.get("generator_defaults")
    if gen_defaults_raw is not None and not isinstance(gen_defaults_raw, Mapping):
        raise ValueError("generator_defaults must be a mapping when provided.")
    gen_defaults: Mapping[str, Any] = gen_defaults_raw if isinstance(gen_defaults_raw, Mapping) else {}

    # models (optional; empty list for judge-only mode)
    models_raw = raw.get("models")
    models: list[ModelSpec] = []
    if models_raw is not None:
        if not isinstance(models_raw, Sequence) or isinstance(models_raw, (str, bytes)):
            raise ValueError("models must be a list when provided.")
        for i, entry in enumerate(models_raw):
            models.append(_parse_model_entry(entry, gen_defaults, i))

    # evaluator (optional)
    evaluator_raw = _optional_mapping(raw, "evaluator")
    evaluator = _parse_role_llm_block(evaluator_raw, key_name="evaluator") if evaluator_raw is not None else None

    inputs = _parse_inputs(raw, base_dir)
    prompts = _parse_prompts(raw, base_dir)
    paths = _parse_paths(raw, base_dir)

    return SuiteConfig(
        out_root=out_root,
        models=models,
        evaluator=evaluator,
        inputs=inputs,
        prompts=prompts,
        paths=paths,
    )


# ---------------------------------------------------------------------------
# SuiteConfig -> BenchConfig converters
# ---------------------------------------------------------------------------


def model_spec_to_bench_config(suite: SuiteConfig, spec: ModelSpec) -> BenchConfig:
    """Build a per-model ``BenchConfig`` from a ``SuiteConfig`` + ``ModelSpec``."""
    generator = RoleLLMConfig(
        backend=spec.backend,
        provider=spec.provider,
        model=spec.model,
        temperature=spec.temperature,
        max_new_tokens=spec.max_new_tokens,
        max_requests_per_minute=spec.max_requests_per_minute,
        max_tokens_per_minute=spec.max_tokens_per_minute,
        reasoning_effort=spec.reasoning_effort,
        vllm=(dict(spec.vllm) if isinstance(spec.vllm, Mapping) else None),
    )
    generation = GenerationConfig(model_type=spec.prompt_style)

    # Build a temporary config to compute generator_id (does not depend on final out_root).
    tmp = BenchConfig(
        out_root=suite.out_root / spec.run_name,
        inputs=suite.inputs,
        generator=generator,
        evaluator=suite.evaluator,
        prompts=suite.prompts,
        paths=suite.paths,
        generation=generation,
    )
    generator_id = tmp.generator_id()
    out_root = suite.out_root / f"{spec.run_name}_{generator_id}"

    return BenchConfig(
        out_root=out_root,
        inputs=suite.inputs,
        generator=generator,
        evaluator=suite.evaluator,
        prompts=suite.prompts,
        paths=suite.paths,
        generation=generation,
    )


def suite_to_judge_config(suite: SuiteConfig) -> BenchConfig:
    """Build a ``BenchConfig`` for judge-only mode (no generator)."""
    return BenchConfig(
        out_root=suite.out_root,
        inputs=suite.inputs,
        generator=None,
        evaluator=suite.evaluator,
        prompts=suite.prompts,
        paths=suite.paths,
        generation=GenerationConfig(),
    )
