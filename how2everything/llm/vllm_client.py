from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Type, cast

from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn


@dataclass(frozen=True)
class VLLMConfig:
    model: str
    temperature: float = 0.0
    max_new_tokens: int = 2048
    # "generate" uses `LLM.generate(prompts=...)`.
    # "chat" uses `LLM.chat(messages=...)` with optional template overrides.
    mode: Literal["generate", "chat"] = "generate"

    # Optional env var to set before importing/initializing vLLM.
    # Common value: "spawn" for multi-GPU stability in some setups.
    worker_multiproc_method: str | None = None

    # HuggingFace token env var name to read and pass to vLLM.
    # If None/empty, no token is passed.
    hf_token_env: str | None = "HF_TOKEN"

    # vLLM engine init kwargs (passed to `vllm.LLM(**engine_kwargs)`).
    # Example keys: tensor_parallel_size, dtype, revision, trust_remote_code, max_model_len, gpu_memory_utilization, ...
    engine_kwargs: dict[str, Any] = None  # type: ignore[assignment]

    # vLLM SamplingParams kwargs (in addition to temperature/max_tokens).
    # Example keys: top_p, top_k, min_p, repetition_penalty, presence_penalty, frequency_penalty, seed, stop, ...
    sampling_kwargs: dict[str, Any] = None  # type: ignore[assignment]

    # Chat templating controls (only when mode="chat").
    chat_template: str | None = None
    chat_template_path: str | None = None
    chat_template_kwargs: dict[str, Any] = None  # type: ignore[assignment]
    add_generation_prompt: bool = True

    # If output_schema is provided, require vLLM guided decoding support and use the JSON schema.
    # This keeps the pipeline strict and avoids brittle prompt-only JSON.
    require_guided_json: bool = True

    @classmethod
    def from_overrides(
        cls,
        *,
        model: str,
        temperature: float,
        max_new_tokens: int,
        overrides: Mapping[str, Any] | None,
        base_dir: Path | None = None,
    ) -> "VLLMConfig":
        """
        Build a VLLMConfig from a nested mapping under `stage_llm_overrides.<stage>.vllm`.

        Supported top-level keys:
        - mode: "generate" | "chat"
        - worker_multiproc_method: str
        - hf_token_env: str | null
        - engine_kwargs: { ... }
        - sampling_kwargs: { ... }
        - chat_template: str
        - chat_template_path: str (resolved relative to base_dir if provided)
        - chat_template_kwargs: { ... }
        - add_generation_prompt: bool
        - require_guided_json: bool
        """
        o = dict(overrides) if overrides else {}

        def _pop_mapping(key: str) -> dict[str, Any]:
            v = o.pop(key, None)
            if v is None:
                return {}
            if not isinstance(v, Mapping):
                raise ValueError(f"vllm.{key} must be a mapping, got: {type(v).__name__}")
            return dict(v)

        mode = str(o.pop("mode", "generate")).strip().lower()
        if mode not in {"generate", "chat"}:
            raise ValueError(f"vllm.mode must be 'generate' or 'chat', got: {mode!r}")

        worker_multiproc_method = o.pop("worker_multiproc_method", None)
        if worker_multiproc_method is not None and not isinstance(worker_multiproc_method, str):
            raise ValueError("vllm.worker_multiproc_method must be a string or null.")
        worker_multiproc_method = worker_multiproc_method.strip() if isinstance(worker_multiproc_method, str) else None

        hf_token_env = o.pop("hf_token_env", "HF_TOKEN")
        if hf_token_env is not None and not isinstance(hf_token_env, str):
            raise ValueError("vllm.hf_token_env must be a string or null.")
        hf_token_env = hf_token_env.strip() if isinstance(hf_token_env, str) else None

        engine_kwargs = _pop_mapping("engine_kwargs")
        sampling_kwargs = _pop_mapping("sampling_kwargs")
        chat_template_kwargs = _pop_mapping("chat_template_kwargs")

        chat_template = o.pop("chat_template", None)
        if chat_template is not None and not isinstance(chat_template, str):
            raise ValueError("vllm.chat_template must be a string or null.")

        chat_template_path = o.pop("chat_template_path", None)
        if chat_template_path is not None and not isinstance(chat_template_path, str):
            raise ValueError("vllm.chat_template_path must be a string or null.")
        if isinstance(chat_template_path, str):
            p = Path(chat_template_path)
            if base_dir is not None and not p.is_absolute():
                p = (base_dir / p).resolve()
            chat_template_path = str(p)

        add_generation_prompt = o.pop("add_generation_prompt", True)
        if not isinstance(add_generation_prompt, bool):
            raise ValueError("vllm.add_generation_prompt must be a bool.")

        require_guided_json = o.pop("require_guided_json", True)
        if not isinstance(require_guided_json, bool):
            raise ValueError("vllm.require_guided_json must be a bool.")

        if o:
            raise ValueError(f"Unknown vllm override keys: {sorted(o.keys())}")

        return cls(
            model=model,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            mode=cast(Literal["generate", "chat"], mode),
            worker_multiproc_method=worker_multiproc_method,
            hf_token_env=hf_token_env,
            engine_kwargs=engine_kwargs,
            sampling_kwargs=sampling_kwargs,
            chat_template=chat_template.strip() if isinstance(chat_template, str) else None,
            chat_template_path=chat_template_path.strip() if isinstance(chat_template_path, str) else None,
            chat_template_kwargs=chat_template_kwargs,
            add_generation_prompt=add_generation_prompt,
            require_guided_json=require_guided_json,
        )


def _try_parse_json(text: str) -> Optional[dict[str, Any] | list[Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


class VLLMClient:
    """
    Minimal vLLM-backed client for how2mine stages.

    API compatibility targets `DelugeClient.complete_json()`:
    - `complete_json(prompts, output_schema=..., ...) -> list[dict]`
    - populates `last_usage` and `total_usage` dicts (cost=0.0 for local models)
    """

    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.last_usage: dict[str, float | int] = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self.total_usage: dict[str, float | int] = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        if cfg.worker_multiproc_method:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = cfg.worker_multiproc_method

        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            raise ImportError(
                "vLLM backend requested but vllm is not installed. "
                "Install optional deps (e.g. `pip install -e '.[vllm]'`)."
            ) from e

        llm_kwargs: dict[str, Any] = {"model": cfg.model}
        if cfg.hf_token_env:
            tok = os.environ.get(cfg.hf_token_env)
            if tok:
                llm_kwargs["hf_token"] = tok
        if cfg.engine_kwargs:
            llm_kwargs.update(cfg.engine_kwargs)

        # Auto-infer tensor_parallel_size from visible GPUs if not explicitly set.
        if "tensor_parallel_size" not in llm_kwargs:
            try:
                import torch
                n_gpus = torch.cuda.device_count()
                if n_gpus > 1:
                    llm_kwargs["tensor_parallel_size"] = n_gpus
            except Exception:
                pass

        self._llm = LLM(**llm_kwargs)

        # Load optional chat template once.
        self._chat_template: str | None = cfg.chat_template
        if cfg.chat_template_path:
            p = Path(cfg.chat_template_path)
            if not p.exists():
                raise FileNotFoundError(f"vllm.chat_template_path not found: {p}")
            self._chat_template = p.read_text(encoding="utf-8")

    def _guided_decoding_params(self, output_schema: Type[BaseModel] | dict[str, Any]) -> Any:
        try:
            from vllm.sampling_params import GuidedDecodingParams  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "This vLLM installation does not expose GuidedDecodingParams, "
                "but require_guided_json=true and output_schema was provided."
            ) from e

        if isinstance(output_schema, dict):
            schema = output_schema
        elif isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            schema = output_schema.model_json_schema()
        else:
            raise TypeError(f"Unsupported output_schema type for vLLM guided decoding: {type(output_schema)}")

        # vLLM expects a JSON schema dict for guided decoding.
        return GuidedDecodingParams(json=schema)

    def _sampling_params(self, *, output_schema: Type[BaseModel] | dict[str, Any] | None) -> Any:
        from vllm import SamplingParams  # type: ignore

        kwargs: dict[str, Any] = {
            "temperature": float(self.cfg.temperature),
            "max_tokens": int(self.cfg.max_new_tokens),
        }
        if self.cfg.sampling_kwargs:
            kwargs.update(self.cfg.sampling_kwargs)

        if output_schema is not None:
            if not self.cfg.require_guided_json:
                raise ValueError(
                    "output_schema was provided but vllm.require_guided_json=false. "
                    "Refusing to run schema-constrained stage without guided decoding."
                )
            kwargs.pop("stop", None)  # stop sequences are unnecessary/harmful for JSON-guided decoding
            kwargs["guided_decoding"] = self._guided_decoding_params(output_schema)

        return SamplingParams(**kwargs)

    def _generate_texts(
        self,
        prompts: Sequence[str],
        *,
        output_schema: Type[BaseModel] | dict[str, Any] | None,
    ) -> list[str]:
        sp = self._sampling_params(output_schema=output_schema)

        if self.cfg.mode == "chat":
            chat = getattr(self._llm, "chat", None)
            if not callable(chat):
                raise RuntimeError("vllm.mode=chat but this vLLM LLM instance does not support .chat().")
            chat_prompts = [[{"role": "user", "content": p}] for p in prompts]
            chat_kwargs: dict[str, Any] = {
                "add_generation_prompt": bool(self.cfg.add_generation_prompt),
            }
            if self.cfg.chat_template_kwargs:
                chat_kwargs["chat_template_kwargs"] = dict(self.cfg.chat_template_kwargs)
            if self._chat_template:
                chat_kwargs["chat_template"] = self._chat_template
            outs = chat(chat_prompts, sp, **chat_kwargs)
        else:
            outs = self._llm.generate(list(prompts), sampling_params=sp)

        texts: list[str] = []
        # vLLM returns outputs aligned with the input prompts.
        for o in outs:
            if not getattr(o, "outputs", None):
                texts.append("")
                continue
            texts.append(o.outputs[0].text or "")
        return texts

    def complete(
        self,
        prompts: Sequence[str],
        *,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> list[str]:
        """
        Text completion (no structured output).

        This mirrors `DelugeClient.complete()` so higher-level pipelines can
        use the same inference prompts for both deluge and vLLM backends.
        """
        prompts_list = list(prompts)
        if show_progress and prompts_list:
            # vLLM runs in a batch; show a deterministic bar that completes after generation.
            texts = self._generate_texts(prompts_list, output_schema=None)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(progress_desc or "vLLM", total=len(prompts_list))
                progress.update(task, completed=len(prompts_list))
        else:
            texts = self._generate_texts(prompts_list, output_schema=None)

        # Usage tracking: cost is 0; token counts not tracked here.
        usage = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self.last_usage = usage
        self.total_usage["cost_usd"] = float(self.total_usage["cost_usd"]) + float(usage.get("cost_usd", 0.0))
        self.total_usage["input_tokens"] = int(self.total_usage["input_tokens"]) + int(usage.get("input_tokens", 0))
        self.total_usage["output_tokens"] = int(self.total_usage["output_tokens"]) + int(usage.get("output_tokens", 0))
        self.total_usage["cache_read_tokens"] = int(self.total_usage["cache_read_tokens"]) + int(
            usage.get("cache_read_tokens", 0)
        )
        self.total_usage["cache_write_tokens"] = int(self.total_usage["cache_write_tokens"]) + int(
            usage.get("cache_write_tokens", 0)
        )
        return texts

    def complete_json(
        self,
        prompts: Sequence[str],
        *,
        output_schema: Type[BaseModel] | dict[str, Any] | None = None,
        max_parse_retries: int = 2,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> list[dict[str, Any]]:
        prompts_list = list(prompts)

        call_prompts = prompts_list

        # Best-effort progress bar (optional).
        if show_progress and call_prompts:
            # Generate in one batch but still show a deterministic bar.
            completions = self._generate_texts(call_prompts, output_schema=output_schema)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(progress_desc or "vLLM", total=len(call_prompts))
                progress.update(task, completed=len(call_prompts))
        else:
            completions = self._generate_texts(call_prompts, output_schema=output_schema)

        def _parse_dicts(texts: Sequence[str]) -> tuple[list[Optional[dict[str, Any]]], list[int]]:
            parsed_local: list[Optional[dict[str, Any]]] = []
            bad: list[int] = []
            for i, txt in enumerate(texts):
                obj = _try_parse_json(txt)
                if isinstance(obj, dict):
                    parsed_local.append(obj)
                else:
                    parsed_local.append(None)
                    bad.append(i)
            return parsed_local, bad

        parsed, bad_idxs = _parse_dicts(completions)

        attempt = 0
        while bad_idxs and attempt < max_parse_retries:
            attempt += 1
            retry_prompts = [call_prompts[i] for i in bad_idxs]
            retry_completions = self._generate_texts(retry_prompts, output_schema=output_schema)
            new_bad: list[int] = []
            for local_j, txt in enumerate(retry_completions):
                global_i = bad_idxs[local_j]
                obj = _try_parse_json(txt)
                if isinstance(obj, dict):
                    parsed[global_i] = obj
                else:
                    new_bad.append(global_i)
            bad_idxs = new_bad

        if bad_idxs:
            logging.getLogger("how2bench").warning(
                "vLLM returned unparseable output for %d example(s) after %d retries. "
                "These will be recorded as empty dicts (parse failures).",
                len(bad_idxs),
                attempt,
            )

        out = [p if p is not None else {} for p in parsed]

        # Optional validation against pydantic schema (best-effort).
        # Skip empty dicts — those are parse failures already logged above.
        if output_schema is not None and isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            validated: list[dict[str, Any]] = []
            for d in out:
                if not d:
                    validated.append(d)
                    continue
                try:
                    validated.append(output_schema(**d).model_dump())  # type: ignore[arg-type]
                except Exception as e:
                    raise ValueError(f"vLLM output failed Pydantic validation for schema {output_schema.__name__}.") from e
            out = validated

        # Usage tracking: cost is 0; token counts not tracked here.
        usage = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self.last_usage = usage
        self.total_usage["cost_usd"] = float(self.total_usage["cost_usd"]) + float(usage.get("cost_usd", 0.0))
        self.total_usage["input_tokens"] = int(self.total_usage["input_tokens"]) + int(usage.get("input_tokens", 0))
        self.total_usage["output_tokens"] = int(self.total_usage["output_tokens"]) + int(usage.get("output_tokens", 0))
        self.total_usage["cache_read_tokens"] = int(self.total_usage["cache_read_tokens"]) + int(
            usage.get("cache_read_tokens", 0)
        )
        self.total_usage["cache_write_tokens"] = int(self.total_usage["cache_write_tokens"]) + int(
            usage.get("cache_write_tokens", 0)
        )
        return out

