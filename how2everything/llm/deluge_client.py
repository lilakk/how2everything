from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Type

from pydantic import BaseModel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn


@dataclass(frozen=True)
class DelugeLLMConfig:
    model: str
    max_requests_per_minute: int = 1_000
    max_tokens_per_minute: int = 100_000
    max_concurrent_requests: int = 225
    temperature: float = 0.0
    max_new_tokens: int = 2_048
    json_mode: bool = True
    reasoning_effort: Optional[str] = None


def _chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _try_parse_json(text: str) -> Optional[dict[str, Any] | list[Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


class DelugeClient:
    """
    Minimal wrapper around `lm_deluge.LLMClient` for the how2mine pipeline.

    Key behaviors:
    - Prefer schema-enforced structured outputs for OpenAI/Anthropic via `output_schema=...`
    - Otherwise use JSON mode and validate/parse locally.
    """

    def __init__(self, cfg: DelugeLLMConfig):
        self.cfg = cfg
        # Populated after each request batch (including retries).
        # Keys: cost_usd, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
        self.last_usage: dict[str, float | int] = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        # Cumulative usage across all calls for this client instance.
        self.total_usage: dict[str, float | int] = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        # Import lazily so packaging works without lm-deluge until runtime.
        from lm_deluge import LLMClient, SamplingParams

        # Silence lm-deluge's per-call usage printouts by default (it uses `print()`
        # in its tracker). This keeps terminal output readable; how2mine prints its
        # own stage summaries.
        #
        # To re-enable, set: H2E_LM_DELUGE_PRINT_USAGE=1
        if os.environ.get("H2E_LM_DELUGE_PRINT_USAGE", "").strip() != "1":
            try:
                from lm_deluge.tracker import StatusTracker

                if not getattr(StatusTracker, "_h2e_usage_silenced", False):
                    def _no_usage(self) -> None:  # type: ignore[no-redef]
                        return None

                    StatusTracker.log_usage = _no_usage  # type: ignore[assignment]
                    StatusTracker._h2e_usage_silenced = True  # type: ignore[attr-defined]
            except Exception:
                # Best-effort: if lm-deluge internals change, don't fail runs.
                pass

        sp = SamplingParams(
            temperature=cfg.temperature,
            json_mode=cfg.json_mode,
            max_new_tokens=cfg.max_new_tokens,
            reasoning_effort=cfg.reasoning_effort,
        )
        self._client = LLMClient(
            cfg.model,
            max_requests_per_minute=cfg.max_requests_per_minute,
            max_tokens_per_minute=cfg.max_tokens_per_minute,
            max_concurrent_requests=cfg.max_concurrent_requests,
            sampling_params=[sp],
            progress="manual",  # we drive our own rich progress bar
        )

    def complete(
        self,
        prompts: Sequence[str],
        *,
        output_schema: Type[BaseModel] | dict[str, Any] | None = None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> list[str]:
        prompts_list = list(prompts)

        # When show_progress=True, lm-deluge's built-in progress bar often shows
        # `0it ... ?it/s` because it initializes with total=0 and adjusts later.
        # For clearer example-level progress, we drive our own rich bar by waiting
        # for tasks as they complete.
        if show_progress and prompts_list:
            async def _run_with_bar() -> list[Any]:
                # Ensure tracker exists but disable its own progress bar.
                if getattr(self._client, "_tracker", None) is None:
                    self._client.open(total=0, show_progress=False)

                # Start tasks.
                task_ids: list[int] = []
                for p in prompts_list:
                    task_ids.append(self._client.start_nowait(p, output_schema=output_schema))

                # Collect as tasks complete.
                results_by_id: dict[int, Any] = {}
                it = self._client.as_completed(task_ids)
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]{task.description}"),
                    BarColumn(bar_width=30),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task(progress_desc or "LLM", total=len(task_ids))
                    async for tid, resp in it:
                        results_by_id[tid] = resp
                        progress.advance(task)

                # Preserve original ordering.
                ordered: list[Any] = [results_by_id[tid] for tid in task_ids]

                # Close tracker if we opened it (best-effort).
                try:
                    self._client.close()
                except Exception:
                    pass
                return ordered

            import asyncio

            resps = asyncio.run(_run_with_bar())
        else:
            resps = self._client.process_prompts_sync(
                prompts_list,
                output_schema=output_schema,
                show_progress=False,
            )

        # Track aggregate usage for this call.
        usage = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        for r in resps:
            if r is None:
                continue
            # cost
            c = getattr(r, "cost", None)
            if isinstance(c, (int, float)):
                usage["cost_usd"] += float(c)
            # tokens
            u = getattr(r, "usage", None)
            if u is not None:
                it = getattr(u, "input_tokens", None)
                ot = getattr(u, "output_tokens", None)
                crt = getattr(u, "cache_read_tokens", None)
                cwt = getattr(u, "cache_write_tokens", None)
                if isinstance(it, int):
                    usage["input_tokens"] += it
                if isinstance(ot, int):
                    usage["output_tokens"] += ot
                if isinstance(crt, int):
                    usage["cache_read_tokens"] += crt
                if isinstance(cwt, int):
                    usage["cache_write_tokens"] += cwt

        self.last_usage = usage
        # Accumulate total usage across calls.
        self.total_usage["cost_usd"] = float(self.total_usage["cost_usd"]) + float(usage.get("cost_usd", 0.0))
        self.total_usage["input_tokens"] = int(self.total_usage["input_tokens"]) + int(usage.get("input_tokens", 0))
        self.total_usage["output_tokens"] = int(self.total_usage["output_tokens"]) + int(usage.get("output_tokens", 0))
        self.total_usage["cache_read_tokens"] = int(self.total_usage["cache_read_tokens"]) + int(
            usage.get("cache_read_tokens", 0)
        )
        self.total_usage["cache_write_tokens"] = int(self.total_usage["cache_write_tokens"]) + int(
            usage.get("cache_write_tokens", 0)
        )

        out: list[str] = []
        for r in resps:
            if r is None or r.completion is None:
                out.append("")
            else:
                out.append(r.completion)
        return out

    def complete_json(
        self,
        prompts: Sequence[str],
        *,
        output_schema: Type[BaseModel] | dict[str, Any] | None = None,
        max_parse_retries: int = 2,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return a list of parsed JSON dicts.

        Notes:
        - If output_schema is provided, the provider may enforce a JSON schema response format.
          We still parse locally (and retry on malformed output) for robustness.
        """
        prompts_list = list(prompts)
        # Aggregate usage across the main call + any retry calls.
        agg_usage: dict[str, float | int] = {
            "cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        def _add_usage(u: dict[str, float | int]) -> None:
            agg_usage["cost_usd"] = float(agg_usage["cost_usd"]) + float(u.get("cost_usd", 0.0))
            agg_usage["input_tokens"] = int(agg_usage["input_tokens"]) + int(u.get("input_tokens", 0))
            agg_usage["output_tokens"] = int(agg_usage["output_tokens"]) + int(u.get("output_tokens", 0))
            agg_usage["cache_read_tokens"] = int(agg_usage["cache_read_tokens"]) + int(u.get("cache_read_tokens", 0))
            agg_usage["cache_write_tokens"] = int(agg_usage["cache_write_tokens"]) + int(u.get("cache_write_tokens", 0))

        completions = self.complete(
            prompts_list,
            output_schema=output_schema,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
        _add_usage(self.last_usage)

        parsed: list[Optional[dict[str, Any]]] = []
        bad_idxs: list[int] = []
        for i, txt in enumerate(completions):
            obj = _try_parse_json(txt)
            if isinstance(obj, dict):
                parsed.append(obj)
            else:
                parsed.append(None)
                bad_idxs.append(i)

        # Retry malformed outputs individually with a stricter suffix.
        attempt = 0
        while bad_idxs and attempt < max_parse_retries:
            attempt += 1
            retry_prompts = [
                prompts_list[i]
                + "\n\nReturn ONLY valid json. Do not include any other text."
                for i in bad_idxs
            ]
            retry_completions = self.complete(
                retry_prompts, output_schema=output_schema, show_progress=False
            )
            _add_usage(self.last_usage)
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
                "Deluge returned unparseable output for %d example(s) after %d retries. "
                "These will be recorded as empty dicts (parse failures).",
                len(bad_idxs),
                attempt,
            )

        self.last_usage = agg_usage
        return [p if p is not None else {} for p in parsed]

