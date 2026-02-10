from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from how2everything.bench.config import InputsConfig
from how2everything.bench.schemas import BenchExample
from how2everything.mine.document_sources import iter_documents, iter_documents_many, iter_input_paths


RecordKind = Literal["how2mine_export", "bench"]


def _detect_kind(rec: dict[str, Any]) -> RecordKind:
    if "final_procedure" in rec and "source_example" in rec:
        return "how2mine_export"
    return "bench"


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _as_list_of_str(v: Any) -> list[str]:
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
            elif x is not None:
                s = str(x).strip()
                if s:
                    out.append(s)
        return out
    return []


def _bench_from_how2mine_export(rec: dict[str, Any]) -> BenchExample | None:
    src = rec.get("source_example")
    fp = rec.get("final_procedure")
    if not isinstance(src, dict) or not isinstance(fp, dict):
        return None

    eid = _as_str(src.get("id")).strip()
    if not eid:
        eid = _as_str(rec.get("id")).strip()
    if not eid:
        return None

    topic = _as_str(src.get("topic")).strip() or _as_str(rec.get("topic")).strip()
    goal = _as_str(fp.get("goal")).strip()
    steps = _as_list_of_str(fp.get("steps"))
    resources = _as_list_of_str(fp.get("resources"))
    url = _as_str(src.get("url")).strip() if src.get("url") else ""
    source_text = _as_str(src.get("text")).strip() if src.get("text") else ""

    if not goal or not steps:
        return None

    return BenchExample(
        source_example_id=eid,
        topic=topic,
        goal=goal,
        steps=steps,
        resources=resources,
        url=url,
        source_text=source_text,
    )


def _bench_from_bench(rec: dict[str, Any]) -> BenchExample | None:
    eid = _as_str(rec.get("source_example_id") or rec.get("id")).strip()
    goal = _as_str(rec.get("goal")).strip()
    steps = _as_list_of_str(rec.get("steps"))
    topic = _as_str(rec.get("topic")).strip()
    resources = _as_list_of_str(rec.get("resources"))
    url = _as_str(rec.get("url")).strip() if rec.get("url") else ""
    source_text = _as_str(rec.get("source_text")).strip() if rec.get("source_text") else ""
    if not eid or not goal or not steps:
        return None
    return BenchExample(
        source_example_id=eid,
        topic=topic,
        goal=goal,
        steps=steps,
        resources=resources,
        url=url,
        source_text=source_text,
    )


def record_to_bench_example(rec: dict[str, Any], *, kind: str) -> BenchExample | None:
    if kind == "how2mine_export":
        return _bench_from_how2mine_export(rec)
    if kind == "bench":
        return _bench_from_bench(rec)
    if kind == "auto":
        detected = _detect_kind(rec)
        return record_to_bench_example(rec, kind=detected)
    raise ValueError(f"Unsupported inputs.kind: {kind}")


def iter_input_records(cfg: InputsConfig) -> Iterator[dict[str, Any]]:
    # Local path mode.
    if cfg.path is not None:
        p = cfg.path
        if p.is_dir():
            paths = iter_input_paths(p, cfg.include_globs)
            yield from iter_documents_many(paths=paths, fmt=cfg.format, compression=cfg.compression)
            return
        yield from iter_documents(path=p, fmt=cfg.format, compression=cfg.compression)
        return

    # Hugging Face dataset mode.
    if not cfg.hf_repo:
        raise ValueError("inputs.path is omitted but inputs.hf_repo is empty.")
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise ImportError(
            "Reading from Hugging Face requires the `datasets` package. "
            "Install it (pip install datasets)."
        ) from e

    cache_dir = str(cfg.hf_cache_dir) if cfg.hf_cache_dir is not None else None
    ds = load_dataset(
        cfg.hf_repo,
        split=cfg.hf_split,
        streaming=bool(cfg.hf_streaming),
        cache_dir=cache_dir,
    )
    for rec in ds:
        # Streaming datasets yield dict-like objects already; normalize to plain dict.
        if isinstance(rec, dict):
            yield rec
        else:
            try:
                yield dict(rec)  # type: ignore[arg-type]
            except Exception:
                continue


def iter_bench_examples(cfg: InputsConfig) -> Iterator[BenchExample]:
    for rec in iter_input_records(cfg):
        if not isinstance(rec, dict):
            continue
        ex = record_to_bench_example(rec, kind=cfg.kind)
        if ex is None:
            continue
        yield ex


def load_bench_examples(cfg: InputsConfig, *, max_examples: int | None = None) -> list[BenchExample]:
    out: list[BenchExample] = []
    for ex in iter_bench_examples(cfg):
        out.append(ex)
        if max_examples is not None and len(out) >= max_examples:
            break
    return out

