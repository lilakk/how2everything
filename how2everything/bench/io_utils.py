from __future__ import annotations

from pathlib import Path
from typing import Any

from how2everything.mine.io_utils import iter_jsonl


def attempted_example_ids(path: Path, *, key: str = "source_example_id") -> set[str]:
    """
    Return a set of ids already present in a JSONL output file.
    """
    out: set[str] = set()
    for rec in iter_jsonl(path):
        if not isinstance(rec, dict):
            continue
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            out.add(v.strip())
    return out


def read_jsonl_by_id(path: Path, *, key: str = "source_example_id") -> dict[str, dict[str, Any]]:
    """
    Read JSONL file into a mapping keyed by key (default source_example_id).
    Later duplicates overwrite earlier ones (append-only resume).
    """
    out: dict[str, dict[str, Any]] = {}
    for rec in iter_jsonl(path):
        if not isinstance(rec, dict):
            continue
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            out[v.strip()] = rec
    return out

