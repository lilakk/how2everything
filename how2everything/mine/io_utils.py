from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def sanitize_topic_name(topic: str) -> str:
    safe = topic.replace("/", "-").replace("\\", "-")
    safe = safe.replace(":", "-").replace(" ", "_")
    safe = safe.replace("&", "and")
    safe = safe.replace(",", "")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch in ("-", "_"))
    return safe or "topic"


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def processed_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for rec in iter_jsonl(path):
        src = rec.get("source_example") if isinstance(rec, dict) else None
        if isinstance(src, dict):
            doc_id = src.get("id")
            if doc_id is not None:
                ids.add(str(doc_id))
    return ids

