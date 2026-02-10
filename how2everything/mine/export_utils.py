from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SIZE_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTP]?B)\s*$", re.IGNORECASE)


def parse_size(size: str) -> int:
    """
    Parse a human file size string like "500MB", "2GB", "1024KB".
    Uses base-2 units (KiB/MiB/GiB) but accepts KB/MB/GB suffixes.
    """
    m = _SIZE_RE.match(size or "")
    if not m:
        raise ValueError(f"Invalid size string: {size!r} (expected like '500MB' or '2GB').")
    num = float(m.group("num"))
    unit = m.group("unit").upper()
    mult = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }[unit]
    out = int(num * mult)
    if out <= 0:
        raise ValueError(f"Size must be > 0, got: {size!r}")
    return out


def _jsonl_line_bytes(rec: dict[str, Any]) -> bytes:
    return (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")


def _shard_name(stem: str, idx: int, ext: str) -> str:
    return f"{stem}_{idx:05d}.{ext}"


def _export_ext(fmt: str) -> str:
    if fmt == "jsonl":
        return "jsonl"
    if fmt == "jsonl_zst":
        return "jsonl.zst"
    if fmt == "parquet":
        return "parquet"
    if fmt == "arrow_ipc":
        return "arrow"
    raise ValueError(f"Unsupported export format: {fmt}")


@dataclass
class ExportResult:
    shard_paths: list[Path]


def export_all_valid(
    *,
    records: Iterable[dict[str, Any]],
    out_dir: Path,
    stem: str = "all_valid",
    fmt: str = "jsonl",
    max_file_size: str | None = None,
) -> ExportResult:
    """
    Export records to out_dir in the requested format, optionally sharding.

    Sharding is based on *estimated uncompressed JSONL bytes* of records to keep
    behavior consistent across formats.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = (fmt or "jsonl").strip()
    ext = _export_ext(fmt)

    max_bytes: int | None = None
    if max_file_size:
        max_bytes = parse_size(max_file_size)

    if fmt == "jsonl":
        return _export_jsonl(records, out_dir=out_dir, stem=stem, ext=ext, max_bytes=max_bytes)
    if fmt == "jsonl_zst":
        return _export_jsonl_zst(records, out_dir=out_dir, stem=stem, ext=ext, max_bytes=max_bytes)
    if fmt == "parquet":
        return _export_parquet(records, out_dir=out_dir, stem=stem, ext=ext, max_bytes=max_bytes)
    if fmt == "arrow_ipc":
        return _export_arrow_ipc(records, out_dir=out_dir, stem=stem, ext=ext, max_bytes=max_bytes)
    raise ValueError(f"Unsupported export format: {fmt}")


def _export_jsonl(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str,
    ext: str,
    max_bytes: int | None,
) -> ExportResult:
    if max_bytes is None:
        p = out_dir / f"{stem}.{ext}"
        with p.open("wb") as f:
            for rec in records:
                f.write(_jsonl_line_bytes(rec))
        return ExportResult([p])

    shard_paths: list[Path] = []
    shard_idx = 0
    cur_path = out_dir / _shard_name(stem, shard_idx, ext)
    shard_paths.append(cur_path)
    f = cur_path.open("wb")
    try:
        cur_bytes = 0
        cur_count = 0
        for rec in records:
            b = _jsonl_line_bytes(rec)
            if cur_count > 0 and (cur_bytes + len(b) > max_bytes):
                f.close()
                shard_idx += 1
                cur_path = out_dir / _shard_name(stem, shard_idx, ext)
                shard_paths.append(cur_path)
                f = cur_path.open("wb")
                cur_bytes = 0
                cur_count = 0
            f.write(b)
            cur_bytes += len(b)
            cur_count += 1
    finally:
        try:
            f.close()
        except Exception:
            pass
    return ExportResult(shard_paths)


def _export_jsonl_zst(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str,
    ext: str,
    max_bytes: int | None,
) -> ExportResult:
    import zstandard as zstd

    if max_bytes is None:
        p = out_dir / f"{stem}.{ext}"
        cctx = zstd.ZstdCompressor(level=3)
        with p.open("wb") as raw:
            with cctx.stream_writer(raw) as zw:
                for rec in records:
                    zw.write(_jsonl_line_bytes(rec))
        return ExportResult([p])

    shard_paths: list[Path] = []
    shard_idx = 0
    cur_path = out_dir / _shard_name(stem, shard_idx, ext)
    shard_paths.append(cur_path)
    cctx = zstd.ZstdCompressor(level=3)
    raw = cur_path.open("wb")
    zw = cctx.stream_writer(raw)
    try:
        cur_bytes = 0  # uncompressed estimate
        cur_count = 0
        for rec in records:
            b = _jsonl_line_bytes(rec)
            if cur_count > 0 and (cur_bytes + len(b) > max_bytes):
                zw.close()
                raw.close()
                shard_idx += 1
                cur_path = out_dir / _shard_name(stem, shard_idx, ext)
                shard_paths.append(cur_path)
                raw = cur_path.open("wb")
                zw = cctx.stream_writer(raw)
                cur_bytes = 0
                cur_count = 0
            zw.write(b)
            cur_bytes += len(b)
            cur_count += 1
    finally:
        try:
            zw.close()
        except Exception:
            pass
        try:
            raw.close()
        except Exception:
            pass
    return ExportResult(shard_paths)


def _export_parquet(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str,
    ext: str,
    max_bytes: int | None,
) -> ExportResult:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if max_bytes is None:
        p = out_dir / f"{stem}.{ext}"
        rows = list(records)
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, p)
        return ExportResult([p])

    shard_paths: list[Path] = []
    shard_idx = 0
    buf: list[dict[str, Any]] = []
    buf_bytes = 0

    def _flush() -> None:
        nonlocal shard_idx, buf, buf_bytes
        if not buf:
            return
        p = out_dir / _shard_name(stem, shard_idx, ext)
        shard_paths.append(p)
        table = pa.Table.from_pylist(buf)
        pq.write_table(table, p)
        shard_idx += 1
        buf = []
        buf_bytes = 0

    for rec in records:
        est = len(_jsonl_line_bytes(rec))
        if buf and (buf_bytes + est > max_bytes):
            _flush()
        buf.append(rec)
        buf_bytes += est
    _flush()
    return ExportResult(shard_paths)


def _export_arrow_ipc(
    records: Iterable[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str,
    ext: str,
    max_bytes: int | None,
) -> ExportResult:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    if max_bytes is None:
        p = out_dir / f"{stem}.{ext}"
        rows = list(records)
        table = pa.Table.from_pylist(rows)
        with p.open("wb") as f:
            with ipc.new_file(f, table.schema) as w:
                w.write_table(table)
        return ExportResult([p])

    shard_paths: list[Path] = []
    shard_idx = 0
    buf: list[dict[str, Any]] = []
    buf_bytes = 0

    def _flush() -> None:
        nonlocal shard_idx, buf, buf_bytes
        if not buf:
            return
        p = out_dir / _shard_name(stem, shard_idx, ext)
        shard_paths.append(p)
        table = pa.Table.from_pylist(buf)
        with p.open("wb") as f:
            with ipc.new_file(f, table.schema) as w:
                w.write_table(table)
        shard_idx += 1
        buf = []
        buf_bytes = 0

    for rec in records:
        est = len(_jsonl_line_bytes(rec))
        if buf and (buf_bytes + est > max_bytes):
            _flush()
        buf.append(rec)
        buf_bytes += est
    _flush()
    return ExportResult(shard_paths)

