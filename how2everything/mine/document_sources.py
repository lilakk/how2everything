from __future__ import annotations

import bz2
import csv
import contextlib
import gzip
import io
import json
import lzma
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from how2everything.mine.io_utils import iter_jsonl


SUPPORTED_FORMATS = {"auto", "jsonl", "csv", "arrow", "parquet"}
SUPPORTED_COMPRESSIONS = {"auto", "none", "zst", "gz", "bz2", "xz"}


def _suffixes_lower(path: Path) -> list[str]:
    return [s.lower() for s in path.suffixes]


def infer_compression(path: Path) -> str:
    """
    Infer compression from file suffixes.

    Examples:
    - *.jsonl.zst -> zst
    - *.csv.gz -> gz
    - *.jsonl -> none
    """
    sufs = _suffixes_lower(path)
    if not sufs:
        return "none"
    last = sufs[-1]
    if last == ".zst":
        return "zst"
    if last == ".gz":
        return "gz"
    if last == ".bz2":
        return "bz2"
    if last in (".xz", ".lzma"):
        return "xz"
    return "none"


def infer_format(path: Path) -> str:
    """
    Infer data format from suffixes (ignoring compression suffixes).
    """
    sufs = _suffixes_lower(path)
    # Strip a recognized compression suffix for format inference.
    if sufs and sufs[-1] in (".zst", ".gz", ".bz2", ".xz", ".lzma"):
        sufs = sufs[:-1]
    if not sufs:
        return "jsonl"
    last = sufs[-1]
    if last in (".jsonl", ".json"):
        return "jsonl"
    if last == ".csv":
        return "csv"
    if last in (".arrow", ".feather", ".ipc"):
        return "arrow"
    if last in (".parquet", ".pq"):
        return "parquet"
    # Default: keep old behavior (treat as JSONL-like).
    return "jsonl"


@contextlib.contextmanager
def _open_text_maybe_compressed(path: Path, compression: str) -> Iterator[io.TextIOBase]:
    """
    Context-managed text stream for `path`, decompressing if needed.
    Ensures all underlying file handles are closed.
    """
    if compression == "none":
        with path.open("r", encoding="utf-8", newline="") as f:
            yield f
        return
    if compression == "gz":
        with gzip.open(path, "rb") as bf:
            with io.TextIOWrapper(bf, encoding="utf-8", newline="") as f:
                yield f
        return
    if compression == "bz2":
        with bz2.open(path, "rb") as bf:
            with io.TextIOWrapper(bf, encoding="utf-8", newline="") as f:
                yield f
        return
    if compression == "xz":
        with lzma.open(path, "rb") as bf:
            with io.TextIOWrapper(bf, encoding="utf-8", newline="") as f:
                yield f
        return
    if compression == "zst":
        import zstandard as zstd  # local import: optional-ish at runtime

        with path.open("rb") as bf:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(bf) as stream:
                with io.TextIOWrapper(stream, encoding="utf-8", newline="") as f:
                    yield f
        return
    raise ValueError(f"Unsupported compression: {compression}")


def iter_jsonl_compressed(path: Path, *, compression: str) -> Iterator[dict[str, Any]]:
    if compression == "none":
        yield from iter_jsonl(path)
        return
    with _open_text_maybe_compressed(path, compression) as f:
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


def iter_csv_rows(path: Path, *, compression: str) -> Iterator[dict[str, Any]]:
    with _open_text_maybe_compressed(path, compression) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # csv returns dict[str, str|None]
            out: dict[str, Any] = {}
            for k, v in row.items():
                if k is None:
                    continue
                out[str(k)] = v if v is not None else ""
            yield out


def _arrow_value_to_python(v: Any) -> Any:
    # pyarrow scalars often have .as_py(); plain python values pass through.
    try:
        as_py = getattr(v, "as_py", None)
        if callable(as_py):
            return as_py()
    except Exception:
        pass
    return v


def iter_arrow_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    # Try file format first, then stream.
    with path.open("rb") as f:
        try:
            reader = ipc.open_file(f)
        except Exception:
            f.seek(0)
            reader = ipc.open_stream(f)

        schema: pa.Schema = reader.schema
        names = list(schema.names)

        # RecordBatchFileReader isn't directly iterable; normalise to a
        # list of batches so both file and stream formats work.
        if isinstance(reader, ipc.RecordBatchFileReader):
            batches = [reader.get_batch(i) for i in range(reader.num_record_batches)]
        else:
            batches = reader  # stream reader is iterable

        for batch in batches:
            cols = [batch.column(i) for i in range(batch.num_columns)]
            for row_idx in range(batch.num_rows):
                out: dict[str, Any] = {}
                for name, col in zip(names, cols):
                    out[str(name)] = _arrow_value_to_python(col[row_idx])
                yield out


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches():
        names = list(batch.schema.names)
        cols = [batch.column(i) for i in range(batch.num_columns)]
        for row_idx in range(batch.num_rows):
            out: dict[str, Any] = {}
            for name, col in zip(names, cols):
                out[str(name)] = _arrow_value_to_python(col[row_idx])
            yield out


def iter_documents(
    *,
    path: Path,
    fmt: str = "auto",
    compression: str = "auto",
) -> Iterator[dict[str, Any]]:
    """
    Yield document records (dicts) from various on-disk formats.

    This function intentionally does NOT impose a schema beyond "dict-like rows".
    Field mapping (id/text/url/topic column names) is handled by config values
    in the pipeline stages.
    """
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported inputs.format: {fmt} (supported: {sorted(SUPPORTED_FORMATS)})")
    if compression not in SUPPORTED_COMPRESSIONS:
        raise ValueError(
            f"Unsupported inputs.compression: {compression} (supported: {sorted(SUPPORTED_COMPRESSIONS)})"
        )
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if fmt == "auto":
        fmt = infer_format(path)
    if compression == "auto":
        compression = infer_compression(path)

    if fmt == "jsonl":
        yield from iter_jsonl_compressed(path, compression=compression)
        return
    if fmt == "csv":
        yield from iter_csv_rows(path, compression=compression)
        return
    if fmt == "arrow":
        if compression != "none":
            raise ValueError("Arrow/IPC inputs do not support external compression; use an uncompressed file.")
        yield from iter_arrow_rows(path)
        return
    if fmt == "parquet":
        if compression != "none":
            raise ValueError("Parquet inputs do not support external compression; use an uncompressed file.")
        yield from iter_parquet_rows(path)
        return

    raise ValueError(f"Unhandled fmt: {fmt}")


def iter_input_paths(root: Path, include_globs: list[str]) -> list[Path]:
    """
    Expand a directory into a deterministic list of input files.

    Non-recursive by design. Globs are interpreted as Path.glob patterns, e.g.
    - "*.jsonl"
    - "*.jsonl.zst"
    - "*.csv*"
    - "*.arrow"
    """
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    if not root.is_dir():
        raise ValueError(f"iter_input_paths expects a directory, got: {root}")
    if not include_globs:
        raise ValueError("inputs.include_globs must be non-empty when inputs.path is a directory.")

    out: list[Path] = []
    seen: set[Path] = set()
    for pat in include_globs:
        pat = str(pat).strip()
        if not pat:
            continue
        for p in root.glob(pat):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)

    out.sort(key=lambda p: p.name)
    return out


def iter_documents_many(
    *,
    paths: list[Path],
    fmt: str = "auto",
    compression: str = "auto",
) -> Iterator[dict[str, Any]]:
    """
    Yield document rows from many input files, in the order provided.
    """
    for p in paths:
        yield from iter_documents(path=p, fmt=fmt, compression=compression)

