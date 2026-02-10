#!/usr/bin/env python3
"""
Deduplicate training examples against a test set using embedding similarity.

For each training example, computes cosine similarity to all test examples
and removes any training example whose nearest-neighbor similarity exceeds
the threshold (default: 0.65).

Supports local files (JSONL, JSON, CSV, Parquet) and HuggingFace datasets.
Field keys support dot-notation for nested access, e.g.:
    --train-goal-key final_procedure.goal
    --train-steps-key final_procedure.steps

Example usage:
    uv run python examples/train/dedup_against_test.py \
        --train-path hf://how2everything/how2train_rl_100k?split=train \
        --test-path hf://how2everything/how2bench?split=train \
        --train-goal-key goal \
        --train-steps-key steps \
        --test-goal-key goal \
        --test-steps-key steps \
        --output-path data/train_deduped.jsonl

The script also writes:
    - A JSONL file with nearest-neighbor matches for inspection (--matches-path)
    - A histogram of similarity scores (next to --matches-path)
    - Optionally, the removed examples (--removed-path)
    - Optionally, cached embeddings for reuse (--save-embeddings / --load-embeddings)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


def resolve_key(obj: dict[str, Any], key: str) -> Any:
    """Resolve a dot-separated key path against a nested dict.

    Example: resolve_key({"a": {"b": 1}}, "a.b") -> 1
    Returns None if any segment is missing or not a dict.
    """
    current: Any = obj
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _as_str(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _as_list_of_str(v: Any) -> list[str] | None:
    if not isinstance(v, list):
        return None
    out = [s.strip() for s in v if isinstance(s, str) and s.strip()]
    return out if out else None


# ---------------------------------------------------------------------------
# Data loading (format-agnostic)
# ---------------------------------------------------------------------------


def _iter_records_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _iter_records_json(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield data


def _iter_records_csv(path: Path) -> Iterator[dict[str, Any]]:
    import csv

    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yield dict(row)


def _iter_records_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("Reading Parquet files requires pyarrow. Install with: pip install pyarrow")
    table = pq.read_table(path)
    for batch in table.to_batches():
        for row in batch.to_pylist():
            yield row


def _iter_records_hf(repo: str, split: str | None) -> Iterator[dict[str, Any]]:
    """Load records from a HuggingFace dataset.

    Tries streaming first for memory efficiency; falls back to full download
    if streaming fails (e.g. script-based datasets).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Reading HuggingFace datasets requires the datasets package. Install with: pip install datasets")

    resolved_split = split or "train"
    ds = load_dataset(repo, split=resolved_split)

    for rec in ds:
        yield dict(rec) if not isinstance(rec, dict) else rec


def _parse_hf_path(path_str: str) -> tuple[str, str | None]:
    """Parse 'hf://repo/name' or 'hf://repo/name?split=train' into (repo, split)."""
    rest = path_str[len("hf://"):]
    split = None
    if "?" in rest:
        rest, query = rest.split("?", 1)
        for part in query.split("&"):
            if part.startswith("split="):
                split = part[len("split="):]
    return rest, split


def iter_records(path_str: str) -> Iterator[dict[str, Any]]:
    """Iterate records from a local file or HuggingFace dataset.

    Supported:
        - Local: .jsonl, .json, .csv, .parquet
        - HuggingFace: hf://repo/name (optionally hf://repo/name?split=train)
    """
    if path_str.startswith("hf://"):
        repo, split = _parse_hf_path(path_str)
        logger.info("Loading HuggingFace dataset: %s (split=%s)", repo, split or "train")
        yield from _iter_records_hf(repo, split)
        return

    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_records_jsonl(path)
    elif suffix == ".json":
        yield from _iter_records_json(path)
    elif suffix == ".csv":
        yield from _iter_records_csv(path)
    elif suffix == ".parquet":
        yield from _iter_records_parquet(path)
    else:
        # Default: try JSONL
        logger.warning("Unknown extension %s, attempting JSONL parse.", suffix)
        yield from _iter_records_jsonl(path)


def _format_for_embedding(goal: str, steps: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return f"{goal}\n{numbered}"


def load_records(
    path_str: str,
    *,
    goal_key: str,
    steps_key: str,
    topic_key: str | None = None,
    max_items: int = 0,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load records and return (texts_for_embedding, info_dicts, raw_records).

    ``path_str`` can be a local file path or ``hf://repo/name[?split=...]``.
    ``goal_key`` and ``steps_key`` support dot-notation for nested fields.
    """
    items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    skipped = 0
    for obj in iter_records(path_str):
        goal = _as_str(resolve_key(obj, goal_key))
        steps = _as_list_of_str(resolve_key(obj, steps_key))
        if goal is None or steps is None:
            skipped += 1
            continue

        topic = ""
        if topic_key:
            topic = _as_str(resolve_key(obj, topic_key)) or ""

        text = _format_for_embedding(goal, steps)
        info = {
            "goal": goal,
            "steps": steps,
            "topic": topic,
        }
        items.append((text, info, obj))

    if skipped:
        logger.warning(
            "%s: skipped %d records (missing or invalid %s / %s fields)",
            path_str, skipped, goal_key, steps_key,
        )

    if max_items and len(items) > max_items:
        random.Random(seed).shuffle(items)
        items = items[:max_items]

    texts = [t for t, _, _ in items]
    infos = [i for _, i, _ in items]
    raws = [r for _, _, r in items]
    return texts, infos, raws


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def compute_embeddings(
    model: Any,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def find_nearest(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    chunk_size: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query, find the index and cosine similarity of its nearest corpus vector.

    Processes in chunks to keep memory bounded.
    Returns (best_indices, best_scores) arrays of shape (n_query,).
    """
    n_query = query_emb.shape[0]
    best_indices = np.zeros(n_query, dtype=np.int64)
    best_scores = np.full(n_query, -np.inf, dtype=np.float32)

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        # Normalized embeddings → dot product = cosine similarity.
        sims = query_emb[start:end] @ corpus_emb.T
        best_indices[start:end] = np.argmax(sims, axis=1)
        best_scores[start:end] = np.max(sims, axis=1)
        logger.info("Processed %d–%d / %d", start, end, n_query)

    return best_indices, best_scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_similarity_distribution(scores: np.ndarray, out_path: str, bins: int = 100) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping similarity plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, alpha=0.7, color="#2E86AB", edgecolor="white", linewidth=0.5)

    mean_val = float(np.mean(scores))
    median_val = float(np.median(scores))
    p90 = float(np.percentile(scores, 90))
    p95 = float(np.percentile(scores, 95))
    p99 = float(np.percentile(scores, 99))

    plt.axvline(mean_val, color="#E94F37", linestyle="-", linewidth=2, label=f"Mean: {mean_val:.3f}")
    plt.axvline(median_val, color="#F6AE2D", linestyle="--", linewidth=2, label=f"Median: {median_val:.3f}")
    plt.axvline(p90, color="#A23B72", linestyle=":", linewidth=1.5, label=f"90th %ile: {p90:.3f}")
    plt.axvline(p95, color="#5C8001", linestyle=":", linewidth=1.5, label=f"95th %ile: {p95:.3f}")
    plt.axvline(p99, color="#1B998B", linestyle=":", linewidth=1.5, label=f"99th %ile: {p99:.3f}")

    plt.xlabel("Cosine Similarity (train → nearest test)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Nearest-Test Similarity Scores", fontsize=14)
    plt.legend(loc="upper left", fontsize=10)
    plt.xlim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deduplicate training data against a test set using embedding similarity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input formats:\n"
            "  Local files: .jsonl, .json, .csv, .parquet\n"
            "  HuggingFace: hf://repo/name  (default split: train)\n"
            "               hf://repo/name?split=train\n"
            "\n"
            "Field keys support dot-notation for nested JSON, e.g.:\n"
            "  --train-goal-key final_procedure.goal\n"
            "  --train-steps-key final_procedure.steps\n"
            "\n"
            "Examples:\n"
            "  # Local JSONL files\n"
            "  python dedup_against_test.py \\\n"
            "      --train-path data/train.jsonl \\\n"
            "      --test-path data/test.jsonl \\\n"
            "      --output-path data/train_deduped.jsonl\n"
            "\n"
            "  # Test set from HuggingFace\n"
            "  python dedup_against_test.py \\\n"
            "      --train-path data/train.jsonl \\\n"
            "      --test-path hf://how2everything/how2bench \\\n"
            "      --test-goal-key goal --test-steps-key steps \\\n"
            "      --output-path data/train_deduped.jsonl"
        ),
    )

    # I/O
    p.add_argument(
        "--train-path", type=str, required=True,
        help="Training data. Local file (.jsonl, .json, .csv, .parquet) or hf://repo/name.",
    )
    p.add_argument(
        "--test-path", type=str, required=True,
        help="Test data. Local file (.jsonl, .json, .csv, .parquet) or hf://repo/name[?split=...].",
    )
    p.add_argument("--output-path", type=Path, required=True, help="Output path for deduplicated training JSONL.")

    # Field keys (with sensible defaults)
    p.add_argument("--train-goal-key", type=str, default="goal", help="Key path to goal in training records. (default: goal)")
    p.add_argument("--train-steps-key", type=str, default="steps", help="Key path to steps in training records. (default: steps)")
    p.add_argument("--train-topic-key", type=str, default=None, help="Optional key path to topic in training records.")
    p.add_argument("--test-goal-key", type=str, default="goal", help="Key path to goal in test records. (default: goal)")
    p.add_argument("--test-steps-key", type=str, default="steps", help="Key path to steps in test records. (default: steps)")
    p.add_argument("--test-topic-key", type=str, default=None, help="Optional key path to topic in test records.")

    # Dedup settings
    p.add_argument(
        "--threshold", type=float, default=0.65,
        help="Remove training examples with nearest-test similarity >= this value. (default: 0.65)",
    )
    p.add_argument(
        "--model", type=str, default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer model for embeddings. (default: Qwen/Qwen3-Embedding-0.6B)",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Encoding batch size. (default: 64)")
    p.add_argument("--chunk-size", type=int, default=5000, help="Similarity computation chunk size. (default: 5000)")
    p.add_argument("--seed", type=int, default=42, help="Random seed. (default: 42)")
    p.add_argument("--max-train", type=int, default=0, help="Max training samples (0 = all). (default: 0)")
    p.add_argument("--max-test", type=int, default=0, help="Max test samples (0 = all). (default: 0)")
    p.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Device for the embedding model. (default: auto)",
    )

    # Optional outputs
    p.add_argument("--removed-path", type=Path, default=None, help="Write removed examples here for inspection.")
    p.add_argument(
        "--matches-path", type=Path, default=None,
        help="Write per-train nearest-test match details (JSONL). Default: <output_path>.matches.jsonl",
    )
    p.add_argument("--save-embeddings", type=Path, default=None, help="Cache embeddings to this .npz file.")
    p.add_argument("--load-embeddings", type=Path, default=None, help="Load cached embeddings from this .npz file.")
    return p.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    if requested == "cuda" and not has_cuda:
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return "cuda" if has_cuda else "cpu"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    matches_path = args.matches_path or args.output_path.with_suffix(".matches.jsonl")

    logger.info("Train: %s (goal=%s, steps=%s, topic=%s)", args.train_path, args.train_goal_key, args.train_steps_key, args.train_topic_key)
    logger.info("Test:  %s (goal=%s, steps=%s, topic=%s)", args.test_path, args.test_goal_key, args.test_steps_key, args.test_topic_key)

    # ------------------------------------------------------------------
    # 1. Load data or cached embeddings
    # ------------------------------------------------------------------
    if args.load_embeddings:
        logger.info("Loading cached embeddings from %s", args.load_embeddings)
        t0 = time.perf_counter()
        data = np.load(args.load_embeddings, allow_pickle=True)
        train_emb = data["train_emb"]
        test_emb = data["test_emb"]
        train_infos = data["train_infos"].tolist()
        test_infos = data["test_infos"].tolist()
        # Raw records aren't saved in the cache; reload them.
        _, _, train_raws = load_records(
            args.train_path,
            goal_key=args.train_goal_key,
            steps_key=args.train_steps_key,
            topic_key=args.train_topic_key,
            max_items=args.max_train,
            seed=args.seed,
        )
        logger.info(
            "Loaded embeddings: train=%s, test=%s (%.1fs)",
            train_emb.shape, test_emb.shape, time.perf_counter() - t0,
        )
    else:
        logger.info("Loading training data...")
        train_texts, train_infos, train_raws = load_records(
            args.train_path,
            goal_key=args.train_goal_key,
            steps_key=args.train_steps_key,
            topic_key=args.train_topic_key,
            max_items=args.max_train,
            seed=args.seed,
        )
        logger.info("  %d training examples", len(train_texts))

        logger.info("Loading test data...")
        test_texts, test_infos, _ = load_records(
            args.test_path,
            goal_key=args.test_goal_key,
            steps_key=args.test_steps_key,
            topic_key=args.test_topic_key,
            max_items=args.max_test,
            seed=args.seed,
        )
        logger.info("  %d test examples", len(test_texts))

        if not train_texts:
            raise ValueError(f"No usable records in {args.train_path} (check --train-goal-key / --train-steps-key)")
        if not test_texts:
            raise ValueError(f"No usable records in {args.test_path} (check --test-goal-key / --test-steps-key)")

        # Load embedding model
        device = _resolve_device(args.device)
        logger.info("Loading embedding model: %s (device=%s)", args.model, device)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, device=device)

        # Compute embeddings
        logger.info("Encoding %d training texts...", len(train_texts))
        t0 = time.perf_counter()
        train_emb = compute_embeddings(model, train_texts, batch_size=args.batch_size)
        logger.info("  done (%.1fs)", time.perf_counter() - t0)

        logger.info("Encoding %d test texts...", len(test_texts))
        t0 = time.perf_counter()
        test_emb = compute_embeddings(model, test_texts, batch_size=args.batch_size)
        logger.info("  done (%.1fs)", time.perf_counter() - t0)

        # Optionally cache embeddings
        if args.save_embeddings:
            args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Saving embeddings to %s", args.save_embeddings)
            np.savez_compressed(
                args.save_embeddings,
                train_emb=train_emb,
                test_emb=test_emb,
                train_infos=np.array(train_infos, dtype=object),
                test_infos=np.array(test_infos, dtype=object),
            )

    # ------------------------------------------------------------------
    # 2. Find nearest test example for each training example
    # ------------------------------------------------------------------
    logger.info("Finding nearest test example for each training example (chunk_size=%d)...", args.chunk_size)
    t0 = time.perf_counter()
    best_indices, best_scores = find_nearest(train_emb, test_emb, chunk_size=args.chunk_size)
    logger.info("  done (%.1fs)", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # 3. Write nearest-match details
    # ------------------------------------------------------------------
    logger.info("Writing match details to %s", matches_path)
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    with open(matches_path, "w", encoding="utf-8") as f:
        for i, (train_info, test_idx, score) in enumerate(zip(train_infos, best_indices, best_scores)):
            test_info = test_infos[int(test_idx)]
            record = {
                "train_idx": i,
                "train_goal": train_info["goal"],
                "train_steps": train_info["steps"],
                "train_topic": train_info["topic"],
                "nearest_test_idx": int(test_idx),
                "nearest_test_goal": test_info["goal"],
                "nearest_test_steps": test_info["steps"],
                "nearest_test_topic": test_info["topic"],
                "cosine_similarity": round(float(score), 6),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 4. Filter and write deduplicated training set
    # ------------------------------------------------------------------
    kept_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for raw, score in zip(train_raws, best_scores):
        if score < args.threshold:
            kept_records.append(raw)
        else:
            removed_records.append(raw)

    _write_jsonl(args.output_path, kept_records)
    logger.info("Kept %d / %d examples (wrote %s)", len(kept_records), len(train_raws), args.output_path)

    if args.removed_path:
        _write_jsonl(args.removed_path, removed_records)
        logger.info("Removed %d examples (wrote %s)", len(removed_records), args.removed_path)

    # ------------------------------------------------------------------
    # 5. Summary statistics
    # ------------------------------------------------------------------
    logger.info(
        "Similarity stats: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        np.min(best_scores), np.max(best_scores), np.mean(best_scores), np.median(best_scores),
    )
    for thresh in [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95]:
        count = int(np.sum(best_scores >= thresh))
        pct = 100.0 * count / len(best_scores)
        marker = " <-- threshold" if thresh == args.threshold else ""
        logger.info("  >= %.2f: %d (%.1f%%)%s", thresh, count, pct, marker)

    # Plot
    plot_path = str(matches_path).replace(".jsonl", "_distribution.png")
    plot_similarity_distribution(best_scores, plot_path)

    print(f"\nResults:")
    print(f"  Kept:    {len(kept_records)} / {len(train_raws)} ({100 * len(kept_records) / len(train_raws):.1f}%)")
    print(f"  Removed: {len(removed_records)} (threshold={args.threshold})")
    print(f"  Output:  {args.output_path}")


if __name__ == "__main__":
    main()
