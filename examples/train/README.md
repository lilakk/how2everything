## Training

This folder contains scripts and documentation for training the **how2judge** model (the distilled procedural evaluation judge used by how2bench).

Training experiments were run via [open-instruct](https://github.com/allenai/open-instruct). This folder provides the data preparation utilities and pointers needed to reproduce the training.

### Overview

The training pipeline is:

1. **Curate procedures** — run `h2e mine` to extract validated procedures from web documents
2. **Prepare training data** — deduplicate against the test set (see below), then format for open-instruct
3. **Train** — train models on the prepared training data
4. **Evaluate** — run `h2e bench` to evaluate the trained model

### Deduplicating training data against How2Bench

Before training, remove training examples that are too similar to the test set.
The `dedup_against_test.py` script computes embedding similarity between every training example and its nearest test example, then removes any training example above a cosine similarity threshold (default: **0.65**).

```bash
uv run python examples/train/dedup_against_test.py \
    --train-path hf://how2everything/how2train_rl_100k?split=train \
    --test-path hf://how2everything/how2bench?split=train \
    --output-path data/train_deduped.jsonl \
    --train-goal-key goal \
    --train-steps-key steps \
    --test-goal-key goal \
    --test-steps-key steps \
    --threshold 0.65 \
    --save-embeddings data/embeddings.npz
```

**Supported input formats:** JSONL, JSON, CSV, Parquet, and HuggingFace datasets.
For HuggingFace, use the `hf://` prefix:

```bash
uv run python examples/train/dedup_against_test.py \
    --train-path data/train.jsonl \
    --test-path hf://how2everything/how2bench \
    --test-goal-key goal \
    --test-steps-key steps \
    --output-path data/train_deduped.jsonl
```

You can specify a HF split with `hf://repo/name?split=train` (default: `test`).

Field keys support **dot-notation** for nested JSON. For example, if your training
data uses `{"final_procedure": {"goal": ..., "steps": [...]}}`:

```bash
--train-goal-key final_procedure.goal \
--train-steps-key final_procedure.steps
```

This produces:
- `data/train_deduped.jsonl` — the filtered training set
- `data/train_deduped.matches.jsonl` — per-example nearest-test match details (for inspection)
- `data/train_deduped.matches_distribution.png` — histogram of similarity scores

Additional options:
- `--removed-path data/removed.jsonl` — write removed examples for manual review
- `--save-embeddings cache/emb.npz` — cache embeddings to skip re-encoding on future runs
- `--load-embeddings cache/emb.npz` — load cached embeddings instead of re-computing
- `--device cuda` — force GPU (default: auto-detect)

### Training with `open-instruct`

After preparing the deduplicated training data, we use `open-instruct` at commit [72b1b89](https://github.com/allenai/open-instruct/tree/72b1b89ee1a03c0cc8657e01e97c18265b5fa660) to launch training experiments.
See an example training script [here](https://github.com/allenai/open-instruct/blob/yapeic/exp/scripts/train/how2everything/grpo_qwen3-8b-inst.sh).