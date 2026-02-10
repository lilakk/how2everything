# ⛏️ How2Mine

How2Mine extracts structured procedures (goal + resources + steps) from raw documents through a 6-stage LLM pipeline that filters and refines for quality. All commands use the `h2e` CLI (installed from this repo as a package).

## Quickstart

1) Set your API key (example for OpenAI):

```bash
export OPENAI_API_KEY="..."
```

2) Validate your config (checks config structure, prompt files, API keys, and vLLM availability):

```bash
uv run h2e mine validate --config examples/mine/configs/openai_sync.yaml
```

3) Run the pipeline:

```bash
uv run h2e mine run --config examples/mine/configs/openai_sync.yaml
```

4) Check progress (works during or after a run):

```bash
uv run h2e mine status --out out/how2mine
```

This shows per-topic pass/total counts for every stage, overall yield, and export file details.

### Sample data

The example configs above use files from `sample_data/`, which contain small datasets for quick testing:

| File | Format | Rows | Description |
|---|---|---|---|
| `sample_split1.arrow` | Arrow | 7,000 | 500 docs per topic (14 topics), sampled from DCLM |
| `sample_split2.arrow` | Arrow | 7,000 | Non-overlapping second split, same structure |

The files use `topic_top_choice` as the topic column (configurable via `inputs.topic_field`).

## Pipeline stages

The pipeline processes documents through 6 stages, then exports the results:

1. **Extract Procedures** (`01_procedures/`) — LLM extracts a goal and sequential steps from each document.
2. **Heuristics Filter** (`02_prefilter/`) — Rule-based filter (no LLM): rejects procedures with too few/many steps or excessive n-gram repetition. Step count bounds are configured via `targets.min_steps` (default 4) and `targets.max_steps` (default 15).
3. **LLM Filter** (`03_filter/`) — LLM rejects low-quality procedures (entity-focused, pure math, UI interactions, non-sequential, etc.).
4. **Postprocess** (`04_postprocess/`) — LLM rewrites the goal and steps for clarity.
5. **Extract Resources** (`05_resources/`) — LLM extracts required tools/materials/resources.
6. **Final Validation** (`06_final_filter/`) — LLM rubric check (correctness, sequentiality, no specific entities, goal-steps alignment). Must pass all criteria.

Passing records are then exported to `data/` as `all_valid.jsonl` (and `all_valid_flat.jsonl`). See [Output schema](#output-schema) below for the exported record format.

Each stage writes per-topic JSONL files. The output directory structure looks like:

```
out_root/
├── 01_procedures/       # Per-topic extraction results
├── 02_prefilter/        # After heuristic filtering
├── 03_filter/           # After LLM filtering
├── 04_postprocess/      # After rewriting
├── 05_resources/        # After resource extraction
├── 06_final_filter/     # After final validation
├── data/                # Exported valid records
│   ├── all_valid.jsonl
│   └── all_valid_flat.jsonl
└── manifest.json        # Run metadata and config snapshot
```

## Configuration

### Input data format

How2Mine expects tabular document data (one row per document) with the following fields:

| Field | Default column name | Required | Description |
|-------|-------------------|----------|-------------|
| **id** | `id` | Yes | Unique document identifier |
| **text** | `text` | Yes | Full document text to extract procedures from |
| **url** | `url` | No | Source URL (metadata only) |
| **topic** | `topic` | No | Topic label (only needed in topics mode) |

Column names are configurable via `inputs.id_field`, `inputs.text_field`, etc. in the YAML config.

**Supported file formats:** JSONL (`.jsonl`), CSV (`.csv`), Arrow (`.arrow`, `.feather`), Parquet (`.parquet`). Compression is auto-detected: `.zst`, `.gz`, `.bz2`, `.xz`.

**Single file or directory:** Set `inputs.path` to a file, or to a directory with `inputs.include_globs` to select files (non-recursive), e.g. `["*.jsonl", "*.arrow"]`.

### Budget and target counts

Pipeline behavior is controlled by two independent settings under `targets`:

- **`candidates_per_topic`** (required) — Max documents to extract per topic. This is always the budget.
- **`desired_valid_per_topic`** (optional) — Target number of final-passing results per topic. If set, the pipeline loops in batches of `extract_batch_size`, running all stages each round, until the target is met or the budget is exhausted. If omitted, the pipeline runs all stages once and you get whatever passes. `extract_batch_size` is only used when `desired_valid_per_topic` is set.

```yaml
# Just a budget: process 1000 docs per topic, take whatever passes
targets:
  candidates_per_topic: 1000

# Budget + yield target: keep going until 10 pass per topic, from up to 1000 docs
targets:
  candidates_per_topic: 1000
  desired_valid_per_topic: 10
  extract_batch_size: 100
```

### Topics mode vs no-topics mode

If you omit `topics` (or set it to `[]`), how2mine runs in **no-topics mode**: it processes all input records (up to `targets.candidates_per_topic` total) and writes a single per-stage file named `all.jsonl`.

When topics are specified, records are filtered by their `topic_field` value, and each topic gets its own output file per stage. The 14 topics used in How2Everything are:

```
Art & Design
Crime & Law
Education & Jobs
Electronics & Hardware
Fashion & Beauty
Food & Dining
Health
Home & Hobbies
Industrial
Religion
Science, Math & Technology
Sports & Fitness
Transportation
Travel & Tourism
```

These topics come from the [WebOrganizer](https://github.com/CodeCreator/WebOrganizer) topic taxonomy. To classify your own documents by topic (and optionally format), use the WebOrganizer topic/format classifiers (`WebOrganizer/TopicClassifier`, `WebOrganizer/FormatClassifier` on Hugging Face).

### Export formats

Configure via `export.format`:

- `jsonl` — Plain JSONL (default)
- `jsonl_zst` — Zstandard-compressed JSONL
- `parquet` — Apache Parquet
- `arrow_ipc` — Arrow IPC

Optional sharding: set `export.max_file_size` (e.g. `"500MB"`) to split output into multiple files.


## Output schema

**`all_valid.jsonl`** — Full records with provenance from every stage:

```json
{
  "schema_version": "how2mine.v1",
  "source_example": {
    "id": "doc-123",
    "text": "...",
    "url": "https://example.com/...",
    "topic": "Food & Dining"
  },
  "processed_example": {
    "procedures":  { "stage_passed": true, "goal": "...", "steps": ["..."] },
    "prefilter":   { "stage_passed": true, "reason": "..." },
    "filter":      { "stage_passed": true, "judgment": true, "reason": "..." },
    "postprocess": { "stage_passed": true, "rewritten_goal": "...", "rewritten_steps": ["..."] },
    "resources":   { "stage_passed": true, "resources": ["..."] },
    "final_filter": {
      "stage_passed": true,
      "correctness": { "answer": "yes", "reason": "..." },
      "sequential":  { "answer": "yes", "reason": "..." },
      "no_specific_entity": { "answer": "yes", "reason": "..." },
      "goal_steps_alignment": { "answer": "yes", "reason": "..." }
    }
  },
  "final_procedure": {
    "goal": "How to make sourdough bread",
    "steps": ["Mix flour and water...", "Let rest for 12 hours...", "..."],
    "resources": ["flour", "water", "salt", "Dutch oven"]
  }
}
```

**`all_valid_flat.jsonl`** — Flattened records for downstream use (evaluation, training):

```json
{
  "source_example_id": "doc-123",
  "topic": "Food & Dining",
  "goal": "How to make sourdough bread",
  "steps": ["Mix flour and water...", "Let rest for 12 hours...", "..."],
  "resources": ["flour", "water", "salt", "Dutch oven"]
}
```

## Resuming

The pipeline is fully resumable. Re-running the same config picks up where it left off:

- Each stage skips records that already appear in its output files (matched by document ID).
- On resume, a summary table is printed showing how many documents have been processed and how many passed final validation per topic.
- When `desired_valid_per_topic` is set, topics that have already reached their target are skipped entirely in subsequent rounds — no new candidates are extracted or processed for them.

## LLM backend configuration

- **API calls** are routed through `lm-deluge` (OpenAI/Anthropic/Gemini).
- Large local models via vLLM are supported where implemented. See `mixed_vllm_extract_openai_rest.yaml`.
  - You can also configure different models/backends per stage via a `stage_llm_overrides:` block in the config (e.g. run `procedures` on vLLM, keep the others on API).
