# 🎯 How2Bench

Evaluate LLMs on step-by-step procedure generation. Given a goal, a model generates a candidate procedure which is judged against a reference procedure using a rubric-based evaluation.

The primary metric is **How2Score** = percentage of examples with zero critical failures (range 0-100%, higher is better).

| Stage | Command | What it does |
|---|---|---|
| **run** | `h2e bench run` | Run gen → judge → aggregate end-to-end |
| **gen** | `h2e bench gen` | Generate procedures from models |
| **judge** | `h2e bench judge` | Evaluate model outputs with the How2Score eval protocol |
| **validate** | `h2e bench validate` | Validate a config file without running anything |
| **leaderboard** | `h2e bench leaderboard` | View results (see [Leaderboard](#leaderboard)) |

## Quickstart

Run evaluations end-to-end (gen → judge) then view results in terminal:

```bash
uv run h2e bench run --config examples/bench/configs/official_benchmark.yaml
uv run h2e bench leaderboard --generations-root out/how2bench --pretty
```

By default, we use the 8B How2Judge via vLLM for evaluation. For a quick run **without a GPU**, see [`api_only.yaml`](configs/api_only.yaml), which runs generation and judging through API models only.

To support evaluating many models in parallel, [`official_benchmark.yaml`](configs/official_benchmark.yaml) pairs with `submit_official_array.sh`, which submits a **Slurm job array** with **one task per generator model**. Each task runs **gen → judge → aggregate** and writes outputs under `<out_root>/<run_name>_<generator_id>/`.

```bash
mkdir -p out/how2bench/slurm_logs
chmod +x examples/bench/submit_official_array.sh

# Preview the Slurm jobs without submitting
DRY_RUN=1 examples/bench/submit_official_array.sh \
  --config examples/bench/configs/official_benchmark.yaml

# Submit
PARTITION=gpu-preempt \
CONSTRAINT=a100-80g \
CPUS_PER_GPU=8 \
TIME_LIMIT=2:00:00 \
LOG_DIR="out/how2bench/slurm_logs" \
  examples/bench/submit_official_array.sh \
  --config examples/bench/configs/official_benchmark.yaml
```

The script reads `tensor_parallel_size` from each model's config and automatically groups models by GPU count, submitting one Slurm job array per group. For example, if your config has three 1-GPU models and two 4-GPU models, it submits two arrays: one with `--gpus=1` and one with `--gpus=4`.

<details>
<summary>Slurm internals</summary>

- The script calls `--print-gpu-groups` to get a JSON mapping of `gpu_count → [model_indices]`, then submits one `sbatch` per group.
- Each array task runs:
  - `<venv_python> -m how2everything.bench.slurm.run_one_official_task --config ... --index $SLURM_ARRAY_TASK_ID`
  - `<venv_python>` is the absolute path resolved at submission time via `uv run which python`
- The helper uses the same config parsing and pipeline code as `h2e bench run`, but runs **exactly one** model spec.

</details>

## Inputs

By default, how2bench reads the released benchmark from Hugging Face: `how2everything/how2bench`.

Supported input formats:
- **how2mine export**: records with `final_procedure` and `source_example`
- **bench JSONL**: records with `source_example_id`, `goal`, `steps`, and `resources`

Override with a local file via `inputs.path` in your config.

## Config format

Both the generator and the evaluator (judge) support two backends: **vLLM** for local/open models and **`lm-deluge`** for API models (OpenAI, Anthropic, Gemini). You can mix and match — e.g., generate with a local model and judge with an API model, or vice versa.

### Generator config

See [`official_benchmark.yaml`](configs/official_benchmark.yaml) for canonical generation configs for different types of models (base, instruction-tuned, thinking, and API models).

**Model entry fields:**

| Field | Required | Default | Notes |
|---|---|---|---|
| `model` | yes | — | Model name |
| `provider` | when `backend: deluge` | — | `openai`, `anthropic`, `gemini`, etc. |
| `run_name` | no | sanitized model name | Custom output directory name |
| `prompt_style` | yes | — | `base` or `inst`. Controls which generation prompt template is used. Can be set per-model or in `generator_defaults`. |
| `backend` | no | `deluge` | `deluge` (API) or `vllm` (local) |
| `temperature` | no | `0.0` | |
| `max_new_tokens` | no | `4096` | |
| `vllm` | no | — | vLLM-specific overrides (merged with `generator_defaults.vllm`). Sub-fields: `mode` (`generate`\|`chat`), `engine_kwargs` (forwarded to `vllm.LLM(**)`), `sampling_kwargs` (forwarded to `SamplingParams`, e.g. `top_p`, `top_k`, `min_p`, `stop`), `chat_template_kwargs` (e.g. `enable_thinking`). |
| `max_requests_per_minute` | no | `1000` | API-models only |
| `max_tokens_per_minute` | no | `100000` | API-models only |
| `reasoning_effort` | no | — | API-models only. `lm-deluge` maps this to provider-native formats (OpenAI `reasoning_effort`, Anthropic `thinking.budget_tokens`, Gemini `thinkingLevel`/`thinkingBudget`). Values: `none`, `minimal`, `low`, `medium`, `high` |

Per-model fields override `generator_defaults`.

<details>
<summary>Parsing model outputs (important for models that output reasoning)</summary>

After generation, how2bench parses the raw model completion into a list of discrete steps (`predicted_steps`). The heuristic works as follows:

1. **Strip thinking/answer tags** — if the output contains `</think>` (chain-of-thought models like Qwen3, DeepSeek-R1), everything before the last `</think>` is discarded. If `<answer>...</answer>` tags are present, only the content between them is kept.
2. **Numbered-list matching** — each line is tested against the pattern `^\d+[).:-]? ...` (e.g. `1. Do X`, `2) Do Y`). Matched lines become steps.
3. **Fallback** — if no numbered lines are found, every non-empty line is treated as a step.

The raw completion is always preserved in `model_completion` so no information is lost.

</details>


#### Multi-GPU vLLM

For models that need multiple GPUs, set `tensor_parallel_size` in the `vllm.engine_kwargs` block — either per-model or in `generator_defaults` (per-model values are deep-merged on top of defaults):

```yaml
# Per-model override (only this model uses 4 GPUs)
models:
  - model: meta-llama/Llama-4-Scout-17B-16E-Instruct
    prompt_style: inst
    vllm:
      mode: chat
      engine_kwargs:
        tensor_parallel_size: 4

# Or set as default for all models
generator_defaults:
  vllm:
    engine_kwargs:
      tensor_parallel_size: 4
```

### Evaluator config

By default, we use the 8B How2Judge via vLLM, see [`official_benchmark.yaml`](configs/official_benchmark.yaml). To use an API judge instead (e.g. for a portable run without vLLM), set `evaluator:` in your config. See [`api_only.yaml`](configs/api_only.yaml) for a full end-to-end config that only uses API models.

**Evaluator fields** (under `evaluator:`):

| Field | Default | Notes |
|---|---|---|
| `backend` | `vllm` | `deluge` or `vllm` |
| `provider` | — | Required when `backend: deluge` |
| `model` | `how2everything/how2judge` | |
| `temperature` | `0.0` | |
| `max_new_tokens` | `4096` | |
| `max_requests_per_minute` | `1000` | API-models only |
| `max_tokens_per_minute` | `100000` | API-models only |
| `reasoning_effort` | — | API-models only |

To re-judge existing generations with a different judge, simply change `evaluator:` and re-run — generation is skipped automatically and the new judgments are written to a separate subdirectory. Use `paths.generations` only if the generations file lives outside of `out_root` (see [`judge_existing_generations.yaml`](configs/judge_existing_generations.yaml)).

## Outputs

For each model, how2bench writes to `<out_root>/<run_name>_<generator_id>/`:

```
generations.jsonl
generation_manifest.json
judgments/
  <model>_<judge_id>/
    judgments.jsonl
    judge_manifest.json
    aggregate/
      summary.json
      by_topic.csv
```

<details>
<summary>Output schema</summary>

**`generations.jsonl`** — one record per benchmark example:

```json
{
  "schema_version": "how2bench.generation.v1",
  "generator_id": "abc123",
  "source_example_id": "doc-042",
  "topic": "Food & Dining",
  "goal": "Make a simple pasta carbonara",
  "steps": ["Boil water and cook pasta", "Fry pancetta", "Mix eggs and cheese", "Combine pasta with sauce"],
  "resources": ["pasta", "eggs", "pancetta", "pecorino cheese"],
  "model_completion": "1. Boil salted water and cook spaghetti al dente.\n2. Fry pancetta until crispy.\n3. Whisk eggs and grated pecorino.\n4. Toss hot pasta with pancetta, then off heat add egg mixture.",
  "predicted_steps": ["Boil salted water and cook spaghetti al dente.", "Fry pancetta until crispy.", "..."],
  "prompt": "...",
  "generator": { "backend": "deluge", "provider": "openai", "model": "gpt-4.1" }
}
```

**`judgments.jsonl`** — one record per judged generation:

```json
{
  "schema_version": "how2bench.judgment.v1",
  "judge_id": "xyz789",
  "source_example_id": "doc-042",
  "topic": "Food & Dining",
  "goal": "Make a simple pasta carbonara",
  "steps": ["Boil water and cook pasta", "..."],
  "predicted_steps": ["Boil salted water and cook spaghetti al dente.", "..."],
  "reasoning": "The predicted steps correctly follow the reference procedure...",
  "critical_failures": [],
  "has_failure": false,
  "n_failures": 0,
  "parse_failed": false,
  "judge": { "backend": "vllm", "provider": null, "model": "how2everything/how2judge" }
}
```

**`summary.json`** — aggregate metrics:

```json
{
  "how2score": 0.85,
  "how2score_percent": 85.0,
  "n_examples": 7000,
  "n_with_failures": 1050,
  "failure_rate": 0.15,
  "avg_failures_per_example": 0.22
}
```

**`by_topic.csv`** — per-topic breakdown with columns: `topic`, `n_judged`, `n_with_failures`, `how2score`, `how2score_percent`, `failure_rate`, `avg_failures_per_example`.

</details>

## Leaderboard

To view results:

```bash
uv run h2e bench leaderboard --generations-root out/how2bench
```

| Option | Description |
|---|---|
| `--pretty` | Print a formatted terminal table instead of CSV |
| `-o leaderboard.csv` | Write results to a file |
| `--judge "<model>_<judge_id>"` | Filter to a specific judge (default: aggregate all) |

## Resumability and run identity

Re-running `h2e bench run` on the same config is safe — both generation and judging are **append-only**. Completed examples are skipped (the model is not loaded if all outputs already exist), so you can interrupt and resume without redundant work or GPU allocation.

How2Bench uses stable hashes (`generator_id`, `judge_id`) derived from the resolved config + prompt contents to keep outputs from different configurations separate. These hashes appear in output directory names and manifest files.
