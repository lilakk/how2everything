# 🎯 How2Bench

Evaluate LLMs on step-by-step procedure generation. Given a goal, a model generates a candidate procedure (L2) which is judged against a reference procedure (L1) using a rubric-based evaluation.

The primary metric is **How2Score** = percentage of examples with zero critical failures (range 0-100%, higher is better).

| Stage | Command | What it does |
|---|---|---|
| **run** | `h2e bench run` | Run gen → judge → aggregate end-to-end |
| **gen** | `h2e bench gen` | Generate candidate procedures (L2) for each reference (L1) |
| **judge** | `h2e bench judge` | Judge L2 against L1 using the rubric in `prompts/judge.txt` |
| **validate** | `h2e bench validate` | Validate a config file without running anything |
| **leaderboard** | `h2e bench leaderboard` | Compare multiple models (see [Leaderboard](#leaderboard)) |

## Quickstart

The recommended way to run How2Bench is with [`official_benchmark.yaml`](configs/official_benchmark.yaml) on a Slurm cluster. This evaluates local models with vLLM and judges with the default How2Judge (8B).

`submit_official_array.sh` submits a **Slurm job array** with **one task per generator model** in your config. Each task runs **gen → judge → aggregate** and writes outputs under `<out_root>/<run_name>_<generator_id>/`.

```bash
mkdir -p out/how2bench/slurm_logs
chmod +x examples/bench/submit_official_array.sh

# Preview the Slurm job without submitting
DRY_RUN=1 examples/bench/submit_official_array.sh \
  --config examples/bench/configs/official_benchmark.yaml

# Submit (one Slurm task per model in the config)
PARTITION=gpu-preempt \
CONSTRAINT=a100-80g \
GPUS_PER_TASK=1 \
CPUS_PER_GPU=8 \
MEM_PER_GPU_GB=80 \
TIME_LIMIT=2:00:00 \
LOG_DIR="out/how2bench/slurm_logs" \
  examples/bench/submit_official_array.sh \
  --config examples/bench/configs/official_benchmark.yaml
```

Specify the models you want to evaluate in the yaml file.

<details>
<summary>Slurm internals</summary>

- The array tasks run:
  - `<venv_python> -m how2everything.bench.slurm.run_one_official_task --config ... --index $SLURM_ARRAY_TASK_ID`
  - `<venv_python>` is the absolute path resolved at submission time via `uv run which python`
- The helper uses the same config parsing and pipeline code as `h2e bench run`, but runs **exactly one** model spec.

</details>

## Leaderboard

Print evaluation results across all models you've evaluated:

```bash
# CSV to stdout (copy-paste into a spreadsheet)
uv run h2e bench leaderboard --generations-root out/how2bench/generations/official

# Filter by judge
uv run h2e bench leaderboard --generations-root out/how2bench/generations/official --judge "<model>_<judge_id>"

# Pretty terminal table
uv run h2e bench leaderboard --generations-root out/how2bench/generations/official --pretty

# Both pretty table and CSV file
uv run h2e bench leaderboard --generations-root out/how2bench/generations/official --pretty -o leaderboard.csv
```

`--judge` is optional; if omitted, it aggregates all judge folders it finds.

## Inputs

By default, how2bench reads the released benchmark from Hugging Face: `how2everything/how2bench`.

Supported input formats:
- **how2mine export**: records with `final_procedure` and `source_example`
- **bench JSONL**: records with `source_example_id`, `goal`, `steps`, and `resources`

Override with a local file via `inputs.path` in your config.

## Config format

Both the generator and the evaluator (judge) support two backends: **vLLM** for local/open models and **`lm-deluge`** for API models (OpenAI, Anthropic, Gemini). You can mix and match — e.g., generate with a local model and judge with an API model, or vice versa.

See [`official_benchmark.yaml`](configs/official_benchmark.yaml) for a full example.
Below are some canonical setups for different types of models (base, instruction-tuned, and thinking).

### Generator config

```yaml
out_root: out/how2bench/generations/official

generator_defaults:  # shared defaults for all models
  backend: vllm
  temperature: 0.0
  max_new_tokens: 4096

# list of generator models to evaluate; can override default hyperparams
models:
  - model: allenai/Olmo-3-1025-7B
    run_name: olmo3-1025-7b-stage1-step566000
    prompt_style: base
    vllm:
      engine_kwargs:
        revision: stage1-step566000
      sampling_kwargs:
        stop: ["\n\n"]  # for base models, we stop at double newline to prevent endless repetitions

  - model: Qwen/Qwen3-8B
    run_name: qwen3-8b-no-thinking
    prompt_style: inst
    vllm:
      mode: chat
      chat_template_kwargs:
        enable_thinking: false  # if not set, this defaults to true

  - model: Qwen/Qwen3-8B
    run_name: qwen3-8b-with-thinking  # for thinking models, do not use greedy decoding
    prompt_style: inst
    temperature: 0.6
    vllm:
      mode: chat
      sampling_kwargs:
        top_p: 0.95
        top_k: 20
        min_p: 0.0
```

For API-based models, set `backend: deluge` and `provider`:

```yaml
models:
  - model: gpt-4.1
    provider: openai
    backend: deluge
```

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

#### Multi-GPU vLLM

To run a big open model with vLLM, set in your `generator_defaults:` block:

```yaml
generator_defaults:
  backend: vllm
  temperature: 0.0
  max_new_tokens: 4096
  vllm:
    engine_kwargs:
      # Set this to match Slurm GPUs per task (e.g. GPUS_PER_TASK=4).
      tensor_parallel_size: 4
```

Notes:
- You can pass *any* vLLM engine kwargs via `engine_kwargs` (they are forwarded to `vllm.LLM(**kwargs)`).
- If your cluster requires it, set `HF_TOKEN` (used by vLLM to download gated models).
- In `official_benchmark.yaml`, you can set prompt style per generator entry with `models[].prompt_style: base|inst`.
  This lets you safely mix base and instruct models in one job array without ambiguity.

### Evaluator config

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

By default, how2bench uses **`how2everything/how2judge`** (8B) via **vLLM** for judging. No `evaluator:` config is needed — this is the recommended setup.

To use an API judge instead (e.g. for a portable run without vLLM), set `evaluator:` in your config. See `configs/bench_sync.yaml` for an example.

## Generation

Generation uses prompts from `prompts/`:
- `prompts/inference_base.txt` when `prompt_style: base`
- `prompts/inference_inst.txt` when `prompt_style: inst`

These are resolved automatically; override via `prompts.*` in config if needed.

### Step extraction

After generation, how2bench parses the raw model completion into a list of discrete steps (`predicted_steps`). The heuristic works as follows:

1. **Strip thinking/answer tags** — if the output contains `</think>` (chain-of-thought models like Qwen3, DeepSeek-R1), everything before the last `</think>` is discarded. If `<answer>...</answer>` tags are present, only the content between them is kept.
2. **Numbered-list matching** — each line is tested against the pattern `^\d+[).:-]? ...` (e.g. `1. Do X`, `2) Do Y`). Matched lines become steps.
3. **Fallback** — if no numbered lines are found, every non-empty line is treated as a step.

The raw completion is always preserved in `model_completion` so no information is lost.

## Judging

### Judge-only mode

To judge an existing `generations.jsonl` without re-generating, use `paths.generations`:

```yaml
out_root: out/how2bench/generations/gpt-4.1

paths:
  generations: out/how2bench/generations/gpt-4.1/generations.jsonl

evaluator:
  backend: deluge
  provider: openai
  model: gpt-4.1
```

See `configs/judge_existing_generations.yaml` for a full example.

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

### Output schema

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

## Resumability

Both generation and judging are **append-only and resumable**. Re-running `h2e bench run` on the same config is safe:

- If `generations.jsonl` already contains all examples, generation is **skipped entirely** (the generator model is not loaded).
- If `judgments.jsonl` already contains judgments for all generations, judging is **skipped entirely** (the judge model is not loaded).
- Aggregation always re-runs (it is instant) so `summary.json` stays up to date.

You can interrupt a run and resume it later, or re-run to pick up only missing examples, without redundant work or GPU allocation.

## Run identity

how2bench computes stable hashes to avoid mixing outputs from different configurations:

- **`generator_id`**: hash of resolved generator config + generation prompt SHA-256. Used in the output directory name and written to `generation_manifest.json`.
- **`judge_id`**: hash of resolved evaluator config + judge prompt SHA-256. Used in the judgments subdirectory name and written to `judge_manifest.json` and every judgment record.
