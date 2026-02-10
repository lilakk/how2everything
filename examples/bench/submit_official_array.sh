#!/usr/bin/env bash
#
# Submit how2bench official benchmark as a Slurm job array.
# Each array task runs exactly one generator model spec from the official config.
#
# Usage (example):
#   PARTITION=gpu-preempt CONSTRAINT=a100-80g GPUS_PER_TASK=4 \
#   LOG_DIR=/path/to/logs \
#   examples/bench/submit_official_array.sh --config examples/bench/configs/official_benchmark.yaml
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="${CONFIG:-}"
if [[ $# -ge 2 && "${1:-}" == "--config" ]]; then
  CONFIG="$2"
  shift 2
fi
if [[ -z "${CONFIG}" ]]; then
  CONFIG="examples/bench/configs/official_benchmark.yaml"
fi

WORK_DIR="${WORK_DIR:-$REPO_ROOT}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/out/how2bench/slurm_logs}"

# SBATCH defaults (override via env vars).
PARTITION="${PARTITION:-gpu-preempt}"
CONSTRAINT="${CONSTRAINT:-a100-80g}"
NODES="${NODES:-1}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-80}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"

# Optional modules: comma-separated list, e.g. "CUDA/12.1.1,uri/main"
MODULES="${MODULES:-}"

DRY_RUN="${DRY_RUN:-0}" # DRY_RUN=1 prints the sbatch script but does not submit.

mkdir -p "$LOG_DIR"

cd "$WORK_DIR"

# Resolve the venv python once on the submission host.
# `uv run --frozen which python` returns the absolute path inside the project venv.
VENV_PYTHON="$(uv run which python)"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: could not resolve venv python via uv (got: $VENV_PYTHON)" >&2
  echo "       Make sure you have run 'uv sync' in the project root." >&2
  exit 2
fi
echo "Resolved venv python: $VENV_PYTHON"

NUM_MODELS="$("$VENV_PYTHON" -m how2everything.bench.slurm.run_one_official_task --config "$CONFIG" --print-num-models)"
if ! [[ "$NUM_MODELS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to read num models from config (got: $NUM_MODELS)" >&2
  exit 2
fi
if [[ "$NUM_MODELS" -le 0 ]]; then
  echo "ERROR: config contains 0 models: $CONFIG" >&2
  exit 2
fi

ARRAY_MAX=$((NUM_MODELS - 1))
CPUS=$((GPUS_PER_TASK * CPUS_PER_GPU))
MEM=$((GPUS_PER_TASK * MEM_PER_GPU_GB))

JOB_NAME="${JOB_NAME:-how2bench-official}"

SBATCH_SCRIPT="$(mktemp)"
trap 'rm -f "$SBATCH_SCRIPT"' EXIT

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --job-name=${JOB_NAME}"
  [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=${PARTITION}"
  [[ -n "$CONSTRAINT" ]] && echo "#SBATCH --constraint=${CONSTRAINT}"
  echo "#SBATCH --nodes=${NODES}"
  echo "#SBATCH --gpus=${GPUS_PER_TASK}"
  echo "#SBATCH --cpus-per-task=${CPUS}"
  echo "#SBATCH --mem=${MEM}G"
  echo "#SBATCH --time=${TIME_LIMIT}"
  echo "#SBATCH --array=0-${ARRAY_MAX}"
  echo "#SBATCH --output=${LOG_DIR}/%x-%A_%a.out"
  echo "#SBATCH --error=${LOG_DIR}/%x-%A_%a.err"
  echo
  echo "set -euo pipefail"
  echo
  # Forward API keys to Slurm tasks (for API-based generator/evaluator models).
  for _key in OPENAI_API_KEY ANTHROPIC_API_KEY GEMINI_API_KEY GOOGLE_API_KEY HF_TOKEN; do
    if [[ -n "${!_key:-}" ]]; then
      echo "export ${_key}=$(printf %q "${!_key}")"
    fi
  done
  echo
  echo "cd \"$(printf %q "$WORK_DIR")\""
  echo
  if [[ -n "$MODULES" ]]; then
    echo "# Modules"
    echo "module purge || true"
    IFS=',' read -r -a mods <<< "$MODULES"
    for m in "${mods[@]}"; do
      m="$(echo "$m" | xargs)"
      [[ -n "$m" ]] && echo "module load $(printf %q "$m")"
    done
    echo
  fi
  echo "echo \"SLURM_JOB_ID: \${SLURM_JOB_ID:-N/A}\""
  echo "echo \"SLURM_ARRAY_TASK_ID: \${SLURM_ARRAY_TASK_ID:-N/A}\""
  echo "echo \"Host: \$(hostname)\""
  echo "echo \"Start: \$(date)\""
  echo "nvidia-smi || true"
  echo
  echo "\"$(printf %q "$VENV_PYTHON")\" -m how2everything.bench.slurm.run_one_official_task \\"
  echo "  --config \"$(printf %q "$CONFIG")\" \\"
  echo "  --index \"\${SLURM_ARRAY_TASK_ID}\""
  echo
  echo "echo \"End: \$(date)\""
} > "$SBATCH_SCRIPT"

echo "Config: $CONFIG"
echo "Models: $NUM_MODELS (array 0-$ARRAY_MAX)"
echo "Work dir: $WORK_DIR"
echo "Log dir: $LOG_DIR"
echo "GPUs/task: $GPUS_PER_TASK, CPUs/task: $CPUS, mem: ${MEM}G, time: $TIME_LIMIT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "=== DRY_RUN=1 (sbatch script) ==="
  cat "$SBATCH_SCRIPT"
  exit 0
fi

sbatch "$SBATCH_SCRIPT"

