#!/usr/bin/env bash
#
# Submit how2bench official benchmark as Slurm job arrays.
#
# Models are automatically grouped by GPU requirement (tensor_parallel_size)
# so each group gets its own job array with the right --gpus allocation.
# Set GPUS_PER_TASK to force a single array with a fixed GPU count instead.
#
# Usage (example):
#   PARTITION=gpu-preempt CONSTRAINT=a100-80g \
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
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU_GB="${MEM_PER_GPU_GB:-80}"
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"

# Optional: set GPUS_PER_TASK to force a single array with a fixed GPU count.
# When unset, the script auto-groups models by tensor_parallel_size.
GPUS_PER_TASK="${GPUS_PER_TASK:-}"

# Optional modules: comma-separated list, e.g. "CUDA/12.1.1,uri/main"
MODULES="${MODULES:-}"

DRY_RUN="${DRY_RUN:-0}" # DRY_RUN=1 prints the sbatch script(s) but does not submit.

mkdir -p "$LOG_DIR"

cd "$WORK_DIR"

# Resolve the venv python once on the submission host.
VENV_PYTHON="$(uv run which python)"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ERROR: could not resolve venv python via uv (got: $VENV_PYTHON)" >&2
  echo "       Make sure you have run 'uv sync' in the project root." >&2
  exit 2
fi
echo "Resolved venv python: $VENV_PYTHON"

# Read model count.
NUM_MODELS="$("$VENV_PYTHON" -m how2everything.bench.slurm.run_one_official_task --config "$CONFIG" --print-num-models)"
if ! [[ "$NUM_MODELS" =~ ^[0-9]+$ ]] || [[ "$NUM_MODELS" -le 0 ]]; then
  echo "ERROR: config contains 0 models or failed to parse: $CONFIG (got: $NUM_MODELS)" >&2
  exit 2
fi

# Build GPU groups.
# If GPUS_PER_TASK is set, force all models into one group.
# Otherwise, auto-group by tensor_parallel_size from the config.
if [[ -n "$GPUS_PER_TASK" ]]; then
  # All model indices (0..N-1) in one group with the forced GPU count.
  ALL_INDICES="$(seq -s, 0 $((NUM_MODELS - 1)))"
  GPU_GROUPS_JSON="{\"${GPUS_PER_TASK}\": [${ALL_INDICES}]}"
else
  GPU_GROUPS_JSON="$("$VENV_PYTHON" -m how2everything.bench.slurm.run_one_official_task --config "$CONFIG" --print-gpu-groups)"
fi

echo "Config: $CONFIG"
echo "Models: $NUM_MODELS"
echo "Work dir: $WORK_DIR"
echo "Log dir: $LOG_DIR"

JOB_NAME="${JOB_NAME:-how2bench-official}"

# -----------------------------------------------------------------------
# Helper: generate and submit (or print) one sbatch script.
# Args: $1=gpus_per_task  $2=comma-separated array indices (e.g. "0,1,3")
# -----------------------------------------------------------------------
submit_group() {
  local gpus="$1"
  local indices="$2"
  local cpus=$((gpus * CPUS_PER_GPU))
  local mem=$((gpus * MEM_PER_GPU_GB))

  local sbatch_script
  sbatch_script="$(mktemp)"

  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=${JOB_NAME}"
    [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=${PARTITION}"
    [[ -n "$CONSTRAINT" ]] && echo "#SBATCH --constraint=${CONSTRAINT}"
    echo "#SBATCH --nodes=${NODES}"
    echo "#SBATCH --gpus=${gpus}"
    echo "#SBATCH --cpus-per-task=${cpus}"
    echo "#SBATCH --mem=${mem}G"
    echo "#SBATCH --time=${TIME_LIMIT}"
    echo "#SBATCH --array=${indices}"
    echo "#SBATCH --output=${LOG_DIR}/%x-%A_%a.out"
    echo "#SBATCH --error=${LOG_DIR}/%x-%A_%a.err"
    echo
    echo "set -euo pipefail"
    echo
    # Forward API keys to Slurm tasks.
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
  } > "$sbatch_script"

  echo ""
  echo "--- Group: ${gpus} GPU(s), tasks=[${indices}] ---"
  echo "    GPUs/task: $gpus, CPUs/task: $cpus, mem: ${mem}G, time: $TIME_LIMIT"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "=== DRY_RUN=1 (sbatch script for ${gpus}-GPU group) ==="
    cat "$sbatch_script"
  else
    sbatch "$sbatch_script"
  fi

  rm -f "$sbatch_script"
}

# -----------------------------------------------------------------------
# Iterate over GPU groups and submit one array per group.
# GPU_GROUPS_JSON is e.g. {"1": [0, 1, 3], "4": [2, 4]}
# -----------------------------------------------------------------------

# Parse GPU groups using Python (jq may not be available on all clusters).
while IFS='|' read -r gpus indices; do
  submit_group "$gpus" "$indices"
done < <("$VENV_PYTHON" -c "
import json, sys
groups = json.loads(sys.argv[1])
for gpus, idxs in sorted(groups.items(), key=lambda x: int(x[0])):
    print(f'{gpus}|{','.join(str(i) for i in idxs)}')
" "$GPU_GROUPS_JSON")
