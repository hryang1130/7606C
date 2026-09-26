#!/usr/bin/env bash
# Evaluate the core grid on the fixed held-out test seeds using CPU physics and GPU RGB inference/rendering.
# Override CHECKPOINT=step_010000.pt and SPLIT=val for training-curve diagnostics.
#SBATCH --job-name=dp-rgb-eval
#SBATCH --array=0-95%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
RUN_ROOT=${RUN_ROOT:-runs}
PYTHON=${PYTHON:-.venv/bin/python}
CHECKPOINT=${CHECKPOINT:-final.pt}
SPLIT=${SPLIT:-test}
NUM_ENVS=${NUM_ENVS:-4}

"$PYTHON" scripts/sweep.py eval --tier core --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$RUN_ROOT" --checkpoint "$CHECKPOINT" --split "$SPLIT" --num-envs "$NUM_ENVS"
