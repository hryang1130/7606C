#!/usr/bin/env bash
# Conditional N=400 tier: submit only for tasks whose N=100 -> 200 gain has not saturated.
# The complete tier is 6 tasks × 5 seeds = 30; narrow --array when only some tasks qualify.
#SBATCH --job-name=dp-rgb-n400
#SBATCH --array=0-29%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${DATA_ROOT:?export DATA_ROOT=<maniskill-demogen data/dataset>}"
RUN_ROOT=${RUN_ROOT:-runs}
PYTHON=${PYTHON:-.venv/bin/python}

"$PYTHON" scripts/sweep.py train --tier optional400 --index "$SLURM_ARRAY_TASK_ID" \
  --data-root "$DATA_ROOT" --output-root "$RUN_ROOT" &
training_pid=$!
trap 'kill -USR1 "$training_pid" 2>/dev/null || true' USR1 TERM
while true; do
  wait "$training_pid"
  status=$?
  if [[ $status -eq 138 || $status -eq 143 ]]; then
    continue
  fi
  break
done
if [[ $status -eq 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
