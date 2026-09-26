#!/usr/bin/env bash
# Core data-quantity grid: 6 tasks × (N=25/50 with 3 seeds + N=100/200 with 5 seeds) = 96.
# Submit from the repository root:
#   sbatch --export=ALL,DATA_ROOT=/scratch/$USER/dp-data/dataset,RUN_ROOT=/scratch/$USER/dp-runs slurm/train_array.sbatch
#SBATCH --job-name=dp-rgb-train
#SBATCH --array=0-95%4
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

"$PYTHON" scripts/sweep.py train --tier core --index "$SLURM_ARRAY_TASK_ID" \
  --data-root "$DATA_ROOT" --output-root "$RUN_ROOT" &
training_pid=$!
trap 'kill -USR1 "$training_pid" 2>/dev/null || true' USR1 TERM
while true; do
  wait "$training_pid"
  status=$?
  # A trapped scheduler signal interrupts wait before the child has finished
  # writing resume.pt. Wait again and use the child's eventual exit status.
  if [[ $status -eq 138 || $status -eq 143 ]]; then
    continue
  fi
  break
done
if [[ $status -eq 75 ]]; then
  echo "training checkpointed after a scheduler signal; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
