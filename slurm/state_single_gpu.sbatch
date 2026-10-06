#!/usr/bin/env bash
# One state N=100 run followed by validation and test, using the shared queues.
# Submit from the repository root; DATA_ROOT must point to demogen data/dataset.
#SBATCH --job-name=dp-state-n100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${DATA_ROOT:?export DATA_ROOT=<maniskill-demogen data/dataset>}"
TASK=${TASK:-peginsertionside}
RUN_ROOT=${RUN_ROOT:-runs}
PYTHON=${PYTHON:-.venv/bin/python}
EXPERIMENT=configs/experiments/state_n100.toml
stage_pid=""
interrupted=0
trap 'interrupted=1; if [[ -n "$stage_pid" ]]; then kill -USR1 "$stage_pid" 2>/dev/null || true; fi' USR1 TERM

run_stage() {
  if [[ $interrupted -eq 1 ]]; then return 75; fi
  "$@" &
  stage_pid=$!
  if [[ $interrupted -eq 1 ]]; then kill -USR1 "$stage_pid" 2>/dev/null || true; fi
  local status
  while true; do
    wait "$stage_pid"
    status=$?
    if [[ $interrupted -eq 1 && ( $status -eq 138 || $status -eq 143 ) ]]; then continue; fi
    break
  done
  stage_pid=""
  if [[ $interrupted -eq 1 && $status -eq 0 ]]; then return 75; fi
  return "$status"
}

finish_stage() {
  local status=$1
  if [[ $status -eq 75 ]]; then
    echo "stage interrupted; requeueing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
    echo "ERROR: scontrol requeue failed; resubmit the same command to continue" >&2
    exit 1
  fi
  if [[ $status -ne 0 ]]; then exit "$status"; fi
}

echo "state N=100 seed=1 task=$TASK run_root=$RUN_ROOT data_root=$DATA_ROOT"
run_stage "$PYTHON" scripts/train_queue.py \
  --task "$TASK" --experiment "$EXPERIMENT" --workers 1 \
  --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"
finish_stage "$?"

for split in val test; do
  run_stage "$PYTHON" scripts/eval_queue.py \
    --task "$TASK" --experiment "$EXPERIMENT" --workers 1 \
    --output-root "$RUN_ROOT" \
    --checkpoint final.pt --split "$split" --num-envs 4
  finish_stage "$?"
done
