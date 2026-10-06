#!/usr/bin/env bash
# Train one task of a sweep on both GPUs of a single Slurm job.
#
# The QOS allows every user only one submitted job, so this entry point never
# uses a job array. It hands the task's declared runs to scripts/train_queue.py,
# which skips runs whose final.pt already matches and feeds the rest to two
# workers (one GPU each) from a shared queue. Re-submitting the same command is
# safe: finished runs are skipped and interrupted runs resume from resume.pt.
#
# TASK selects the task, EXPERIMENT selects the grid (default: data-size).
# Submit from the repository root:
#   sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT=/scratch/$USER/dp-data/dataset,RUN_ROOT=/scratch/$USER/dp-runs slurm/train_dual_gpu.sbatch
#   sbatch --export=ALL,TASK=pickcube,EXPERIMENT=configs/experiments/backbone.toml,DATA_ROOT=...,RUN_ROOT=... slurm/train_dual_gpu.sbatch
#
# When the queue reports interrupted runs (exit 75: a trainer checkpointed
# after the preemption signal, or runs were never started), this script
# requeues the job instead of failing it so the next allocation continues.
# If scontrol requeue itself fails, the job exits non-zero with a message
# instead of pretending the interrupted runs were rescheduled.
#
# Extra arguments after the script name are forwarded to scripts/train_queue.py
# as-is, which is how a reduced-budget smoke adds --set overrides:
#   sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/smoke.toml,DATA_ROOT=...,RUN_ROOT=... \
#     slurm/train_dual_gpu.sbatch --set train.total_iters=200 \
#     --set train.validation_steps=[200] --set train.checkpoint_steps=[200]
#
# --time is the default debug partition's MaxTime (18 h); a longer request
# stays pending with PartitionTimeLimit. A grid that needs longer is not
# lost: USR1 at walltime - 120 s checkpoints the runs and the job requeues.
#SBATCH --job-name=dp-rgb-train-dual
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${TASK:?export TASK=<task name, for example peginsertionside>}"
: "${DATA_ROOT:?export DATA_ROOT=<maniskill-demogen data/dataset>}"
EXPERIMENT=${EXPERIMENT:-configs/experiments/data_size.toml}
RUN_ROOT=${RUN_ROOT:-runs}
PYTHON=${PYTHON:-.venv/bin/python}

echo "task=$TASK experiment=$EXPERIMENT run_root=$RUN_ROOT data_root=$DATA_ROOT python=$PYTHON"

"$PYTHON" scripts/train_queue.py \
  --task "$TASK" \
  --experiment "$EXPERIMENT" \
  --data-root "$DATA_ROOT" \
  --output-root "$RUN_ROOT" \
  "$@" &
training_pid=$!
trap 'kill -USR1 "$training_pid" 2>/dev/null || true' USR1 TERM
while true; do
  wait "$training_pid"
  status=$?
  # A trapped scheduler signal interrupts wait before the queue has drained the
  # running trainers. Wait again and use the queue's eventual exit status.
  if [[ $status -eq 138 || $status -eq 143 ]]; then
    continue
  fi
  break
done
if [[ $status -eq 75 ]]; then
  echo "queue checkpointed interrupted runs; requeueing job $SLURM_JOB_ID"
  if scontrol requeue "$SLURM_JOB_ID"; then
    exit 0
  fi
  echo "ERROR: scontrol requeue failed; resubmit the same command to continue" >&2
  exit 1
fi
exit "$status"
