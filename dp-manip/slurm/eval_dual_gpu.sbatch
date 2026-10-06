#!/usr/bin/env bash
# Evaluate one task's runs on both GPUs of a single Slurm job.
#
# The QOS allows every user only one submitted job, so this entry point never
# uses a job array. It hands the task's runs to scripts/eval_queue.py, which
# runs at most two evaluators (one GPU each) from a shared queue and writes one
# log per run under RUN_ROOT/logs/eval. Re-submitting the same command is safe:
# evaluations are idempotent and overwrite their own result JSON.
#
# TASK selects the task, EXPERIMENT selects the grid (default: data-size).
# CHECKPOINT/SPLIT/NUM_ENVS keep their existing meaning. Submit from the
# repository root:
#   sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT=/scratch/$USER/dp-runs slurm/eval_dual_gpu.sbatch
#   sbatch --export=ALL,TASK=pickcube,CHECKPOINT=step_010000.pt,SPLIT=val,RUN_ROOT=... slurm/eval_dual_gpu.sbatch
#
# When the queue reports interrupted runs (exit 75), this script requeues the
# job instead of failing it; if scontrol requeue itself fails, the job exits
# non-zero with a message instead of pretending the runs were rescheduled.
#SBATCH --job-name=dp-rgb-eval-dual
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${TASK:?export TASK=<task name, for example peginsertionside>}"
EXPERIMENT=${EXPERIMENT:-configs/experiments/data_size.toml}
RUN_ROOT=${RUN_ROOT:-runs}
PYTHON=${PYTHON:-.venv/bin/python}
CHECKPOINT=${CHECKPOINT:-final.pt}
SPLIT=${SPLIT:-test}
NUM_ENVS=${NUM_ENVS:-4}
extra_args=()
if [[ -n "${MAX_EPISODE_STEPS:-}" ]]; then
  extra_args+=(--max-episode-steps "$MAX_EPISODE_STEPS")
fi

echo "task=$TASK experiment=$EXPERIMENT run_root=$RUN_ROOT checkpoint=$CHECKPOINT split=$SPLIT num_envs=$NUM_ENVS python=$PYTHON"

"$PYTHON" scripts/eval_queue.py \
  --task "$TASK" \
  --experiment "$EXPERIMENT" \
  --output-root "$RUN_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --num-envs "$NUM_ENVS" "${extra_args[@]}" &
evaluation_pid=$!
trap 'kill -USR1 "$evaluation_pid" 2>/dev/null || true' USR1 TERM
while true; do
  wait "$evaluation_pid"
  status=$?
  # A trapped scheduler signal interrupts wait before the queue has drained the
  # running evaluators. Wait again and use the queue's eventual exit status.
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
