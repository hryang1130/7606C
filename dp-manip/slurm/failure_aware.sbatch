#!/usr/bin/env bash
# Run one failure-aware study step on one GPU (docs/failure_aware_finetuning_plan.md §8 Phase 5).
#
# The QOS allows one submitted job per user, so every step is its own job and
# the arguments name the step. Steps are idempotent: collection refuses to
# overwrite raw rollouts, fine-tuning resumes from resume.pt, and closed-loop
# evaluation skips results that already exist. Submit from the repository root.
# Usually the whole study is driven by scripts/failure_pipeline.py (see its
# docstring for the smoke and study commands); single steps also work:
#
#   sbatch slurm/failure_aware.sbatch scripts/collect_rollouts.py collect "$RUN_ROOT/placesphere_rgb_unet_n100_s1/checkpoints/final.pt" --split train --max-episode-steps 200
#   sbatch slurm/failure_aware.sbatch scripts/finetune_dp.py "$CKPT" --label failure --lr 1e-5 --steps 20000 --checkpoint-steps 5000 10000
#   sbatch slurm/failure_aware.sbatch scripts/failure_study.py eval --task placesphere --split tuning --arm A
#
# Collection must use the lock file's [task] max_episode_steps (plan §13);
# record-collection refuses rollouts collected at another horizon.
#
# A fine-tune that checkpointed after the preemption signal exits 75; the job
# is then requeued and continues from resume.pt.
#SBATCH --job-name=dp-failure-aware
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# 18 h is the debug partition's MaxTime; a longer step is requeued and resumes.
#SBATCH --time=18:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PYTHON=${PYTHON:-.venv/bin/python}
if [[ $# -lt 1 ]]; then
  echo "usage: sbatch slurm/failure_aware.sbatch <script.py> [arguments...]" >&2
  exit 2
fi
case "$1" in
  scripts/failure_pipeline.py | scripts/collect_rollouts.py | scripts/finetune_dp.py | scripts/failure_study.py) ;;
  *)
    echo "refusing to run $1: expected a failure-aware study script" >&2
    exit 2
    ;;
esac

echo "python=$PYTHON step: $*"
"$PYTHON" "$@" &
step_pid=$!
trap 'kill -USR1 "$step_pid" 2>/dev/null || true' USR1 TERM
while true; do
  wait "$step_pid"
  status=$?
  # A trapped scheduler signal interrupts wait before the step has finished
  # checkpointing; wait again for its real exit status.
  if [[ $status -eq 138 || $status -eq 143 ]]; then
    continue
  fi
  break
done

if [[ $status -eq 75 ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "$SLURM_JOB_ID"; then
    echo "step interrupted after checkpointing; job $SLURM_JOB_ID requeued"
    exit 0
  fi
  echo "step interrupted (exit 75) but the job could not be requeued; resubmit the same command" >&2
fi
exit "$status"
