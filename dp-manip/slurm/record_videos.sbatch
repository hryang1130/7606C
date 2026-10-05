#!/usr/bin/env bash
# Record 1080p success/failure rollout videos for one or more checkpoints.
#
# One GPU, checkpoints in sequence (the QOS allows one submitted job per user).
# Each checkpoint is checked against its saved evaluation; see
# scripts/record_rollout_videos.py. The job stops at the first checkpoint that
# fails, so a mismatch is never followed by more recording.
#
# CHECKPOINTS is a space-separated list of final.pt paths. SPLIT, SUCCESS and
# FAILURE map to the script's flags; EXTRA_ARGS is passed through unchanged
# (for example "--max-episode-steps 200" or "--shader rt-fast"). Submit from
# the repository root:
#   sbatch --export=ALL,CHECKPOINTS="$HOME/dp-runs/pickcube_rgb_unet_n100_s1/checkpoints/final.pt" slurm/record_videos.sbatch
#SBATCH --job-name=dp-rgb-videos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%A.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${CHECKPOINTS:?export CHECKPOINTS=<space-separated final.pt paths>}"
PYTHON=${PYTHON:-.venv/bin/python}
SPLIT=${SPLIT:-test}
SUCCESS=${SUCCESS:-3}
FAILURE=${FAILURE:-3}
EXTRA_ARGS=${EXTRA_ARGS:-}

echo "split=$SPLIT success=$SUCCESS failure=$FAILURE extra=[$EXTRA_ARGS] python=$PYTHON"
for checkpoint in $CHECKPOINTS; do
  echo "== $checkpoint"
  # shellcheck disable=SC2086  # EXTRA_ARGS is a flag list on purpose
  "$PYTHON" scripts/record_rollout_videos.py "$checkpoint" \
    --split "$SPLIT" --success "$SUCCESS" --failure "$FAILURE" $EXTRA_ARGS
done
