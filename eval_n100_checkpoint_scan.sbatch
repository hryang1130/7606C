#!/usr/bin/env bash
# Diagnostic: evaluate the saved intermediate checkpoints of the PickCube
# N=100 UNet runs (seeds 1-5) so we can compare final.pt against the best of
# {10k, 30k, 60k, 100k}. Each eval writes <run>/eval/test_step_<N>.json and
# never touches the existing test_final.json.
#
# 15 evaluations total (5 seeds x 3 checkpoints), run two at a time, one per GPU.
#SBATCH --job-name=dp-n100-ckpt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-dp-n100-ckpt-%j.out

set -uo pipefail
cd /userhome/cs5/u3684238/7606C/dp-manip

RUN_ROOT=/userhome/cs5/u3684238/dp-runs-pickcube
PY=.venv/bin/python
LOGS="$RUN_ROOT/logs/eval"

tasks=()
for CK in step_010000.pt step_030000.pt step_060000.pt; do
  for S in 1 2 3 4 5; do
    tasks+=("$CK $S")
  done
done

echo "queued ${#tasks[@]} evaluations at $(date)"

idx=0
for task in "${tasks[@]}"; do
  # shellcheck disable=SC2086
  set -- $task
  ck=$1
  seed=$2
  run="pickcube_rgb_unet_n100_s${seed}"
  gpu=$(( idx % 2 ))
  echo "[$(date +%H:%M:%S)] gpu$gpu $run $ck"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_dp.py \
    "$RUN_ROOT/$run/checkpoints/$ck" \
    --split test \
    --num-envs 4 \
    > "$LOGS/${run}.${ck}.log" 2>&1 &
  idx=$(( idx + 1 ))
  # Two evaluations in flight, one per GPU.
  if [ $(( idx % 2 )) -eq 0 ]; then
    wait
  fi
done
wait

echo "ALL DONE $(date)"
touch "$RUN_ROOT/logs/eval/N100_SCAN_COMPLETE"
