#!/usr/bin/env bash
# Evaluate the spatial-softmax N=100 UNet runs (seeds 1-3) at every saved
# checkpoint so the result can be compared with the avg arm's already-evaluated
# 10k/30k/60k/100k numbers. 12 evaluations, two at a time, one per GPU.
#SBATCH --job-name=dp-ss32-eval
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/userhome/cs5/u3684238/7606C/slurm-dp-ss32-eval-%j.out

set -uo pipefail
cd /userhome/cs5/u3684238/7606C/dp-manip

RUN_ROOT=/userhome/cs5/u3684238/dp-runs-pickcube
PY=.venv/bin/python
LOGS="$RUN_ROOT/logs/eval"

tasks=()
for CK in step_010000.pt step_030000.pt step_060000.pt final.pt; do
  for S in 1 2 3; do
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
  run="pickcube_rgb_unet_ss32_n100_s${seed}"
  ckpt="$RUN_ROOT/$run/checkpoints/$ck"
  if [ ! -f "$ckpt" ]; then
    echo "[$(date +%H:%M:%S)] SKIP missing $ckpt"
    continue
  fi
  gpu=$(( idx % 2 ))
  echo "[$(date +%H:%M:%S)] gpu$gpu $run $ck"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_dp.py "$ckpt" \
    --split test --num-envs 4 \
    > "$LOGS/${run}.${ck}.log" 2>&1 &
  idx=$(( idx + 1 ))
  if [ $(( idx % 2 )) -eq 0 ]; then
    wait
  fi
done
wait

echo "ALL DONE $(date)"
