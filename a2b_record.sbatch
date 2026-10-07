#!/usr/bin/env bash
# Record the closed-loop gripper trace for a UNet and a Transformer checkpoint
# (same evaluation as eval_dp.py, plus a per-step observer). One GPU, two runs
# in sequence; output goes to <run>/gripper_trace_test_final.json.
#SBATCH --job-name=dp-a2b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/userhome/cs5/u3684238/7606C/slurm-dp-a2b-%j.out

set -uo pipefail
cd /userhome/cs5/u3684238/7606C/dp-manip

RUN_ROOT=/userhome/cs5/u3684238/dp-runs-pickcube
PY=.venv/bin/python

for run in pickcube_rgb_unet_n100_s1 pickcube_rgb_transformer_n100_s1; do
  echo "== $run"
  "$PY" /userhome/cs5/u3684238/7606C/a2b_record.py \
    "$RUN_ROOT/$run/checkpoints/final.pt" \
    --episodes 24 \
    --output "$RUN_ROOT/$run/gripper_trace_test_final.json" \
    || { echo "FAILED $run"; exit 1; }
done

echo "ALL DONE $(date)"
