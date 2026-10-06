#!/usr/bin/env bash
# Mechanism check, then the exchange-rate curve (configs/failure_aware/takeover_curve.toml),
# in one job (one job per user). Stages are write-once/resumable; the job ends
# after analyze and submits nothing else.
#SBATCH --job-name=dp-takeover-curve
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%x-%A.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
OUT=${1:?output directory}
run() { echo "== $* $(date -Is)"; .venv/bin/python -u scripts/takeover_curve.py "$@" --output "$OUT"; }
echo "== tests $(date -Is)"
.venv/bin/python -m unittest tests.test_takeover_mechanism tests.test_takeover_data tests.test_takeover
for s in s1 s2 s3; do
  for arm in B C D; do run mechanism --checkpoint "$s" --arm "$arm"; done
done
for s in s1 s2 s3; do
  run build --checkpoint "$s"
  for arm in C25 C50 D150 D300; do
    run finetune --checkpoint "$s" --arm "$arm"
    run evaluate --checkpoint "$s" --arm "$arm"
  done
  df -h "$HOME" | tail -1
done
run analyze
