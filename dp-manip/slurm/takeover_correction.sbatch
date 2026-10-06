#!/usr/bin/env bash
#SBATCH --job-name=dp-takeover-corr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%A.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
.venv/bin/python -u scripts/takeover_correction.py "$@"
