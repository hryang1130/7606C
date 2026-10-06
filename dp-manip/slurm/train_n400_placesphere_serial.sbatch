#!/usr/bin/env bash
# PlaceSphere N=400 data-size tier, five seeds one after another in one job
# (the QOS allows one job per user, so slurm/train_optional400_array.sbatch
# cannot run as an array). Each run trains exactly like the core grid, then is
# evaluated on val (50) and test (100) at horizon 200 like the N<=200 runs
# (eval/{val,test}_final_h200.json).
#
# Disk (approved 2026-10-05): first remove resume.pt of the finished
# takeover-correction fine-tunes, then each N=400 run's resume.pt once its
# final.pt exists. final.pt and the intermediate checkpoints are kept.
#SBATCH --job-name=dp-n400-serial
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=slurm-%x-%A.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
DATA_ROOT=${DATA_ROOT:-/userhome/cs5/u3684139/maniskill-demogen/data/dataset}
RUN_ROOT=${RUN_ROOT:-/userhome/cs5/u3684139/dp-runs-placesphere}
CORRECTION=${CORRECTION:-/userhome/cs5/u3684139/dp-runs-placesphere-takeover/correction}
EXPERIMENT=configs/experiments/data_size_optional400.toml

drop_resume() {  # <checkpoint dir>: delete resume.pt only next to a final.pt
  if [[ -f "$1/final.pt" && -f "$1/resume.pt" ]]; then
    echo "removing $1/resume.pt"
    rm -f "$1/resume.pt"
  fi
}

for dir in "$CORRECTION"/s*/models/*/checkpoints; do
  [[ -d "$dir" ]] && drop_resume "$dir"
done
df -h "$HOME" | tail -1

for index in 0 1 2 3 4; do
  run=placesphere_rgb_unet_n400_s$((index + 1))
  echo "== train $run $(date -Is)"
  .venv/bin/python -u scripts/sweep.py train --experiment "$EXPERIMENT" --task placesphere \
    --index "$index" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"
  drop_resume "$RUN_ROOT/$run/checkpoints"
  for split in val test; do
    [[ -f "$RUN_ROOT/$run/eval/${split}_final_h200.json" ]] && continue
    echo "== eval $run $split $(date -Is)"
    .venv/bin/python -u scripts/eval_dp.py "$RUN_ROOT/$run/checkpoints/final.pt" \
      --split "$split" --num-envs 4 --max-episode-steps 200
  done
  df -h "$HOME" | tail -1
done
