#!/usr/bin/env bash
# Every takeover-correction stage in one job (the QOS allows one job per user).
# Stages are resumable and write-once, so s1's finished collection is skipped
# and a rerun continues where it stopped. The order surfaces build/finetune/
# evaluate problems on s1 before the remaining collections run. The job ends
# after analyze and submits nothing else.
#SBATCH --job-name=dp-takeover-pipe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
OUT=${1:?output directory}
run() { echo "== $* $(date -Is)"; .venv/bin/python -u scripts/takeover_correction.py "$@" --output "$OUT"; }
echo "== tests $(date -Is)"
.venv/bin/python -m unittest tests.test_takeover_data tests.test_takeover tests.test_data_adapter \
  tests.test_finetune tests.test_data_nesting tests.test_failure_pipeline tests.test_checkpoint_lifecycle
for s in s1 s2 s3; do
  run collect --checkpoint "$s"
  run build --checkpoint "$s"
  run finetune --checkpoint "$s" --arm C
  run evaluate --checkpoint "$s" --arm B
  run evaluate --checkpoint "$s" --arm C
  run finetune --checkpoint "$s" --arm D
  run evaluate --checkpoint "$s" --arm D
done
run analyze
