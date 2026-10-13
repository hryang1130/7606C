#!/usr/bin/env bash
# Run several tasks inside ONE allocation, at most JOBS at a time (for clusters that limit
# the number of jobs per user). Each task is a separate generate_task.py process with its
# own log, <out>/logs/<task>.out; rerunning continues where a task stopped (.done markers).
#
#   ./run_all.sh                                   # all tasks, JOBS = CPUs of this allocation
#   JOBS=3 ./run_all.sh pickcube stackcube         # some tasks
#   OUT=/scratch/$USER/demos ./run_all.sh          # output root (generate_task.py --out)
#   ARGS="--stages expert" ./run_all.sh            # extra generate_task.py arguments
#
# One task needs about one CPU (simulation, planning) and a little GPU time (rendering).
# Inside an existing interactive allocation: srun --jobid=<id> --overlap ./run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
unset UV_PROJECT_ENVIRONMENT
export PY=${PY:-.venv/bin/python} OUT=${OUT:-data} ARGS=${ARGS:-}
JOBS=${JOBS:-${SLURM_CPUS_ON_NODE:-$(nproc)}}
tasks=("$@")
[ ${#tasks[@]} -gt 0 ] || tasks=(pickcube stackcube pushcube pullcube peginsertionside plugcharger placesphere liftpegupright)
mkdir -p "$OUT/logs"
echo "=== run_all $(date "+%F %T"): ${tasks[*]}; $JOBS at a time; out $OUT; logs $OUT/logs/<task>.out"

# xargs -P keeps JOBS processes running; each prints one line when it ends.
printf '%s\n' "${tasks[@]}" | xargs -P "$JOBS" -n 1 bash -c '
  echo "=== [$(date +%H:%M:%S)] started $1"
  # shellcheck disable=SC2086  # ARGS is a list of arguments
  if "$PY" generate_task.py "$1" --out "$OUT" $ARGS > "$OUT/logs/$1.out" 2>&1; then
    echo "=== [$(date +%H:%M:%S)] $1 OK"
  else
    echo "=== [$(date +%H:%M:%S)] $1 FAILED (see $OUT/logs/$1.out)"; exit 1
  fi' run_task
status=$?
[ "$status" = 0 ] && echo "=== all tasks OK" || echo "=== some tasks failed; rerun to continue them"
exit "$status"
