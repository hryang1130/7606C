#!/usr/bin/env bash
# Generate the dataset of ONE task (one teammate, one task):
#
#   ./generate.sh <task> [generate_task.py options]
#
#   tasks: pickcube stackcube pushcube pullcube peginsertionside plugcharger placesphere liftpegupright
#
#   ./generate.sh pickcube                           # everything, plan defaults, into ./data
#   ./generate.sh plugcharger --out /scratch/$USER/demos
#   sbatch --export=TASK=pickcube slurm/generate_task.sbatch    # the same as a batch job
#
# Needs ./setup.sh first (once, on a node with internet) and a node that can render
# (check with ./check_env.sh). Rerunning the same command continues where it stopped.
# At the end it prints where the training data is and how many demos it has.
set -uo pipefail
cd "$(dirname "$0")"
unset UV_PROJECT_ENVIRONMENT
PY=${PY:-.venv/bin/python}
TASKS="pickcube stackcube pushcube pullcube peginsertionside plugcharger placesphere liftpegupright"

task=${1:-}
if [ -z "$task" ] || [[ " $TASKS " != *" $task "* ]]; then
  echo "usage: $0 <task> [generate_task.py options]   (tasks: $TASKS)" >&2
  exit 2
fi
shift
[ -x "$PY" ] || { echo "no .venv: run ./setup.sh first" >&2; exit 1; }

out=data
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do [ "${args[i]}" = --out ] && out=${args[i + 1]}; done
mkdir -p "$out/logs"
log="$out/logs/$task.out"
echo "=== $task: full output in $log"

"$PY" generate_task.py "$task" "$@" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}

"$PY" - "$out" "$task" <<'PY'
import glob, json, sys
from tasks import TASKS
out, task = sys.argv[1], sys.argv[2]
env = TASKS[task].env_id
files = sorted(glob.glob(f"{out}/dataset/*/{env}/motionplanning/trajectory.state.*.physx_cpu.h5"))
print(f"=== {task} ({env}) dataset")
for path in files:
    info = json.load(open(path[:-3] + ".export_info.json"))
    print(f"{info['split']:5s} {info['num_demos']:3d} demos, {info['total_steps']} steps, seeds {info['seeds'][0]}-{info['seeds'][-1]}, "
          f"action {info['action_dim']} ({info['control_mode']}), obs {info['obs_dim']}, obs_rgb/state {info['obs_rgb_state_dim']}, "
          f"images {info['obs_rgb_image_shape']}\n      {path}")
if not files:
    print("no exported files yet")
for png in glob.glob(f"{out}/dataset/train/{env}/motionplanning/sample.png"):
    print(f"preview (first and last frames): {png}")
PY
if [ "$status" = 0 ]; then
  echo "=== $task done"
else
  echo "=== $task not finished (see $log); rerun the same command to continue"
fi
exit "$status"
