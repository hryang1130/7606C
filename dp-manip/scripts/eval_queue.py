#!/usr/bin/env python3
"""Evaluate the runs of one sweep with one GPU per queue worker.

The evaluation workflow shares the training queue (plan §6.9): the same run
list, the same dynamic workers and the same one-visible-GPU-per-worker
environment. This entry point builds the evaluation command with the existing
:func:`dp_manip.runlist.eval_command`, keeps ``CHECKPOINT``, ``SPLIT`` and
``NUM_ENVS`` semantics, writes one log per run to ``<output-root>/logs/eval``
and reports every failure in the final summary.

A run whose checkpoint is missing fails in the preflight with the missing path
printed, so it is never silently skipped and never costs a GPU slot.

The exit status is 0 when every evaluation finished, 1 when any run failed or
could not start, and 75 when runs were interrupted (the Slurm requeue
convention). ``--output-root`` defaults to ``$RUN_ROOT``, ``--checkpoint`` to
``$CHECKPOINT``, ``--split`` to ``$SPLIT`` and ``--num-envs`` to ``$NUM_ENVS``.

```bash
RUN_ROOT=/scratch/$USER/dp-runs python scripts/eval_queue.py --task peginsertionside
CHECKPOINT=step_010000.pt SPLIT=val RUN_ROOT=... python scripts/eval_queue.py --task pickcube
```
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.runlist import (  # noqa: E402
    DEFAULT_EXPERIMENT,
    TASKS,
    Run,
    checkpoint_path,
    eval_command,
    runs,
)
from dp_manip.scheduler import (  # noqa: E402
    MAX_WORKERS,
    Job,
    QueueSummary,
    gpu_environment,
    run_queue,
)


def queue_evaluation(
    runs_to_evaluate: Sequence[Run],
    *,
    output_root: str | Path,
    logs_dir: str | Path | None = None,
    workers: int = MAX_WORKERS,
    checkpoint: str = "final.pt",
    split: str = "test",
    episodes: int | None = None,
    num_envs: int | None = None,
    render_backend: str | None = None,
    python: str | None = None,
    max_episode_steps: int | None = None,
    command_builder: Callable[[Run], Sequence[str]] | None = None,
) -> QueueSummary:
    """Run every checkpointed run through the shared GPU queue.

    ``command_builder`` exists so tests can inject a fake evaluation command;
    the default is the same :func:`dp_manip.runlist.eval_command` the sweep CLI
    uses. Runs without their checkpoint fail in the preflight and appear as
    failures in the summary instead of being skipped.
    """
    if command_builder is None:
        def command_builder(run: Run) -> Sequence[str]:
            return eval_command(
                run,
                output_root=output_root,
                checkpoint=checkpoint,
                split=split,
                episodes=episodes,
                num_envs=num_envs,
                render_backend=render_backend,
                python=python,
                max_episode_steps=max_episode_steps,
            )

    ready: list[Run] = []
    missing: list[Run] = []
    for run in runs_to_evaluate:
        path = checkpoint_path(run, output_root, checkpoint)
        (ready if path.is_file() else missing).append(run)
    for run in missing:
        print(
            f"[preflight] failed {run.name}: missing checkpoint "
            f"{checkpoint_path(run, output_root, checkpoint)}",
            flush=True,
        )
    if logs_dir is None:
        label = "eval" if max_episode_steps is None else f"eval-{split}-h{max_episode_steps}"
        logs_dir = Path(output_root).expanduser().resolve() / "logs" / label
    return run_queue(
        [Job(name=run.name, command=command_builder(run)) for run in ready],
        workers=workers,
        logs_dir=logs_dir,
        failed=[run.name for run in missing],
        worker_env=gpu_environment(workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--task", choices=TASKS, help="restrict the sweep to one declared task")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=os.environ.get("RUN_ROOT") or ROOT / "runs",
        help="run directory root (default: $RUN_ROOT, else <repository>/runs)",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("CHECKPOINT") or "final.pt",
        help="checkpoint file inside each run (default: $CHECKPOINT, else final.pt)",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val", "train"),
        default=os.environ.get("SPLIT") or "test",
        help="evaluation split (default: $SPLIT, else test)",
    )
    parser.add_argument("--episodes", type=int, help="override the split's episode count")
    parser.add_argument("--max-episode-steps", type=int, help="evaluation-only horizon override")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=os.environ.get("NUM_ENVS") or None,
        help="parallel physx_cpu worker processes (default: $NUM_ENVS, else the config value)",
    )
    parser.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        choices=tuple(range(1, MAX_WORKERS + 1)),
        help="parallel evaluators; a single Slurm job has at most 2 GPUs",
    )
    args = parser.parse_args(argv)
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = queue_evaluation(
        runs(args.experiment, args.task),
        output_root=args.output_root,
        workers=args.workers,
        checkpoint=args.checkpoint,
        split=args.split,
        episodes=args.episodes,
        num_envs=args.num_envs,
        render_backend=args.render_backend,
        max_episode_steps=args.max_episode_steps,
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
