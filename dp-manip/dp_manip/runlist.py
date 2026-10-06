"""Canonical experiment run lists for the sweep CLI and cluster schedulers.

This module is the single source of truth mapping a declared experiment
(``configs/experiments/*.toml``) and the task set (``configs/tasks/*.toml``) to
concrete runs. It resolves each run's config exactly like the trainer does and
derives the run directory name through :func:`dp_manip.config.default_run_name`,
so any scheduler consuming it keeps the array indices, run names, result
directories and the N=100 UNet / backbone UNet reuse relationship unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .completion import Completion, completion_state
from .config import (
    Config,
    ExperimentSpec,
    data_root_override,
    default_run_name,
    load,
    load_experiment,
)


ROOT = Path(__file__).resolve().parents[1]

# Task order defines the sweep identity: array indices and the task-major run
# order shown by ``scripts/sweep.py show``. New tasks are appended, never
# inserted, so existing indices keep pointing at the same run.
TASKS = (
    "pickcube",
    "stackcube",
    "pushcube",
    "pullcube",
    "peginsertionside",
    "plugcharger",
    "placesphere",
    "liftpegupright",
)
DEFAULT_EXPERIMENT = ROOT / "configs" / "experiments" / "data_size.toml"


@dataclass(frozen=True)
class Run:
    """One declared run cell: a task, an experiment value and a seed."""

    task: str
    value: Any  # the declared grid value, e.g. 50 or "transformer"
    seed: int
    experiment: Path

    @property
    def config(self) -> Path:
        return ROOT / "configs" / "tasks" / f"{self.task}.toml"

    def resolve(self, overrides: Sequence[str] = ()) -> Config:
        """Resolve the config train_dp.py will see for this cell.

        ``overrides`` are the same ``--set`` values the queue adds to the
        command, applied before the replicate seed exactly like
        ``config.load_run`` does, so a plan with the same overrides sees the
        config a run actually trained with.
        """
        return load(
            self.config,
            [*overrides, f"train.seed={self.seed}"],
            experiment=self.experiment,
            experiment_value=self.value,
        )

    @property
    def name(self) -> str:
        return default_run_name(self.resolve())

    def directory(self, output_root: str | Path) -> Path:
        """Run directory shared by trainer and sweep: ``<output_root>/<name>``."""
        return Path(output_root).expanduser().resolve() / self.name


def runs(
    experiment_path: str | Path = DEFAULT_EXPERIMENT,
    task: str | None = None,
) -> list[Run]:
    """Enumerate the declared runs in sweep order.

    The full grid is task-major: every declared ``(value, seed)`` cell for the
    first task, then the next task. ``task`` restricts the grid to one declared
    task and preserves that order, so the result is always a subsequence of the
    full grid.
    """
    if task is not None and task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {list(TASKS)}")
    experiment = Path(experiment_path)
    spec: ExperimentSpec = load_experiment(experiment)
    cells = tuple((value, seed) for value in spec.values for seed in spec.seeds_for(value))
    selected = TASKS if task is None else (task,)
    return [Run(name, value, seed, experiment) for name in selected for value, seed in cells]


@dataclass(frozen=True)
class PlannedRun:
    """A declared run plus the trainer-consistent state of its run directory."""

    run: Run
    completion: Completion


def plan_runs(
    experiment_path: str | Path = DEFAULT_EXPERIMENT,
    task: str | None = None,
    *,
    output_root: str | Path,
    overrides: Sequence[str] = (),
    data_root: str | Path | None = None,
) -> list[PlannedRun]:
    """Attach a completion state to every declared run under ``output_root``.

    Read-only: it only inspects ``<output_root>/<run name>`` and never creates
    directories, starts training or writes any completion file of its own.
    Pass the same ``overrides`` and ``data_root`` as the training command (for
    example a smoke ``--set train.total_iters=200`` plus the ``--data-root`` a
    Slurm script exports), so a finished run trained with them is recognized
    as completed instead of looking like a config conflict. ``data_root`` is
    optional; without it the declared config's own root is used.
    """
    runtime = list(overrides)
    if data_root is not None:
        runtime.append(data_root_override(data_root))
    return [
        PlannedRun(
            run,
            completion_state(run.resolve(runtime), run.directory(output_root)),
        )
        for run in runs(experiment_path, task)
    ]


def train_command(
    run: Run,
    *,
    output_root: str | Path,
    data_root: str | Path | None = None,
    overrides: Sequence[str] = (),
    python: str | None = None,
) -> list[str]:
    """The one training command for a declared run.

    ``scripts/sweep.py`` and the queue entry point both call this function, so
    the task/value/seed mapping and the run directory name cannot drift between
    the manual array workflow and the scheduler. ``--device`` is deliberately
    absent: ``scripts/train_dp.py`` keeps its ``cuda`` default, and GPU
    isolation is the worker's ``CUDA_VISIBLE_DEVICES``, not a command flag.
    ``overrides`` become ``--set SECTION.KEY=VALUE`` arguments (for example, a
    reduced-budget smoke run); they are ordinary config overrides, not a second
    experiment definition.
    """
    command = [
        python or sys.executable,
        str(ROOT / "scripts" / "train_dp.py"),
        "--config",
        str(run.config),
        "--experiment",
        str(run.experiment),
        "--experiment-value",
        str(run.value),
        "--seed",
        str(run.seed),
        "--exp",
        run.name,
        "--output-root",
        str(output_root),
        "--resume",
        "auto",
    ]
    if data_root is not None:
        command.extend(("--data-root", str(data_root)))
    for override in overrides:
        command.extend(("--set", override))
    return command


def checkpoint_path(
    run: Run, output_root: str | Path, name: str = "final.pt"
) -> Path:
    """Checkpoint path the sweep CLI passes to the evaluator.

    Unlike :meth:`Run.directory`, this is the literal ``<output-root>/<run
    name>/checkpoints/<name>`` path, without expanding or resolving it, so the
    evaluation command stays byte-identical to the array workflow.
    """
    return Path(output_root) / run.name / "checkpoints" / name


def eval_command(
    run: Run,
    *,
    output_root: str | Path,
    checkpoint: str = "final.pt",
    split: str = "test",
    episodes: int | None = None,
    num_envs: int | None = None,
    render_backend: str | None = None,
    python: str | None = None,
    max_episode_steps: int | None = None,
) -> list[str]:
    """The one evaluation command for a declared run.

    ``scripts/sweep.py`` and the evaluation queue both call this function. A
    ``train``-split diagnostic without ``--episodes`` uses the experiment's
    declared budget, so every nested data-size subset evaluates the same K
    training seeds.
    """
    command = [
        python or sys.executable,
        str(ROOT / "scripts" / "eval_dp.py"),
        str(checkpoint_path(run, output_root, checkpoint)),
        "--split",
        split,
    ]
    if episodes is None and split == "train":
        episodes = load_experiment(run.experiment).train_eval_episodes
    if episodes is not None:
        command.extend(("--episodes", str(episodes)))
    if num_envs is not None:
        command.extend(("--num-envs", str(num_envs)))
    if render_backend is not None:
        command.extend(("--render-backend", render_backend))
    if max_episode_steps is not None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        command.extend(("--max-episode-steps", str(max_episode_steps)))
    return command
