#!/usr/bin/env python3
"""Drive the failure-aware study to its next pause point (plan §8 Phase 5 runbook).

Reads the lock file, runs every stage that is not done yet in runbook order and
stops at the next point that needs a person:

1. after checkpoint 1's rollouts are collected, to re-project the GPU budget
   from the measured rollout cost (plan §10.3) — continue with --budget-confirmed;
2. when the offline gate fails (plan §6.7) — the study ends there;
3. before the test split, so the lock file is committed first — continue with
   --include-test.

Every stage is idempotent, so the same command is simply resubmitted after a
pause, a preemption (exit 75, requeued by slurm/failure_aware.sbatch) or a
crash. The study and a smoke run differ only in their arguments:

    # smoke: scaled-down protocol, scratch lock file and rollout root
    sbatch slurm/failure_aware.sbatch scripts/failure_pipeline.py --task placesphere \\
      --run-root "$RUN_ROOT" --max-episode-steps 200 --protocol configs/failure_aware/smoke_protocol.toml \\
      --lock "$SCRATCH/smoke/placesphere.toml" --rollout-root "$SCRATCH/smoke" \\
      --budget-confirmed --include-test

    # study
    sbatch slurm/failure_aware.sbatch scripts/failure_pipeline.py --task placesphere --run-root "$RUN_ROOT" \\
      --max-episode-steps 200

``--max-episode-steps`` is locked by select-cell on the first run; later runs
may omit it, and a value that differs from the lock is refused. The replication
task (plan §12) adds ``--no-low-success``, so it stops when its baseline is
outside the band instead of running in low-success mode.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import collect_rollouts  # noqa: E402
import failure_study as study_cli  # noqa: E402
import finetune_dp  # noqa: E402

from dp_manip import failure_study as study  # noqa: E402
from dp_manip.failure_lock import LockFile, lock_path  # noqa: E402
from dp_manip.failure_protocol import DEFAULT_PROTOCOL, FailureProtocol, load_protocol  # noqa: E402
from dp_manip.failure_rollout import DATASET_DIR, RAW_HOLDOUT, RAW_TRAIN, SUMMARY  # noqa: E402

EXIT_REQUEUE = 75
# Plan §10.1: 10k fine-tuning steps ~ 0.4 GPU-h; a guided episode ~ 1.6 baseline episodes.
FINETUNE_HOURS_PER_STEP = 0.4 / 10_000
GUIDED_COST = 1.6


class Preempted(Exception):
    """The scheduler asked the job to stop; resubmitting resumes from the lock file."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-root", type=Path, required=True, help="main-track run root (baseline cells)")
    parser.add_argument("--rollout-root", type=Path, help="root of failure_aware/<task>/ (default: --run-root)")
    parser.add_argument(
        "--max-episode-steps",
        type=study_cli.positive_int,
        help="study horizon passed to select-cell (plan §1, §13); default: the checkpoints' recorded one",
    )
    parser.add_argument(
        "--no-low-success",
        action="store_true",
        help="passed to select-cell: stop if no baseline cell is in the band (plan §12.3, replication task)",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock", type=Path, help="default: configs/failure_aware/<task>.toml (study protocol only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    parser.add_argument("--budget-confirmed", action="store_true", help="continue past the budget pause")
    parser.add_argument("--include-test", action="store_true", help="continue into the test split")
    parser.add_argument(
        "--finetune-set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="machine override forwarded to finetune_dp.py, e.g. train.num_workers=3",
    )
    return parser.parse_args(argv)


class Pipeline:
    def __init__(self, args: argparse.Namespace, envs_factory: Callable):
        self.args = args
        self.envs_factory = envs_factory
        protocol_path = Path(args.protocol).resolve()
        default_lock = lock_path(args.task).resolve()
        if args.lock is None and protocol_path != DEFAULT_PROTOCOL.resolve():
            raise ValueError("a non-study protocol (smoke) needs its own --lock")
        self.lock_path = Path(args.lock).resolve() if args.lock else default_lock
        if self.lock_path == default_lock and protocol_path != DEFAULT_PROTOCOL.resolve():
            raise ValueError(f"{default_lock} is the study's lock file; a smoke run must use another --lock")
        self.protocol: FailureProtocol = load_protocol(protocol_path)
        self.rollout_root = Path(args.rollout_root or args.run_root).resolve()
        # The study's rollouts live under the run root; a smoke run writing there
        # would leave raw files the study then refuses (built under another protocol).
        if protocol_path != DEFAULT_PROTOCOL.resolve() and self.rollout_root == Path(args.run_root).resolve():
            raise ValueError("a non-study protocol (smoke) needs its own --rollout-root, not the run root")
        self.common = ["--task", args.task, "--lock", str(self.lock_path), "--protocol", str(protocol_path)]
        self.render = ["--render-backend", args.render_backend] if args.render_backend else []

    # ----------------------------------------------------------- plumbing

    @property
    def lock(self) -> LockFile:
        return LockFile(self.lock_path)

    def study(self, *argv: str) -> None:
        self.guard_signals()
        study_cli.main([*argv, *self.common], envs_factory=self.envs_factory)

    def guard_signals(self) -> None:
        """(Re-)install the preemption handler; the trainer installs its own while it runs."""

        def stop(signum, _frame):
            raise Preempted(f"signal {signum}")

        for name in ("SIGUSR1", "SIGTERM"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), stop)

    def collect(self, checkpoint: str, split: str, overwrite: bool) -> None:
        self.guard_signals()
        argv = [
            "collect", checkpoint, "--split", split, "--protocol", str(self.protocol.path),
            "--run-root", str(self.rollout_root), "--device", self.args.device, *self.render,
        ]
        horizon = study.study_horizon(self.lock)
        if horizon is not None:
            argv += ["--max-episode-steps", str(horizon)]
        collect_rollouts.main([*argv, *(["--overwrite"] if overwrite else [])], envs_factory=self.envs_factory)

    def finetune(self, checkpoint: str, label: str, lr: float, steps: int, extra_steps=()) -> None:
        argv = [
            checkpoint, "--label", label, "--lr", repr(float(lr)), "--steps", str(int(steps)),
            "--protocol", str(self.protocol.path), "--run-root", str(self.rollout_root), "--device", self.args.device,
        ]
        if extra_steps:
            argv += ["--checkpoint-steps", *map(str, extra_steps)]
        for override in self.args.finetune_set:
            argv += ["--set", override]
        self.guard_signals()
        if finetune_dp.main(argv) == EXIT_REQUEUE:
            raise Preempted("fine-tuning checkpointed after a scheduler signal")

    # ------------------------------------------------------------ stages

    def run(self) -> int:
        if not self.lock.has("task"):
            print("== select-cell")
            horizon = self.args.max_episode_steps
            self.study(
                "select-cell", "--run-root", str(self.args.run_root), "--rollout-root", str(self.rollout_root),
                *(["--max-episode-steps", str(horizon)] if horizon is not None else []),
                *(["--no-low-success"] if self.args.no_low_success else []),
            )
        locked_horizon = study.study_horizon(self.lock)
        if self.args.max_episode_steps is not None and self.args.max_episode_steps != locked_horizon:
            raise ValueError(
                f"the lock file's horizon is {locked_horizon}, not --max-episode-steps {self.args.max_episode_steps}"
            )
        locked_root = self.lock.require("task", "select-cell").get("rollout_root")
        if locked_root is None or Path(locked_root).resolve() != self.rollout_root:
            raise ValueError(f"the lock file's rollout root is {locked_root}, not {self.rollout_root}")
        seeds = study.checkpoint_seeds(self.lock)

        for index, seed in enumerate(seeds):
            self.collection(seed)
            if index == 0 and not self.args.budget_confirmed:
                self.report_projection()
                print(
                    "PAUSED (budget, plan §10.3): check the projection above, record any cut in the plan's "
                    "§11 change log, then resubmit with --budget-confirmed."
                )
                return 0

        self.models(seeds)
        if not self.lock.has("gate"):
            print("== gate")
            self.study("gate", "--device", self.args.device)
        if not self.lock.get("gate.passed"):
            print("STOPPED: the offline gate failed; report the offline losses (plan §6.7). The study ends here.")
            return 0

        if not self.lock.has("guidance.dry_run"):
            print("== dry-run")
            self.study("dry-run", "--device", self.args.device, *self.render)
        if not self.lock.has("guidance.selection"):
            for arm in ("F", "A"):
                print(f"== tuning {arm}")
                self.study("eval", "--split", "tuning", "--arm", arm, "--device", self.args.device, *self.render)
            print("== select-alpha")
            self.study("select-alpha")
        low = study.low_success(self.lock)
        if not low and not self.lock.has("guidance.c1"):
            print("== tuning C1")
            self.study("eval", "--split", "tuning", "--arm", "C1", "--device", self.args.device, *self.render)
            print("== select-c1")
            self.study("select-c1")

        if not self.args.include_test:
            self.study("status")
            print(
                f"PAUSED (test): the design is locked in {self.lock_path}. Commit it, "
                "then resubmit with --include-test to run the test split."
            )
            return 0
        for arm in study.ARMS:
            if low and arm in ("C1", "C2"):
                continue
            print(f"== test {arm}")
            self.study("eval", "--split", "test", "--arm", arm, "--device", self.args.device, *self.render)
        self.study("status")
        print("DONE: every test result is written.")
        return 0

    def collection(self, seed: int) -> None:
        checkpoint = self.lock.require(f"checkpoints.s{seed}", "select-cell")["path"]
        rollout = study.rollout_dir(self.lock, seed)
        for split, raw in (("train", RAW_TRAIN), ("holdout", RAW_HOLDOUT)):
            if not (rollout / raw).with_suffix(".json").is_file():
                print(f"== s{seed}: collect {split}")
                # A raw HDF5 without its sidecar is an interrupted write.
                self.collect(checkpoint, split, overwrite=(rollout / raw).exists())
        if not (rollout / DATASET_DIR / SUMMARY).is_file():
            print(f"== s{seed}: build")
            self.guard_signals()
            collect_rollouts.main(
                ["build", str(rollout), "--protocol", str(self.protocol.path), "--overwrite"],
                envs_factory=self.envs_factory,
            )
        if not self.lock.has(f"collection.s{seed}"):
            self.study("record-collection", "--seed", str(seed))

    def models(self, seeds: list[int]) -> None:
        pilot_seed = seeds[0]
        if not self.lock.has("finetune"):
            checkpoint = self.lock.require(f"checkpoints.s{pilot_seed}", "select-cell")["path"]
            longest = max(self.protocol.pilot.steps)
            shorter = sorted(step for step in self.protocol.pilot.steps if step < longest)
            for lr in self.protocol.pilot.learning_rates:
                print(f"== pilot fine-tune lr={lr:g}")
                self.finetune(checkpoint, "failure", lr, longest, shorter)
            print("== pilot")
            self.study("pilot", "--device", self.args.device)
        finetune = self.lock.require("finetune", "pilot")
        lr, steps = float(finetune["lr"]), int(finetune["steps"])
        for seed in seeds:
            if self.lock.has(f"models.s{seed}"):
                continue
            checkpoint = self.lock.require(f"checkpoints.s{seed}", "select-cell")["path"]
            labels = [] if seed == int(finetune["pilot_seed"]) else ["failure"]
            if not study.low_success(self.lock):
                labels.append("success")
            for label in labels:
                print(f"== s{seed}: fine-tune {label}")
                self.finetune(checkpoint, label, lr, steps)
            self.study("record-models", "--seed", str(seed))

    def report_projection(self) -> None:
        """Measured rollout cost of checkpoint 1 and the projected study total (plan §10.3)."""
        lock, protocol = self.lock, self.protocol
        seeds = study.checkpoint_seeds(lock)
        rollout = study.rollout_dir(lock, seeds[0])
        seconds = episodes = 0.0
        for raw in (RAW_TRAIN, RAW_HOLDOUT):
            provenance = json.loads((rollout / raw).with_suffix(".json").read_text())["rollout_provenance"]
            seconds += provenance["wall_time_s"]
            episodes += len(provenance["seeds_used"])
        per_episode = seconds / 3600 / episodes
        train_rollouts = lock.require(f"collection.s{seeds[0]}", "record-collection")["train_rollouts"]
        n = len(seeds)
        grid = len(protocol.guidance.alpha_grid)
        test = protocol.evaluation.test_episodes
        lines = {
            "collection": n * (train_rollouts + protocol.collection.holdout_episodes) * per_episode,
            "fine-tuning (pilot + 2n-1 models at the longest steps)": FINETUNE_HOURS_PER_STEP
            * max(protocol.pilot.steps)
            * (len(protocol.pilot.learning_rates) + 2 * n - 1),
            "dry run": protocol.guidance.dry_run_episodes * (1 + GUIDED_COST) * per_episode,
            "tuning F, A, C1": 3 * grid * protocol.guidance.tuning_episodes * n * GUIDED_COST * per_episode,
            "test F, A, C1": 3 * n * test * GUIDED_COST * per_episode,
            "test B, C2": 2 * n * test * per_episode,
        }
        print(f"measured: {per_episode * 100:.3f} GPU-h per 100 baseline episodes ({episodes:.0f} episodes)")
        for name, hours in lines.items():
            print(f"  {name:<55s} {hours:6.2f} GPU-h")
        print(f"  {'projected total':<55s} {sum(lines.values()):6.2f} GPU-h (cap 24; cut order §10.3)")


def main(argv: list[str] | None = None, envs_factory: Callable = study_cli.default_envs_factory) -> int:
    args = parse_args(argv)
    previous = {
        name: signal.getsignal(getattr(signal, name)) for name in ("SIGUSR1", "SIGTERM") if hasattr(signal, name)
    }
    try:
        return Pipeline(args, envs_factory).run()
    except Preempted as reason:
        print(f"preempted ({reason}); resubmit the same command to continue", flush=True)
        return EXIT_REQUEUE
    finally:
        for name, handler in previous.items():
            signal.signal(getattr(signal, name), handler)


if __name__ == "__main__":
    raise SystemExit(main())
