#!/usr/bin/env python3
"""Train an RGB or complete-state policy from ``maniskill-demogen`` datasets.

This is the sweep/Slurm entry point: it resolves a task config plus an optional
experiment grid value and hands the result to the single trainer in
``dp_manip.trainer``. The unified task/experiment/value entry point is
``scripts/run_experiment.py``; both share the same training pipeline.

Training is cluster-first: the selected episodes' RGB is decoded into RAM up
front (``--set data.preload=false`` reads it lazily in DataLoader workers
instead; see configs/README.md), the run is fixed to an optimizer-step budget,
restartable checkpoints also carry Python/NumPy/torch RNG state, and no
ManiSkill installation is needed until closed-loop evaluation. Batches are drawn per optimizer step from
``(seed, step)`` so a Slurm requeue continues the same stochastic trajectory as
a continuous run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip import invariants  # noqa: E402
from dp_manip.config import Config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="task-only config")
    parser.add_argument("--experiment", type=Path, help="experiment specification or override TOML")
    parser.add_argument("--experiment-value", help="selected value from an experiment grid")
    parser.add_argument("--data-root", type=Path, help="override data.root (for example, a scratch dataset directory)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--exp", help="run directory name; default: <task>_rgb_<backbone>_n<N>_s<seed>")
    parser.add_argument("--num-demos", type=int, help="runtime override for data.num_demos")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value (repeatable)",
    )
    return parser.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> Config:
    """Resolve the layered config plus this invocation's runtime overrides."""
    return config_lib.load_run(
        args.config,
        args.overrides,
        experiment=args.experiment,
        experiment_value=args.experiment_value,
        num_demos=args.num_demos,
        seed=args.seed,
        data_root=args.data_root,
    )


def experiment_context(args: argparse.Namespace, cfg: Config) -> dict | None:
    """Describe the declared cell when ``--experiment`` is a grid spec (plan §19)."""
    if args.experiment is None or args.experiment_value is None:
        return None
    spec = config_lib.load_experiment_optional(args.experiment)
    if spec is None:
        return None
    return invariants.experiment_context(cfg, spec, args.experiment_value, spec_path=args.experiment)


def main() -> int:
    args = parse_args()
    cfg = resolve_config(args)
    # Imported late so config resolution stays importable without the heavy
    # torch training stack (the same reason both entry points share it).
    from dp_manip.trainer import run_training

    return run_training(
        cfg,
        output_root=args.output_root,
        run_name=args.exp,
        device=args.device,
        resume=args.resume,
        experiment_context=experiment_context(args, cfg),
    )


if __name__ == "__main__":
    raise SystemExit(main())
