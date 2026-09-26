#!/usr/bin/env python3
"""Evaluate an RGB checkpoint on fixed validation, test, or training seeds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import from_dict  # noqa: E402
from dp_manip.data import NormalizationStats  # noqa: E402
from dp_manip.envs import make_eval_envs  # noqa: E402
from dp_manip.evaluate import evaluate  # noqa: E402
from dp_manip.policy import DiffusionPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episodes", type=int, help="override the split's episode count")
    parser.add_argument("--num-envs", type=int, help="parallel physx_cpu worker processes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = from_dict(checkpoint["config"])
    train_data = checkpoint["train_data"]
    stats = NormalizationStats.from_dict(checkpoint["normalization"])
    policy = DiffusionPolicy(
        cfg.policy,
        cfg.vision,
        image_shape=tuple(train_data["image_shape"]),
        proprio_dim=int(train_data["proprio_dim"]),
        action_dim=int(train_data["action_dim"]),
        stats=stats,
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    if args.split == "test":
        seeds = cfg.test_seeds()
    elif args.split == "val":
        seeds = cfg.val_seeds()
    else:
        # Overfitting diagnostic from the experiment plan: the first 25
        # demonstration seeds are shared by every nested data-size subset.
        seeds = [int(seed) for seed in train_data["seeds"][:25]]
    if args.episodes is not None:
        if args.episodes < 1 or args.episodes > len(seeds):
            raise ValueError(f"--episodes must be in [1, {len(seeds)}] for split {args.split}")
        seeds = seeds[: args.episodes]
    requested_envs = args.num_envs or cfg.eval.num_envs
    num_envs = math.gcd(len(seeds), requested_envs)
    if num_envs != requested_envs:
        print(f"adjusting num_envs from {requested_envs} to {num_envs} so all waves are full")

    envs = make_eval_envs(cfg, num_envs, args.render_backend)
    try:
        result = evaluate(
            policy,
            envs,
            seeds,
            device,
            inference_seed=cfg.eval.inference_seed,
        )
    finally:
        envs.close()
    result.update(
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_step=int(checkpoint["step"]),
        split=args.split,
        env_id=cfg.task.env_id,
        sim_backend=cfg.task.sim_backend,
        num_envs=num_envs,
        inference_seed=cfg.eval.inference_seed,
    )
    output = args.output
    if output is None:
        run_dir = args.checkpoint.resolve().parents[1]
        output = run_dir / "eval" / f"{args.split}_{args.checkpoint.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = result["summary"]
    print(
        f"{args.split}: success_once={summary['success_once']:.3f} "
        f"success_at_end={summary['success_at_end']:.3f} "
        f"({summary['num_episodes']} episodes); {output}"
    )


if __name__ == "__main__":
    main()
