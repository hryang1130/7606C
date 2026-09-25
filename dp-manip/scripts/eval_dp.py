#!/usr/bin/env python3
"""Evaluate a trained checkpoint on held-out test seeds (or another seed split).

Uses the config stored in the checkpoint, so control mode, horizons and episode
length always match training. Writes results/<exp>/<split>_<ckpt>.json.

Splits:
  test   config eval.test_seed_* (default; the only split for reported numbers)
  val    config eval.val_seed_* (the seeds used to pick best.pt)
  train  the demo seeds: a diagnostic. A correct pipeline should reproduce the
         demos here; failure on train seeds points to an env/data mismatch
         rather than to poor generalization.

Example (wsl):
  .venv/bin/python scripts/eval_dp.py checkpoints/pickcube_smoke/final.pt
  .venv/bin/python scripts/eval_dp.py checkpoints/pickcube_smoke/final.pt --split train
  .venv/bin/python scripts/eval_dp.py checkpoints/pickcube_smoke/final.pt --save-states
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip.data import load_demos  # noqa: E402
from dp_manip.envs import make_eval_envs  # noqa: E402
from dp_manip.evaluate import evaluate  # noqa: E402
from dp_manip.policy import load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("test", "val", "train"), default="test")
    parser.add_argument("--episodes", type=int, help="evaluate only the first N seeds of the split")
    parser.add_argument("--video", action="store_true", help="record videos of the first env")
    parser.add_argument("--save-states", action="store_true",
                        help="save every episode's env states for offline rendering (scripts/render_episodes.py)")
    parser.add_argument("--seed", type=int, default=0, help="torch seed for diffusion sampling noise")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA not available, using cpu")
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    policy, cfg, ckpt = load_checkpoint(args.checkpoint, device)
    root = Path(__file__).resolve().parents[1]
    if args.split == "train":
        seeds = load_demos(root / cfg.data.demo_path, cfg.data.num_demos).seeds
    else:
        seeds = cfg.test_seeds() if args.split == "test" else cfg.val_seeds()
    seeds = seeds[: args.episodes] if args.episodes else seeds
    exp_dir = root / "results" / args.checkpoint.parent.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    tag = args.checkpoint.stem

    # Largest env count <= eval.num_envs that divides the number of seeds.
    num_envs = max(n for n in range(1, min(cfg.eval.num_envs, len(seeds)) + 1) if len(seeds) % n == 0)
    envs = make_eval_envs(cfg, num_envs,
                          video_dir=str(exp_dir / f"videos_{args.split}_{tag}") if args.video else None,
                          states_dir=str(exp_dir / f"states_{args.split}_{tag}") if args.save_states else None)
    try:
        result = evaluate(policy, envs, seeds, device)
    finally:
        envs.close()
    result.update(checkpoint=str(args.checkpoint), iteration=ckpt["iteration"], split=args.split,
                  sampling_seed=args.seed)
    out = exp_dir / f"{args.split}_{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(f"{cfg.task.env_id} {tag} (iter {ckpt['iteration']}): success_once={s.get('success_once'):.3f} "
          f"success_at_end={s.get('success_at_end'):.3f} over {s['num_episodes']} {args.split} episodes "
          f"(seeds {seeds[0]}..{seeds[-1]}), {s['mean_inference_ms']:.1f} ms/inference -> {out}")


if __name__ == "__main__":
    main()
