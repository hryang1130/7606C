#!/usr/bin/env python3
"""Train a state-based Diffusion Policy on ManiSkill demos and evaluate it on held-out seeds.

Training loop follows ManiSkill's DP baseline (AdamW, cosine LR with warmup,
EMA weights used for evaluation). Outputs:
  results/<exp>/config.json     resolved config + run metadata
  results/<exp>/metrics.jsonl   training loss / lr, one line per log step
  results/<exp>/val_<iter>.json per-seed validation results
  results/<exp>/summary.json    timings (train vs validation), peak GPU memory, val curve
  checkpoints/<exp>/{best,final}.pt   best = highest validation success_once

Only validation seeds are used here; report numbers with scripts/eval_dp.py on
the test seeds.

Example (wsl):
  .venv/bin/python scripts/train_dp.py --config configs/pickcube_state_jointpos.toml \
      --exp pickcube_smoke --set train.total_iters=300 --set eval.val_episodes=10
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.data import ActionNormalizer, WindowSampler, load_demos  # noqa: E402
from dp_manip.policy import DiffusionPolicy, num_params, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--exp", required=True, help="experiment name (output directory)")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-eval", action="store_true", help="skip validation rollouts (no ManiSkill needed)")
    parser.add_argument("--video", action="store_true", help="record videos of the first validation env")
    parser.add_argument("--force", action="store_true", help="overwrite an existing experiment directory")
    return parser.parse_args()


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return torch.Generator().manual_seed(seed)


def main() -> None:
    args = parse_args()
    cfg = config_lib.load(args.config, args.overrides)
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / args.exp
    ckpt_dir = root / "checkpoints" / args.exp
    if (result_dir.exists() or ckpt_dir.exists()) and not args.force:
        sys.exit(f"{result_dir} or {ckpt_dir} exists; pick another --exp or pass --force")
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("warning: CUDA not available, using cpu")
        device = torch.device("cpu")
    generator = seed_everything(cfg.train.seed)

    # Data and consistency checks between the dataset, config and eval seeds.
    demos = load_demos(root / cfg.data.demo_path, cfg.data.num_demos)
    for name, have, want in [("env_id", demos.env_id, cfg.task.env_id),
                             ("control_mode", demos.control_mode, cfg.task.control_mode),
                             ("obs_mode", demos.obs_mode, cfg.task.obs_mode)]:
        if have != want:
            sys.exit(f"dataset {name}={have!r} but config says {want!r}")
    overlap = set(demos.seeds) & (set(cfg.val_seeds()) | set(cfg.test_seeds()))
    if overlap:
        sys.exit(f"eval seeds overlap demo seeds: {sorted(overlap)[:10]}")
    longest = max(e.actions.shape[0] for e in demos.episodes)
    if longest > cfg.task.max_episode_steps:
        print(f"warning: longest demo has {longest} steps > max_episode_steps={cfg.task.max_episode_steps}")

    normalizer = ActionNormalizer.fit(demos)
    sampler = WindowSampler(demos, normalizer, cfg.policy.obs_horizon, cfg.policy.pred_horizon, device)

    policy = DiffusionPolicy(cfg.policy, demos.obs_dim, demos.act_dim, normalizer).to(device)
    ema_policy = copy.deepcopy(policy)

    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.train.lr, betas=(0.95, 0.999),
                                  weight_decay=cfg.train.weight_decay)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=cfg.train.warmup_steps,
                                 num_training_steps=cfg.train.total_iters)
    # Same call as the baseline. Without use_ema_warmup=True, diffusers ignores
    # `power`; the decay is min(0.9999, (1 + step) / (10 + step)).
    ema = EMAModel(parameters=policy.parameters(), power=0.75)

    run_info = {
        "config": cfg.to_dict(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "demo_seeds": demos.seeds,
        "num_windows": len(sampler),
        "num_transitions": int(sum(e.actions.shape[0] for e in demos.episodes)),
        "obs_dim": demos.obs_dim,
        "act_dim": demos.act_dim,
        "num_params": num_params(policy.noise_pred_net),
        "normalizer": normalizer.state_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "config.json").write_text(json.dumps(run_info, indent=2))
    print(f"{len(demos.episodes)} demos, {run_info['num_transitions']} transitions, "
          f"{len(sampler)} windows, {run_info['num_params'] / 1e6:.2f}M params, device {device}")

    envs = None
    if not args.no_eval:
        from dp_manip.envs import make_eval_envs
        envs = make_eval_envs(cfg, cfg.eval.num_envs,
                              video_dir=str(result_dir / "videos") if args.video else None)

    best = -1.0
    best_iteration = None
    val_curve: list[dict] = []
    val_time = 0.0
    metrics_file = (result_dir / "metrics.jsonl").open("a")

    def run_eval(iteration: int) -> None:
        nonlocal best, best_iteration, val_time
        from dp_manip.evaluate import evaluate

        tick = time.time()
        ema.copy_to(ema_policy.parameters())
        result = evaluate(ema_policy, envs, cfg.val_seeds(), device)
        result["iteration"] = iteration
        (result_dir / f"val_{iteration:07d}.json").write_text(json.dumps(result, indent=2))
        s = result["summary"]
        print(f"[val {iteration}] success_once={s.get('success_once', float('nan')):.3f} "
              f"success_at_end={s.get('success_at_end', float('nan')):.3f} "
              f"over {s['num_episodes']} episodes ({s['wall_time_s']:.0f}s)")
        val_curve.append({"iteration": iteration, **{k: s[k] for k in ("success_once", "success_at_end") if k in s}})
        score = s.get("success_once", 0.0)
        if score > best:
            best, best_iteration = score, iteration
            save_checkpoint(ckpt_dir / "best.pt", policy=policy, ema_policy=ema_policy,
                            config=cfg.to_dict(), iteration=iteration, extra={"val": s})
        val_time += time.time() - tick

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    policy.train()
    start = time.time()
    for iteration in range(1, cfg.train.total_iters + 1):
        obs_seq, act_seq = sampler.sample(cfg.train.batch_size, generator)
        loss = policy.compute_loss(obs_seq, act_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema.step(policy.parameters())

        if iteration % cfg.train.log_freq == 0 or iteration == 1:
            record = {"iteration": iteration, "loss": loss.item(),
                      "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - start}
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            print(f"[train {iteration}/{cfg.train.total_iters}] loss={record['loss']:.4f} "
                  f"lr={record['lr']:.2e} {record['elapsed_s']:.0f}s")
        if envs is not None and cfg.train.eval_freq and iteration % cfg.train.eval_freq == 0 \
                and iteration != cfg.train.total_iters:
            run_eval(iteration)
        if cfg.train.save_freq and iteration % cfg.train.save_freq == 0:
            ema.copy_to(ema_policy.parameters())
            save_checkpoint(ckpt_dir / f"iter_{iteration:07d}.pt", policy=policy, ema_policy=ema_policy,
                            config=cfg.to_dict(), iteration=iteration)

    if envs is not None:
        run_eval(cfg.train.total_iters)
        envs.close()
    ema.copy_to(ema_policy.parameters())
    save_checkpoint(ckpt_dir / "final.pt", policy=policy, ema_policy=ema_policy,
                    config=cfg.to_dict(), iteration=cfg.train.total_iters)
    metrics_file.close()

    total = time.time() - start
    summary = {
        "exp": args.exp,
        "num_demos": len(demos.episodes),
        "total_iters": cfg.train.total_iters,
        "wall_time_s": total,
        # Everything that is not validation: optimizer steps, logging, checkpoint saves.
        "train_time_s": total - val_time,
        "train_ms_per_iter": 1000 * (total - val_time) / cfg.train.total_iters,
        "val_time_s": val_time,
        "val_runs": len(val_curve),
        "peak_gpu_mem_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_gpu_mem_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "final_loss": loss.item(),
        "best_val_success_once": best if best_iteration is not None else None,
        "best_iteration": best_iteration,
        "val_curve": val_curve,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    mem = summary["peak_gpu_mem_reserved_mb"]
    print(f"done in {total:.0f}s (train {summary['train_time_s']:.0f}s = {summary['train_ms_per_iter']:.1f} ms/iter, "
          f"val {val_time:.0f}s){f', peak GPU reserved {mem:.0f} MiB' if mem else ''}; "
          f"outputs in {result_dir} and {ckpt_dir}")


if __name__ == "__main__":
    main()
