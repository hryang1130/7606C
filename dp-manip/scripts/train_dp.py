#!/usr/bin/env python3
"""Train an RGB Diffusion Policy from ``maniskill-demogen`` datasets.

The entry point is cluster-first: RGB is read lazily with worker processes,
training is fixed to an optimizer-step budget, checkpoints are restartable,
and no ManiSkill installation is needed until closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.data import (  # noqa: E402
    DatasetInfo,
    NormalizationStats,
    RGBWindowDataset,
    compute_normalization,
    read_dataset_info,
)
from dp_manip.policy import DiffusionPolicy, num_params  # noqa: E402
from dp_manip.training import (  # noqa: E402
    ExponentialMovingAverage,
    atomic_torch_save,
    averaged_state_dict,
    cosine_warmup,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="override data.root (for example, a scratch dataset directory)")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--exp", help="run directory name; default: <task>_rgb_unet_n<N>_s<seed>")
    parser.add_argument("--num-demos", type=int, choices=(25, 50, 100, 200, 400))
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
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_data_path(data_root: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else data_root / path


def check_dataset(info: DatasetInfo, cfg: config_lib.Config, split: str) -> None:
    if info.env_id != cfg.task.env_id:
        raise ValueError(f"{split} dataset env_id={info.env_id!r}, expected {cfg.task.env_id!r}")
    if info.control_mode != cfg.task.control_mode:
        raise ValueError(
            f"{split} dataset control_mode={info.control_mode!r}, expected {cfg.task.control_mode!r}"
        )
    if split == "train" and any(seed >= 4_000 for seed in info.seeds):
        raise ValueError("training demonstrations must use seeds below 4000")
    if split == "val" and any(not 4_000 <= seed < 5_000 for seed in info.seeds):
        raise ValueError("validation demonstrations must use seeds in [4000, 5000)")


def dataset_record(info: DatasetInfo) -> dict:
    return {
        "path": str(info.path),
        "env_id": info.env_id,
        "control_mode": info.control_mode,
        "num_demos": len(info.episodes),
        "num_transitions": info.num_transitions,
        "seeds": info.seeds,
        "image_shape": list(info.image_shape),
        "proprio_dim": info.proprio_dim,
        "action_dim": info.action_dim,
        "cameras": list(info.cameras),
        "rgb_env_info": info.rgb_env_info,
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def validate(
    policy: DiffusionPolicy,
    ema: ExponentialMovingAverage,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    seed: int = 0,
) -> float:
    generator = torch.Generator(device=device).manual_seed(seed)
    total_loss, total_items = 0.0, 0
    was_training = policy.training
    with ema.average_parameters(policy):
        policy.eval()
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                with torch.amp.autocast(device.type, enabled=amp):
                    loss = policy.compute_loss(
                        batch["rgb"], batch["state"], batch["actions"], generator=generator
                    )
                count = batch["rgb"].shape[0]
                total_loss += loss.detach().item() * count
                total_items += count
    policy.train(was_training)
    return total_loss / total_items


def inference_payload(
    policy: DiffusionPolicy,
    ema: ExponentialMovingAverage,
    cfg: config_lib.Config,
    train_info: DatasetInfo,
    val_info: DatasetInfo,
    stats: NormalizationStats,
    step: int,
) -> dict:
    return {
        "format_version": 1,
        "model": averaged_state_dict(policy, ema),
        "config": cfg.to_dict(),
        "train_data": dataset_record(train_info),
        "val_data": dataset_record(val_info),
        "normalization": stats.to_dict(),
        "step": step,
    }


def main() -> int:
    args = parse_args()
    cfg = config_lib.load(args.config, args.overrides)
    if args.num_demos is not None:
        cfg.data.num_demos = args.num_demos
    if args.seed is not None:
        cfg.train.seed = args.seed
    cfg.validate()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; cluster training must run on a GPU node")
    seed_everything(cfg.train.seed)

    data_root = (args.data_root or Path(cfg.data.root)).expanduser()
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    train_info = read_dataset_info(
        resolve_data_path(data_root, cfg.data.train_path), cfg.data.num_demos
    )
    val_info = read_dataset_info(
        resolve_data_path(data_root, cfg.data.val_path), cfg.data.val_num_demos
    )
    check_dataset(train_info, cfg, "train")
    check_dataset(val_info, cfg, "val")
    if (
        train_info.image_shape,
        train_info.proprio_dim,
        train_info.action_dim,
        train_info.cameras,
    ) != (
        val_info.image_shape,
        val_info.proprio_dim,
        val_info.action_dim,
        val_info.cameras,
    ):
        raise ValueError("training and validation dataset schemas differ")
    rollout_seeds = set(cfg.val_seeds()) | set(cfg.test_seeds())
    overlap = (set(train_info.seeds) | set(val_info.seeds)) & rollout_seeds
    if overlap:
        raise ValueError(f"rollout seeds overlap demonstration seeds: {sorted(overlap)[:10]}")

    stats = compute_normalization(train_info)
    train_dataset = RGBWindowDataset(train_info, cfg.policy.obs_horizon, cfg.policy.pred_horizon)
    val_dataset = RGBWindowDataset(val_info, cfg.policy.obs_horizon, cfg.policy.pred_horizon)
    experiment = args.exp or (
        f"{cfg.task.name}_rgb_unet_n{cfg.data.num_demos}_s{cfg.train.seed}"
    )
    run_dir = args.output_root.expanduser().resolve() / experiment
    checkpoint_dir = run_dir / "checkpoints"
    final_path = checkpoint_dir / "final.pt"
    resume_path = checkpoint_dir / "resume.pt"
    if final_path.is_file():
        print(f"{experiment}: final checkpoint already exists; nothing to do")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy = DiffusionPolicy(
        cfg.policy,
        cfg.vision,
        image_shape=train_info.image_shape,
        proprio_dim=train_info.proprio_dim,
        action_dim=train_info.action_dim,
        stats=stats,
    ).to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.train.lr,
        betas=(0.95, 0.999),
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(
            step, warmup_steps=cfg.train.warmup_steps, total_steps=cfg.train.total_iters
        ),
    )
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = ExponentialMovingAverage(policy, cfg.train.ema_decay)
    start_step = 0

    if resume_path.is_file():
        if args.resume == "never":
            raise FileExistsError(f"{resume_path} exists; use --resume auto or choose another experiment")
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume["config"] != cfg.to_dict():
            raise ValueError("resume checkpoint config differs from this invocation")
        policy.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        ema.load_state_dict(resume["ema"], policy)
        start_step = int(resume["step"])
        print(f"resuming {experiment} from optimizer step {start_step}")
    elif args.resume == "never" and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} is not empty")

    remaining = cfg.train.total_iters - start_step
    sampler_generator = torch.Generator().manual_seed(cfg.train.seed + start_step * 1_000_003)
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=max(cfg.train.batch_size, remaining * cfg.train.batch_size),
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.train.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        # Training workers remain alive for the whole run. Keep validation in
        # the main process rather than doubling the Slurm CPU allocation.
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    run_info = {
        "experiment": experiment,
        "config": cfg.to_dict(),
        "train_data": dataset_record(train_info),
        "val_data": dataset_record(val_info),
        "normalization": stats.to_dict(),
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        "num_params": num_params(policy),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(
        f"{experiment}: {len(train_info.episodes)} train demos / {len(train_dataset)} windows; "
        f"{len(val_info.episodes)} validation demos; {num_params(policy) / 1e6:.1f}M params; {device}"
    )

    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; saving a resumable checkpoint after this optimizer step", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)

    def save_resume(step: int) -> None:
        atomic_torch_save(
            {
                "format_version": 1,
                "config": cfg.to_dict(),
                "step": step,
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
            },
            resume_path,
        )

    metrics_path = run_dir / "metrics.jsonl"
    metrics_file = metrics_path.open("a", encoding="utf-8")
    start_time = time.time()
    last_loss = float("nan")
    policy.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step, batch in enumerate(train_loader, start=start_step + 1):
        if step > cfg.train.total_iters:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            loss = policy.compute_loss(batch["rgb"], batch["state"], batch["actions"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(policy)
        last_loss = loss.detach().item()

        record = None
        if step == 1 or step % cfg.train.log_freq == 0:
            record = {
                "step": step,
                "train_loss": last_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_s": time.time() - start_time,
            }
            print(
                f"[{step:06d}/{cfg.train.total_iters}] loss={last_loss:.5f} "
                f"lr={record['lr']:.2e} elapsed={record['elapsed_s']:.0f}s",
                flush=True,
            )

        if step in cfg.train.validation_steps or step == cfg.train.total_iters:
            validation_loss = validate(policy, ema, val_loader, device, use_amp)
            record = record or {"step": step, "train_loss": last_loss}
            record["val_loss"] = validation_loss
            print(f"[{step:06d}] fixed validation denoising loss={validation_loss:.5f}", flush=True)

        if record is not None:
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

        if step in cfg.train.checkpoint_steps:
            atomic_torch_save(
                inference_payload(policy, ema, cfg, train_info, val_info, stats, step),
                checkpoint_dir / f"step_{step:06d}.pt",
            )
        if step % cfg.train.resume_freq == 0 or stop_requested:
            save_resume(step)
        if stop_requested:
            metrics_file.close()
            return 75

    atomic_torch_save(
        inference_payload(policy, ema, cfg, train_info, val_info, stats, cfg.train.total_iters),
        final_path,
    )
    save_resume(cfg.train.total_iters)
    metrics_file.close()
    elapsed = time.time() - start_time
    summary = {
        "experiment": experiment,
        "final_step": cfg.train.total_iters,
        "final_train_loss": last_loss,
        "wall_time_s_this_invocation": elapsed,
        "peak_gpu_mem_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "peak_gpu_mem_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None
        ),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"finished {experiment} in {elapsed / 3600:.2f} h; checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
