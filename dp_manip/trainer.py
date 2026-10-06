"""The single training pipeline shared by every experiment entry point.

``scripts/train_dp.py`` (Slurm sweep cells) and ``scripts/run_experiment.py``
(task + experiment + value) differ only in how they resolve a :class:`Config`;
both hand the resolved config to :func:`run_training`. Experiments must not grow
their own trainer, so any new experiment is a config file, never a pipeline.
The restartable-state utilities (EMA, sampler, RNG capture, checkpoint payloads)
live in ``dp_manip.training``.
"""

from __future__ import annotations

import json
import math
import platform
import random
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config as config_lib
from . import finetune as finetune_lib
from .completion import RunState, completion_state
from .config import Config
from .data import (
    DatasetInfo,
    NormalizationStats,
    ObservationWindowDataset,
    compute_normalization,
    read_dataset_info,
)
from .finetune import FinetuneSpec
from .metadata import dataset_fingerprint, file_sha256, git_revision
from .policy import (
    DiffusionPolicy,
    adapt_legacy_state_dict,
    load_policy_state_dict,
    num_params,
)
from .training import (
    ExponentialMovingAverage,
    StepSeededIndexSampler,
    atomic_torch_save,
    averaged_state_dict,
    constant_warmup,
    cosine_warmup,
    resume_checkpoint,
    set_rng_state,
)


ROOT = Path(__file__).resolve().parents[1]


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


def check_dataset(
    info: DatasetInfo, cfg: Config, split: str, seed_range: tuple[int, int] | None = None
) -> None:
    """Check env, control mode and seeds; ``seed_range`` replaces the expert rule
    for rollout datasets (failure-aware plan §3.5)."""
    if info.env_id != cfg.task.env_id:
        raise ValueError(f"{split} dataset env_id={info.env_id!r}, expected {cfg.task.env_id!r}")
    if info.control_mode != cfg.task.control_mode:
        raise ValueError(
            f"{split} dataset control_mode={info.control_mode!r}, expected {cfg.task.control_mode!r}"
        )
    if info.obs_mode != cfg.task.obs_mode:
        raise ValueError(f"{split} dataset observation mode differs from task.obs_mode")
    if seed_range is not None:
        finetune_lib.check_seed_range(info, seed_range, split)
        return
    if split == "train" and any(seed >= 4_000 for seed in info.seeds):
        raise ValueError("training demonstrations must use seeds below 4000")
    if split == "val" and any(not 4_000 <= seed < 5_000 for seed in info.seeds):
        raise ValueError("validation demonstrations must use seeds in [4000, 5000)")


def check_horizon(cfg: Config, *infos: DatasetInfo) -> None:
    """Refuse an evaluation horizon shorter than the longest demonstration.

    Training itself never uses ``task.max_episode_steps``, but every checkpoint
    records it and evaluation stops there: PlaceSphere's demonstrations are
    90-150 steps, and its registered 50-step default made every evaluation fail.
    """
    lengths = [episode.length for info in infos for episode in info.episodes]
    if not lengths or cfg.task.max_episode_steps >= max(lengths):
        return
    suggested = math.ceil(max(max(lengths), 2 * sum(lengths) / len(lengths)) / 50) * 50
    raise ValueError(
        f"task.max_episode_steps={cfg.task.max_episode_steps} is shorter than the longest demonstration "
        f"({max(lengths)} steps): every evaluation episode would stop before the expert finishes. "
        f"Set max_episode_steps in configs/tasks/{cfg.task.name}.toml, e.g. {suggested} "
        "(2 x mean demonstration length, rounded up to 50; docs/final-plan.md §1)"
    )


def dataset_record(info: DatasetInfo) -> dict:
    return {
        "path": str(info.path),
        "env_id": info.env_id,
        "control_mode": info.control_mode,
        "num_demos": len(info.episodes),
        "num_transitions": info.num_transitions,
        "seeds": info.seeds,
        "image_shape": list(info.image_shape) if info.image_shape is not None else None,
        "obs_mode": info.obs_mode,
        "observation_key": info.observation_key,
        "proprio_dim": info.proprio_dim,
        "action_dim": info.action_dim,
        "cameras": list(info.cameras),
        "rgb_env_info": info.rgb_env_info,
        "fingerprint": dataset_fingerprint(info),
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
                        batch.get("rgb"), batch["proprio"], batch["actions"], generator=generator
                    )
                count = batch["actions"].shape[0]
                total_loss += loss.detach().item() * count
                total_items += count
    policy.train(was_training)
    return total_loss / total_items


def inference_payload(
    policy: DiffusionPolicy,
    ema: ExponentialMovingAverage,
    cfg: Config,
    train_info: DatasetInfo,
    val_info: DatasetInfo,
    stats: NormalizationStats,
    step: int,
    finetune: Mapping[str, Any] | None = None,
) -> dict:
    payload = {
        "format_version": 2,
        "model": averaged_state_dict(policy, ema),
        "config": cfg.to_dict(),
        "train_data": dataset_record(train_info),
        "val_data": dataset_record(val_info),
        "normalization": stats.to_dict(),
        "step": step,
    }
    if finetune is not None:
        payload["finetune"] = dict(finetune)
    return payload


def run_training(
    cfg: Config,
    *,
    output_root: Path,
    run_name: str | None = None,
    device: str = "cuda",
    resume: str = "auto",
    experiment_context: Mapping[str, Any] | None = None,
    finetune: FinetuneSpec | None = None,
) -> int:
    """Train, checkpoint and validate one resolved configuration.

    ``experiment_context`` carries the declared cell metadata (experiment
    name/variable/value and the control hash) from the entry point into
    ``run.json``; it is ``None`` for runs outside an experiment grid.

    ``finetune`` starts from a baseline checkpoint instead of from scratch
    (``dp_manip.finetune``): its weights and normalization are reused, the
    spec's modules are frozen, and the rollout datasets must come from that
    checkpoint. Without it, training is exactly the baseline pipeline.

    Returns ``0`` on completion, ``75`` after a scheduler signal wrote
    ``resume.pt`` (the Slurm requeue convention), or ``0`` immediately when
    ``final.pt`` for the same config already exists.
    """
    cfg.validate()
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; cluster training must run on a GPU node")
    seed_everything(cfg.train.seed)

    init_checkpoint: dict | None = None
    finetune_record: dict | None = None
    if finetune is not None:
        init_path = Path(finetune.init_checkpoint).expanduser().resolve()
        init_checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        finetune_lib.check_locked_sections(cfg, config_lib.from_recorded(init_checkpoint["config"]))
        finetune_record = {
            "spec": finetune.to_dict(),
            "init_checkpoint_sha256": file_sha256(init_path),
            "init_checkpoint_step": int(init_checkpoint["step"]),
        }

    data_root = Path(cfg.data.root).expanduser()
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    train_info = read_dataset_info(
        resolve_data_path(data_root, cfg.data.train_path), cfg.data.num_demos,
        obs_mode=cfg.task.obs_mode,
    )
    val_info = read_dataset_info(
        resolve_data_path(data_root, cfg.data.val_path), cfg.data.val_num_demos,
        obs_mode=cfg.task.obs_mode,
    )
    check_dataset(train_info, cfg, "train", finetune.train_seed_range if finetune else None)
    check_dataset(val_info, cfg, "val", finetune.val_seed_range if finetune else None)
    if finetune is None:
        # Fine-tuning reuses the baseline's recorded config, whose horizon the
        # failure-aware study overrides at evaluation time (failure-aware plan §13).
        check_horizon(cfg, train_info, val_info)
    if init_checkpoint is not None and finetune_record is not None:
        for split, info in (("train", train_info), ("val", val_info)):
            if finetune.require_rollout_source:
                finetune_lib.check_rollout_source(info, finetune_record["init_checkpoint_sha256"], split)
            finetune_lib.check_schema(info, init_checkpoint["train_data"], split)
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

    # Fine-tuning must stay in the baseline's normalized action/proprio
    # coordinates (failure-aware plan §4.2): never recompute statistics.
    stats = (
        NormalizationStats.from_dict(init_checkpoint["normalization"])
        if init_checkpoint is not None
        else compute_normalization(train_info)
    )
    preload_start = time.time()
    train_dataset = ObservationWindowDataset(
        train_info, cfg.policy.obs_horizon, cfg.policy.pred_horizon, preload=cfg.data.preload
    )
    val_dataset = ObservationWindowDataset(
        val_info, cfg.policy.obs_horizon, cfg.policy.pred_horizon, preload=cfg.data.preload
    )
    if cfg.data.preload:
        preloaded_gb = (train_dataset.preloaded_bytes + val_dataset.preloaded_bytes) / 2**30
        print(
            f"preloaded {len(train_info.episodes)} train + {len(val_info.episodes)} val episodes "
            f"in {time.time() - preload_start:.1f} s ({preloaded_gb:.3f} GiB)"
        )
    experiment = run_name or config_lib.default_run_name(cfg)
    run_dir = output_root.expanduser().resolve() / experiment
    checkpoint_dir = run_dir / "checkpoints"
    final_path = checkpoint_dir / "final.pt"
    resume_path = checkpoint_dir / "resume.pt"
    # The planner uses this same decision, so an existing final.pt is never
    # reused for a different configuration by either entry point.
    finished = completion_state(cfg, run_dir, finetune=finetune_record)
    if finished.state is RunState.CONFLICT:
        raise FileExistsError(
            f"{run_dir} holds a finished run with a different config; "
            "choose another --exp or output root"
        )
    if finished.state is RunState.COMPLETED:
        print(f"{experiment}: final checkpoint already exists; nothing to do")
        return 0
    # Decided before the directories below are created, so --resume never
    # accepts a fresh run directory and rejects only earlier content.
    had_content = run_dir.is_dir() and any(run_dir.iterdir())
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    policy = DiffusionPolicy(
        cfg.policy,
        cfg.vision,
        cfg.diffusion,
        image_shape=train_info.image_shape,
        proprio_dim=train_info.proprio_dim,
        action_dim=train_info.action_dim,
        stats=stats,
        obs_mode=cfg.task.obs_mode,
    ).to(device)
    frozen: list[str] = []
    if init_checkpoint is not None and finetune is not None:
        load_policy_state_dict(policy, init_checkpoint["model"])
        # Freeze before the optimizer and the EMA are built: both then see only
        # the trainable parameters, and frozen weights stay bit-identical.
        frozen = finetune_lib.freeze_modules(policy, finetune.frozen_modules)
        init_checkpoint = None  # release the baseline weights
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
    )
    if finetune is not None and finetune.lr_schedule == "constant_with_warmup":
        def lr_factor(step: int) -> float:
            return constant_warmup(step, warmup_steps=cfg.train.warmup_steps)
    else:
        def lr_factor(step: int) -> float:
            return cosine_warmup(
                step, warmup_steps=cfg.train.warmup_steps, total_steps=cfg.train.total_iters
            )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = ExponentialMovingAverage(policy, cfg.ema.decay)
    start_step = 0

    if resume_path.is_file():
        if resume == "never":
            raise FileExistsError(f"{resume_path} exists; use --resume auto or choose another experiment")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if not config_lib.same_run(config_lib.from_recorded(checkpoint["config"]), cfg):
            raise ValueError("resume checkpoint config differs from this invocation")
        if checkpoint.get("finetune") != finetune_record:
            raise ValueError("resume checkpoint fine-tuning record differs from this invocation")
        load_policy_state_dict(policy, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema_state = checkpoint["ema"]
        # Resume checkpoints written before the ObservationEncoder extraction
        # name trainable parameters without the ``observation_encoder.`` prefix.
        ema_state["shadow"] = adapt_legacy_state_dict(ema_state["shadow"])
        ema.load_state_dict(ema_state, policy)
        start_step = int(checkpoint["step"])
        if set_rng_state(checkpoint.get("rng")):
            print("restored Python/NumPy/torch CPU/CUDA RNG state from resume.pt")
        else:
            print(
                "warning: resume.pt predates RNG capture; the post-resume trajectory "
                "will not match a continuous run"
            )
        print(f"resuming {experiment} from optimizer step {start_step}")
    elif resume == "never" and had_content:
        raise FileExistsError(f"{run_dir} is not empty")

    sampler = StepSeededIndexSampler(
        len(train_dataset),
        batch_size=cfg.train.batch_size,
        seed=cfg.train.seed,
        first_step=start_step + 1,
        last_step=cfg.train.total_iters,
    )
    # Worker seeds come from a dedicated generator rather than the global torch
    # RNG, so constructing the loader cannot shift the restored stream.
    worker_generator = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        generator=worker_generator,
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
        # Declared experiment cell (name/variable/value/seed/control_hash), so a
        # run can be audited without re-deriving the grid.
        "experiment_context": experiment_context,
        "config": cfg.to_dict(),
        "git": git_revision(ROOT),
        "train_data": dataset_record(train_info),
        "val_data": dataset_record(val_info),
        # The data-size experiment compares nested subsets; record exactly which
        # demonstrations were selected so a run can be audited after the fact.
        "data_selection": {
            "rule": "episode_seed_ascending_prefix",
            "train": {"num_demos": len(train_info.episodes), "demo_seeds": train_info.seeds},
            "val": {"num_demos": len(val_info.episodes), "demo_seeds": val_info.seeds},
        },
        "normalization": stats.to_dict(),
        **(
            {"finetune": finetune_record, "frozen_parameters": frozen}
            if finetune_record is not None
            else {}
        ),
        "num_train_windows": len(train_dataset),
        "num_val_windows": len(val_dataset),
        # The sampler is derived from (seed, step) rather than from a stateful
        # stream, so run-to-run audits can reproduce the demo order.
        "sampler": {
            "scheme": "step_seeded_with_replacement",
            "seed": cfg.train.seed,
            "batch_size": cfg.train.batch_size,
        },
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
            resume_checkpoint(
                config=cfg.to_dict(),
                step=step,
                model=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                extra={"finetune": finetune_record} if finetune_record is not None else None,
            ),
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
            loss = policy.compute_loss(batch.get("rgb"), batch["proprio"], batch["actions"])
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
                inference_payload(
                    policy, ema, cfg, train_info, val_info, stats, step, finetune_record
                ),
                checkpoint_dir / f"step_{step:06d}.pt",
            )
        if step % cfg.train.resume_freq == 0 or stop_requested:
            save_resume(step)
        if stop_requested:
            metrics_file.close()
            return 75

    atomic_torch_save(
        inference_payload(
            policy, ema, cfg, train_info, val_info, stats, cfg.train.total_iters, finetune_record
        ),
        final_path,
    )
    save_resume(cfg.train.total_iters)
    metrics_file.close()
    elapsed = time.time() - start_time
    summary = {
        "experiment": experiment,
        "control_hash": (experiment_context or {}).get("control_hash"),
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
