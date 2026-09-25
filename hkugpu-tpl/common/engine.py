from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def cache_windows(dataset, device):
    samples = [dataset[index] for index in range(len(dataset))]
    observations, actions = zip(*samples)
    return (torch.stack(observations).to(device, non_blocking=True),
            torch.stack(actions).to(device, non_blocking=True))


@contextmanager
def use_ema(model, shadow_parameters):
    original = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.no_grad():
        for parameter, shadow in zip(model.parameters(), shadow_parameters):
            parameter.copy_(shadow)
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value in zip(model.parameters(), original):
                parameter.copy_(value)


def _save_checkpoint(path, model, task_name, model_name, config, args, step, validation_loss,
                     obs_dim, action_dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "model": model.state_dict(),
        "task": task_name,
        "model_name": model_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "config": config,
        "step": step,
        "validation_loss": validation_loss,
        "seed": args.seed,
        "train_args": vars(args),
    }, temporary)
    temporary.replace(path)


def train(task_name: str, task, args):
    config = task.config.TASK_CONFIG
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        print(f"Training on {properties.name} ({properties.total_memory / 2**30:.1f} GiB)")
    else:
        print("WARNING: CUDA unavailable; using CPU for a smoke run")

    dataset_path = args.dataset or config["dataset"]
    training_data, validation_data = task.data.load_datasets(
        dataset_path, args.seed, args.validation_fraction or config["validation_fraction"],
        args.num_demos, config,
    )
    train_obs, train_actions = cache_windows(training_data, device)
    val_obs, val_actions = cache_windows(validation_data, device)
    obs_dim, action_dim = train_obs.shape[-1], train_actions.shape[-1]
    if args.model not in config["default_models"]:
        raise ValueError(f"Unknown model {args.model!r}; available: {config['default_models']}")
    policy = task.policy.build_policy(args.model, obs_dim, action_dim, config).to(device)

    amp_dtype = None
    if device.type == "cuda" and args.amp != "off":
        if args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()):
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
    if args.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but the GPU does not support it")

    compiled = False
    train_policy = policy
    if device.type == "cuda" and not args.no_compile:
        try:
            train_policy = torch.compile(policy, mode="reduce-overhead")
            compiled = True
        except Exception as error:
            print(f"torch.compile setup failed; using eager training: {error}")
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, betas=(0.95, 0.999),
        weight_decay=1e-6, fused=(device.type == "cuda"),
    )
    warmup_steps = min(500, max(1, args.steps // 20))

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, args.steps - warmup_steps))
        return 0.5 * (1 + np.cos(np.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    ema_parameters = [parameter.detach().clone() for parameter in policy.parameters()]
    run_dir = Path(args.output) / task_name / args.model / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    training_episodes = getattr(training_data, "episode_count", len(training_data))
    validation_episodes = getattr(validation_data, "episode_count", len(validation_data))
    (run_dir / "run_config.json").write_text(json.dumps({
        "task": task_name, "model": args.model, "seed": args.seed,
        "config": config, "steps": args.steps, "batch_size": args.batch_size,
        "dataset": str(dataset_path), "compile": compiled,
        "amp": str(amp_dtype), "train_episodes": training_episodes,
        "validation_episodes": validation_episodes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Train/validation episodes: {training_episodes}/{validation_episodes}")
    print(f"Model={args.model}; parameters={sum(p.numel() for p in policy.parameters()):,}; compile={compiled}; AMP={amp_dtype}")

    best_loss = float("inf")
    started = time.time()
    for step in range(1, args.steps + 1):
        policy.train()
        indices = torch.randint(len(train_actions), (args.batch_size,), device=device)
        observations, actions = train_obs[indices], train_actions[indices]
        optimizer.zero_grad(set_to_none=True)
        amp_context = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)
        with amp_context:
            loss = task.training.compute_loss(train_policy, observations, actions, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            decay = min(0.995, (1 + step) / (10 + step))
            for shadow, parameter in zip(ema_parameters, policy.parameters()):
                shadow.lerp_(parameter.detach(), 1 - decay)

        record = None
        if step % args.log_every == 0 or step == 1:
            record = {
                "step": step, "train_loss": float(loss.detach()),
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
        if step % args.validate_every == 0 or step == args.steps:
            with use_ema(policy, ema_parameters):
                rng_devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(91673)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(91673)
                    policy.eval()
                    losses = []
                    with torch.inference_mode():
                        for start in range(0, len(val_actions), args.batch_size):
                            batch_obs = val_obs[start:start + args.batch_size]
                            batch_actions = val_actions[start:start + args.batch_size]
                            amp_context = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)
                            with amp_context:
                                losses.append(task.training.compute_loss(policy, batch_obs, batch_actions, config).float())
                    validation_loss = torch.stack(losses).mean().item()
                policy.train()
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    _save_checkpoint(run_dir / "best.pt", policy, task_name, args.model, config, args,
                                     step, best_loss, obs_dim, action_dim)
                _save_checkpoint(run_dir / "latest.pt", policy, task_name, args.model, config, args,
                                 step, validation_loss, obs_dim, action_dim)
            record = record or {"step": step, "train_loss": None, "learning_rate": lr_scheduler.get_last_lr()[0]}
            record["validation_loss"] = validation_loss
            print(f"step={step}/{args.steps} validation_loss={validation_loss:.6f}")
        elif args.save_every > 0 and step % args.save_every == 0:
            with use_ema(policy, ema_parameters):
                _save_checkpoint(run_dir / "latest.pt", policy, task_name, args.model, config, args,
                                 step, best_loss, obs_dim, action_dim)

        if record is not None:
            record.setdefault("elapsed_seconds", time.time() - started)
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            if step % args.log_every == 0 or step == 1:
                print(f"step={step}/{args.steps} loss={record['train_loss']:.6f}")
    print(f"Best validation diffusion loss={best_loss:.6f}; output={run_dir}")
