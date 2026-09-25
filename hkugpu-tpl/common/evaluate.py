from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


def evaluate(task_name: str, task, args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["task"] != task_name:
        raise ValueError(f"Checkpoint task {checkpoint['task']!r} does not match requested task {task_name!r}")
    config = checkpoint["config"]
    torch.manual_seed(args.first_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.first_seed)
    device = torch.device("cuda" if args.backend == "physx_cuda" else "cpu")
    policy = task.policy.build_policy(checkpoint["model_name"], checkpoint["obs_dim"],
                                      checkpoint["action_dim"], config).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    if device.type == "cuda" and not args.no_compile and hasattr(task.policy, "compile_for_inference"):
        try:
            policy = task.policy.compile_for_inference(policy)
        except Exception as error:
            print(f"Inference compilation unavailable; using eager model: {error}")

    env_count = min(args.num_envs, args.episodes)
    env = None
    rows = []
    try:
        env = task.environment.make_eval_env(config, env_count, args.backend)
        for start in range(args.first_seed, args.first_seed + args.episodes, env_count):
            seeds = list(range(start, min(start + env_count, args.first_seed + args.episodes)))
            active = len(seeds)
            if active != env_count:
                env.close()
                env = task.environment.make_eval_env(config, active, args.backend)
            observations, _ = env.reset(seed=seeds)
            success = torch.zeros(active, dtype=torch.bool, device=device)
            elapsed = 0
            while elapsed < config["max_episode_steps"]:
                observations = task.environment.observation_tensor(observations, device)
                use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    sequence = task.policy.predict_actions(policy, observations, args.inference_steps, config)
                action_chunk = task.environment.select_action_chunk(sequence, config)
                count = min(action_chunk.shape[1], config["max_episode_steps"] - elapsed)
                for index in range(count):
                    observations, _, _, _, info = env.step(action_chunk[:, index].float())
                    elapsed += 1
                    success |= task.environment.success_mask(info, active, device)
            values = success.cpu().numpy().astype(bool)
            rows.extend({"seed": seed, "success": bool(value), "checkpoint": str(args.checkpoint)}
                        for seed, value in zip(seeds, values))
            print(f"Seeds {seeds[0]}-{seeds[-1]}: {int(values.sum())}/{active} success")
    finally:
        if env is not None:
            env.close()

    output = args.output or Path(args.checkpoint).parent / f"evaluation_{args.first_seed}_{args.episodes}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("seed", "success", "checkpoint"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "task": task_name, "model": checkpoint["model_name"], "training_seed": checkpoint["seed"],
        "episodes": len(rows), "successes": int(sum(row["success"] for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "first_seed": args.first_seed, "inference_steps": args.inference_steps,
        "simulation_backend": args.backend,
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {output} and {summary_path}")
