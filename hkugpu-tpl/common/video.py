from __future__ import annotations

from pathlib import Path

import torch


def record_video(task_name: str, task, args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["task"] != task_name:
        raise ValueError(f"Checkpoint task {checkpoint['task']!r} does not match requested task {task_name!r}")
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = task.policy.build_policy(checkpoint["model_name"], checkpoint["obs_dim"],
                                      checkpoint["action_dim"], config).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        if not args.no_compile and hasattr(task.policy, "compile_for_inference"):
            policy = task.policy.compile_for_inference(policy)

    output = args.output or Path(args.checkpoint).parent / "videos"
    output.mkdir(parents=True, exist_ok=True)
    env = task.environment.make_video_env(config, output)
    try:
        for episode in range(args.episodes):
            observations, _ = env.reset(seed=args.seed + episode)
            done = False
            success = False
            while not done:
                policy_obs = task.environment.video_observation(observations, device)
                use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    sequence = task.policy.predict_actions(policy, policy_obs, args.inference_steps, config)
                actions = task.environment.select_action_chunk(sequence, config)
                for index in range(actions.shape[1]):
                    observations, _, terminated, truncated, info = env.step(
                        task.environment.video_action(actions[0, index])
                    )
                    success = success or bool(task.environment.success_mask(info, 1, device)[0].item())
                    done = bool(terminated or truncated)
                    if done:
                        break
            print(f"episode={episode} seed={args.seed + episode} success={success}")
    finally:
        env.close()
    print(f"Videos saved under {output}")
