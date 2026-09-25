from __future__ import annotations

import tempfile

import torch


def smoke(task_name: str, task):
    config = task.config.TASK_CONFIG
    obs_dim, action_dim = config.get("obs_dim"), config.get("action_dim")
    if not isinstance(obs_dim, int) or not isinstance(action_dim, int):
        raise ValueError("Set config.TASK_CONFIG['obs_dim'] and ['action_dim'] before running the smoke test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in config["default_models"]:
        policy = task.policy.build_policy(model_name, obs_dim, action_dim, config).to(device)
        observations = torch.randn(2, config["obs_horizon"], obs_dim, device=device)
        actions = torch.randn(2, config["prediction_horizon"], action_dim, device=device).clamp(-1, 1)
        loss = task.training.compute_loss(policy, observations, actions, config)
        loss.backward()
        assert torch.isfinite(loss), f"{model_name} produced a non-finite loss"
        prediction = task.policy.predict_actions(policy, observations, 2, config)
        expected = (2, config["prediction_horizon"], action_dim)
        assert tuple(prediction.shape) == expected, f"{model_name}: expected {expected}, got {tuple(prediction.shape)}"
        print(f"{model_name}: forward/backward/inference passed ({sum(p.numel() for p in policy.parameters()):,} params)")

    backend = "physx_cuda" if device.type == "cuda" else "physx_cpu"
    env = task.environment.make_eval_env(config, 1, backend)
    try:
        observations, _ = env.reset(seed=[606])
        observations = task.environment.observation_tensor(observations, device)
        action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
        observations, _, _, _, info = env.step(action)
        task.environment.success_mask(info, 1, device)
        print(f"{backend} reset/step passed")
    finally:
        env.close()

    with tempfile.TemporaryDirectory(prefix=f"{task_name}-video-smoke-") as directory:
        env = task.environment.make_video_env(config, directory)
        try:
            observations, _ = env.reset(seed=607)
            for _ in range(min(3, config["max_episode_steps"])):
                observations, _, terminated, truncated, _ = env.step(
                    task.environment.video_action(torch.zeros(action_dim))
                )
                if terminated or truncated:
                    break
        finally:
            env.close()
    print(f"Video recorder and ffmpeg passed; task={task_name}; device={device}")
