from __future__ import annotations

import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch


def make_eval_env(config: dict, num_envs: int, backend: str):
    env = gym.make(
        config["task_name"], num_envs=num_envs, sim_backend=backend,
        obs_mode=config["obs_mode"], control_mode=config["control_mode"],
        reward_mode=config["reward_mode"], max_episode_steps=config["max_episode_steps"],
        reconfiguration_freq=0,
    )
    from mani_skill.utils.wrappers import FrameStack
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    env = FrameStack(env, num_stack=config["obs_horizon"])
    return ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=True, record_metrics=True)


def make_video_env(config: dict, output_dir):
    env = gym.make(
        config["task_name"], obs_mode=config["obs_mode"], control_mode=config["control_mode"],
        reward_mode=config["reward_mode"], sim_backend="physx_cpu", render_mode="rgb_array",
        max_episode_steps=config["max_episode_steps"],
    )
    from mani_skill.utils.wrappers import FrameStack, RecordEpisode

    env = FrameStack(env, num_stack=config["obs_horizon"])
    return RecordEpisode(
        env, output_dir=str(output_dir), save_trajectory=False, save_video=True,
        info_on_video=True, source_type="train_template", source_desc="policy evaluation rollout",
    )


def observation_tensor(observation, device):
    if isinstance(observation, dict):
        raise TypeError("task_06 expects flat state observations")
    return observation.to(device) if torch.is_tensor(observation) else torch.as_tensor(observation, device=device)


def video_observation(observation, device):
    tensor = observation_tensor(observation, device)
    if tensor.ndim == 1:
        return tensor[None, None, :]
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"Unexpected single-environment observation shape: {tuple(tensor.shape)}")


def select_action_chunk(sequence: torch.Tensor, config: dict):
    start = config["obs_horizon"] - 1
    return sequence[:, start:start + config["action_horizon"]]


def success_mask(info: dict, num_envs: int, device):
    value = info.get("success")
    if value is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return torch.as_tensor(value, dtype=torch.bool, device=device).reshape(num_envs)


def video_action(action: torch.Tensor):
    return action.detach().cpu().numpy().astype(np.float32)
