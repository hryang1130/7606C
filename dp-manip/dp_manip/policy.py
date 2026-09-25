"""Diffusion Policy agent (state observations, 1D conditional UNet, DDPM).

Adapted from ManiSkill examples/baselines/diffusion_policy/train.py
(haosulab/ManiSkill@62ff3a5, Apache-2.0). Changes from the baseline:
- dimensions come from the dataset instead of an env object, so the agent can
  be built and checked without ManiSkill installed;
- the model works in normalized action space, and `ActionNormalizer` maps
  sampled actions back to the env's units in `get_action`;
- checkpoints carry the resolved config and normalizer statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .conditional_unet1d import ConditionalUnet1D
from .config import PolicyConfig
from .data import ActionNormalizer


class DiffusionPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, obs_dim: int, act_dim: int, normalizer: ActionNormalizer):
        super().__init__()
        self.obs_horizon = cfg.obs_horizon
        self.act_horizon = cfg.act_horizon
        self.pred_horizon = cfg.pred_horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.normalizer = normalizer

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=cfg.obs_horizon * obs_dim,
            diffusion_step_embed_dim=cfg.diffusion_step_embed_dim,
            down_dims=cfg.unet_dims,
            n_groups=cfg.n_groups,
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=cfg.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",  # baseline: large effect on performance
            clip_sample=True,  # samples are clipped to [-1, 1] -> actions must be normalized
            prediction_type="epsilon",
        )

    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq (B, obs_horizon, obs_dim) raw; action_seq (B, pred_horizon, act_dim) normalized."""
        B = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=obs_seq.device
        ).long()
        noisy = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def get_action(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """obs_seq (B, obs_horizon, obs_dim) -> (B, act_horizon, act_dim) in env units."""
        B = obs_seq.shape[0]
        obs_cond = obs_seq.flatten(start_dim=1)
        sample = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)
        # Inference uses all training diffusion steps, so set_timesteps is not needed.
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(sample=sample, timestep=k, global_cond=obs_cond)
            sample = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=sample).prev_sample
        start = self.obs_horizon - 1
        chunk = sample[:, start : start + self.act_horizon]
        return self.normalizer.unnormalize(chunk)


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def save_checkpoint(path, *, policy: DiffusionPolicy, ema_policy: DiffusionPolicy,
                    config: dict, iteration: int, extra: dict | None = None) -> None:
    torch.save({
        "policy": policy.state_dict(),
        "ema_policy": ema_policy.state_dict(),
        "normalizer": policy.normalizer.state_dict(),
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "config": config,
        "iteration": iteration,
        "extra": extra or {},
    }, path)


def load_checkpoint(path, device: torch.device, use_ema: bool = True):
    """Return (policy, config dict, checkpoint dict). EMA weights by default, as the baseline evaluates."""
    from .config import from_dict

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = from_dict(ckpt["config"])
    normalizer = ActionNormalizer.from_state_dict(ckpt["normalizer"])
    policy = DiffusionPolicy(cfg.policy, ckpt["obs_dim"], ckpt["act_dim"], normalizer).to(device)
    policy.load_state_dict(ckpt["ema_policy" if use_ema else "policy"])
    policy.eval()
    return policy, cfg, ckpt
