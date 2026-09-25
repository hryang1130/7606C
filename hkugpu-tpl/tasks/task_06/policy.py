from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler

from .models import make_denoiser


class DiffusionPolicy(torch.nn.Module):
    def __init__(self, architecture: str, obs_dim: int, action_dim: int, pred_horizon: int,
                 obs_horizon: int = 2, train_steps: int = 100):
        super().__init__()
        self.architecture = architecture
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.pred_horizon, self.obs_horizon = pred_horizon, obs_horizon
        self.denoiser = make_denoiser(architecture, obs_dim * obs_horizon, action_dim, pred_horizon)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=train_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def predict_actions(self, observations: torch.Tensor, inference_steps: int = 12,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        return self.predict(observations, inference_steps, generator)

    def loss(self, observations: torch.Tensor, actions: torch.Tensor,
             timesteps: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> torch.Tensor:
        batch = actions.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (batch,), device=actions.device, dtype=torch.long,
            )
        noisy = self.noise_scheduler.add_noise(actions, noise, timesteps)
        prediction = self.denoiser(noisy, timesteps, observations)
        return torch.nn.functional.mse_loss(prediction, noise)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.loss(observations, actions)

    @torch.inference_mode()
    def predict(self, observations: torch.Tensor, inference_steps: int = 12,
                generator: torch.Generator | None = None) -> torch.Tensor:
        scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        scheduler.set_timesteps(inference_steps, device=observations.device)
        sample = torch.randn(
            (observations.shape[0], self.pred_horizon, self.action_dim),
            device=observations.device, dtype=observations.dtype, generator=generator,
        )
        for timestep in scheduler.timesteps:
            times = timestep.expand(observations.shape[0])
            noise = self.denoiser(sample, times, observations)
            sample = scheduler.step(noise, timestep, sample, generator=generator).prev_sample
        return sample.clamp(-1, 1)


def build_policy(model_name: str, obs_dim: int, action_dim: int, config: dict) -> DiffusionPolicy:
    return DiffusionPolicy(
        model_name, obs_dim, action_dim, config["prediction_horizon"],
        config["obs_horizon"], config.get("diffusion_train_steps", 100),
    )


def compile_for_inference(policy: DiffusionPolicy) -> DiffusionPolicy:
    policy.denoiser = torch.compile(policy.denoiser, mode="reduce-overhead")
    return policy


def predict_actions(policy: DiffusionPolicy, observations: torch.Tensor, inference_steps: int, config: dict):
    return policy.predict_actions(observations, inference_steps)
