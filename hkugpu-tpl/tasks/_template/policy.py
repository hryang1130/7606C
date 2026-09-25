import torch
from torch import nn

from .models import build_denoiser


class TaskPolicy(nn.Module):
    def __init__(self, model_name, obs_dim, action_dim, config):
        super().__init__()
        self.denoiser = build_denoiser(model_name, obs_dim, action_dim, config)
        # TODO: Construct your algorithm's scheduler/normalizers and save needed dimensions.

    def forward(self, observations, actions):
        # TODO: Return a scalar loss for one minibatch.
        raise NotImplementedError

    @torch.inference_mode()
    def predict_actions(self, observations, inference_steps, config):
        # TODO: Return a tensor [batch, prediction_horizon, action_dim].
        raise NotImplementedError


def build_policy(model_name, obs_dim, action_dim, config):
    return TaskPolicy(model_name, obs_dim, action_dim, config)


def predict_actions(policy, observations, inference_steps, config):
    return policy.predict_actions(observations, inference_steps, config)


def compile_for_inference(policy):
    # TODO: If appropriate, compile the inference network; otherwise return policy unchanged.
    policy.denoiser = torch.compile(policy.denoiser, mode="reduce-overhead")
    return policy
