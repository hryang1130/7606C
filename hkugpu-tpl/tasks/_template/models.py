from torch import nn


class ModelA(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # TODO: Define your first network architecture.
        raise NotImplementedError("Fill in ModelA")

    def forward(self, noisy_actions, timesteps, observations):
        # TODO: Predict the training target for one denoising step.
        raise NotImplementedError


class ModelB(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # TODO: Define a genuinely different architecture for comparison.
        raise NotImplementedError("Fill in ModelB")

    def forward(self, noisy_actions, timesteps, observations):
        raise NotImplementedError


MODEL_REGISTRY = {"model_a": ModelA, "model_b": ModelB}


def build_denoiser(name, obs_dim, action_dim, config):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](obs_dim, action_dim, config)
