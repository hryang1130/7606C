"""Training utilities kept independent of the cluster entry point."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """EMA with the same early-step decay ramp used by the prior DP baseline."""

    def __init__(self, model: nn.Module, target_decay: float):
        self.target_decay = target_decay
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = min(self.target_decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].lerp_(parameter.detach(), 1.0 - decay)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        parameters = dict(model.named_parameters())
        backup = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in backup.items():
                    parameters[name].copy_(value)

    def state_dict(self) -> dict:
        return {
            "target_decay": self.target_decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict, model: nn.Module) -> None:
        self.target_decay = float(state["target_decay"])
        self.num_updates = int(state["num_updates"])
        parameters = dict(model.named_parameters())
        self.shadow = {
            name: value.to(device=parameters[name].device, dtype=parameters[name].dtype)
            for name, value in state["shadow"].items()
        }


def cosine_warmup(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def averaged_state_dict(model: nn.Module, ema: ExponentialMovingAverage) -> dict[str, torch.Tensor]:
    with ema.average_parameters(model):
        return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
