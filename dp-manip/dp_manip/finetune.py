"""Fine-tuning a baseline checkpoint on rollout datasets (failure-aware plan §4).

Fine-tuning is not a second trainer: :func:`dp_manip.trainer.run_training`
takes an optional :class:`FinetuneSpec` and then (1) builds the policy from the
baseline checkpoint's weights and normalization instead of computing new
statistics, (2) freezes the named modules before the optimizer and EMA see
them, and (3) checks that the datasets were collected from that exact
checkpoint and lie in the declared rollout seed ranges. Everything that fixes
the action and diffusion coordinate system must equal the baseline's (§4.2),
so the fine-tuned model and the baseline stay directly comparable.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .config import Config
from .invariants import config_differences

if TYPE_CHECKING:  # typing only: importing torch-backed modules is deferred
    import torch.nn as nn

    from .data import DatasetInfo

LR_SCHEDULES = ("cosine_with_warmup", "constant_with_warmup")
# Config sections that define the model, its coordinate system and its
# evaluation seeds. Only [data], [train] and [ema] may differ from the baseline.
LOCKED_SECTIONS = ("task", "vision", "policy", "diffusion", "eval")


@dataclass(frozen=True)
class FinetuneSpec:
    init_checkpoint: str
    frozen_modules: tuple[str, ...]
    lr_schedule: str
    train_seed_range: tuple[int, int]
    val_seed_range: tuple[int, int]
    # False for datasets that mix expert demonstrations with the checkpoint's own
    # takeover corrections; their entry point checks provenance itself.
    require_rollout_source: bool = True

    def __post_init__(self) -> None:
        if self.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {LR_SCHEDULES}, got {self.lr_schedule!r}")
        for name in ("train_seed_range", "val_seed_range"):
            start, end = getattr(self, name)
            if not 0 <= start < end:
                raise ValueError(f"{name} must satisfy 0 <= start < end")

    def to_dict(self) -> dict[str, Any]:
        """JSON-shaped record (tuples become lists), comparable with ``run.json``."""
        record = json.loads(json.dumps(dataclasses.asdict(self)))
        if record["require_rollout_source"]:
            del record["require_rollout_source"]  # records written before the field existed
        return record


def check_locked_sections(cfg: Config, baseline: Config) -> None:
    """Reject a fine-tuning config that changes the baseline's model or seeds."""
    ours, theirs = cfg.to_dict(), baseline.to_dict()
    differences = config_differences(
        {name: theirs[name] for name in LOCKED_SECTIONS},
        {name: ours[name] for name in LOCKED_SECTIONS},
    )
    if differences:
        details = ", ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in sorted(differences.items()))
        raise ValueError(f"fine-tuning must keep the baseline's {LOCKED_SECTIONS} sections: {details}")


def check_seed_range(info: "DatasetInfo", bounds: Sequence[int], split: str) -> None:
    start, end = bounds
    outside = sorted(seed for seed in info.seeds if not start <= seed < end)
    if outside:
        raise ValueError(f"{split} rollout seeds must lie in [{start}, {end}): {outside[:10]}")


def check_rollout_source(info: "DatasetInfo", checkpoint_sha256: str, split: str) -> None:
    """The dataset must have been collected from the checkpoint being fine-tuned (§4.1)."""
    sidecar = info.path.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    source = (meta.get("rollout_provenance") or {}).get("source_checkpoint") or {}
    if source.get("sha256") != checkpoint_sha256:
        raise ValueError(
            f"{split} dataset {info.path} was collected from checkpoint "
            f"{source.get('sha256')!r}, not from the fine-tuning init {checkpoint_sha256!r}"
        )


def check_schema(info: "DatasetInfo", baseline_train_data: Mapping[str, Any], split: str) -> None:
    ours = (list(info.image_shape), info.proprio_dim, info.action_dim, list(info.cameras))
    theirs = (
        list(baseline_train_data["image_shape"]),
        int(baseline_train_data["proprio_dim"]),
        int(baseline_train_data["action_dim"]),
        list(baseline_train_data["cameras"]),
    )
    if ours != theirs:
        raise ValueError(f"{split} dataset schema {ours} differs from the baseline's {theirs}")


def freeze_modules(model: "nn.Module", prefixes: Sequence[str]) -> list[str]:
    """Set ``requires_grad=False`` under each module prefix; return frozen names."""
    frozen: list[str] = []
    for prefix in prefixes:
        matched = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if name == prefix or name.startswith(prefix + ".")
        ]
        if not matched:
            raise ValueError(f"no parameters under module {prefix!r}")
        for name, parameter in matched:
            parameter.requires_grad_(False)
            frozen.append(name)
    return frozen


def model_run_name(label: str, lr: float, total_iters: int) -> str:
    """Run directory of a fine-tuned model inside its rollout directory."""
    return f"{label}_model_lr{lr:g}_it{int(total_iters)}"


def rollout_overrides(
    dataset_dir: Path,
    label: str,
    *,
    lr: float,
    total_iters: int,
    checkpoint_steps: Sequence[int] = (),
) -> list[str]:
    """Config overrides that point a baseline config at one rollout dataset.

    Trains on ``<label>_train.h5`` and reports the fixed validation denoising
    loss on ``<label>_pilot.h5`` (the pilot part of the holdout, §4.4); the gate
    part of the holdout is never read during training.
    """
    dataset_dir = Path(dataset_dir).resolve()

    def episodes(name: str) -> int:
        meta = json.loads((dataset_dir / f"{name}.json").read_text(encoding="utf-8"))
        return len(meta["episodes"])

    steps = sorted(int(step) for step in checkpoint_steps if 0 < int(step) < total_iters)
    return [
        f"data.root={json.dumps(str(dataset_dir))}",
        f"data.train_path={json.dumps(f'{label}_train.h5')}",
        f"data.val_path={json.dumps(f'{label}_pilot.h5')}",
        f"data.num_demos={episodes(f'{label}_train')}",
        f"data.val_num_demos={episodes(f'{label}_pilot')}",
        f"train.lr={lr!r}",
        f"train.total_iters={int(total_iters)}",
        f"train.checkpoint_steps={steps}",
        f"train.validation_steps={steps}",
    ]
