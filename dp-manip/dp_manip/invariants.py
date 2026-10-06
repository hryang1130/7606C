"""Gate B helpers: declared differences, config diffs and control hashes.

An experiment is only a controlled comparison when every resolved cell shares
the same config outside the declared experimental variable, the replicate seed,
runtime paths and the backbone architecture definitions. These helpers back
``scripts/check_experiment.py`` (drift reports) and the trainer, which records
the ``control_hash`` of every run (REFACTOR_PLAN.md §19).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import Config, ExperimentSpec, match_experiment_value


# A replicate seed, the dataset root and whether RGB is preloaded (the samples
# are bit-identical either way) are runtime metadata, not controls.
SEED_KEY = "train.seed"
RUNTIME_KEYS = frozenset({"data.root", "data.preload"})

# Architecture definitions of the three arms (docs/final-plan.md §6). They may
# differ only inside the backbone experiment, where they belong to the declared
# variable (the plan's ``policy.backbone.*``), e.g. a supplementary
# capacity-matched arm. In any other experiment they are ordinary controls.
BACKBONE_VARIABLE = "policy.backbone"
STRUCTURAL_KEYS = frozenset(
    {
        "policy.unet_dims",
        "policy.kernel_size",
        "policy.n_groups",
        "policy.diffusion_step_embed_dim",
        "policy.transformer_layers",
        "policy.transformer_heads",
        "policy.transformer_embed_dim",
        "policy.transformer_dropout_emb",
        "policy.transformer_dropout_attn",
        "policy.transformer_causal_attn",
        "policy.transformer_cond_layers",
        "policy.mlp_hidden_dim",
        "policy.mlp_layers",
        "policy.mlp_time_embed_dim",
        "policy.mlp_obs_feat_dim",
    }
)


# The task data paths name their control mode (``{control_mode}`` in the task
# files), so a control-mode experiment changes them together with its declared
# variable. There they are part of the variable; in any other experiment they
# are ordinary controls.
CONTROL_MODE_VARIABLE = "task.control_mode"
CONTROL_MODE_DERIVED_KEYS = frozenset({"data.train_path", "data.val_path"})


def allowed_keys(spec: ExperimentSpec) -> set[str]:
    """Keys that may differ between cells of this experiment without drift."""
    allowed = {spec.variable, SEED_KEY, *RUNTIME_KEYS}
    if spec.variable == BACKBONE_VARIABLE:
        allowed |= STRUCTURAL_KEYS
    if spec.variable == CONTROL_MODE_VARIABLE:
        allowed |= CONTROL_MODE_DERIVED_KEYS
    return allowed


def config_differences(
    reference: Mapping[str, Any], candidate: Mapping[str, Any], prefix: str = ""
) -> dict[str, tuple[Any, Any]]:
    """Return dotted ``key -> (reference, candidate)`` where two configs differ."""
    differences: dict[str, tuple[Any, Any]] = {}
    for key in sorted(set(reference) | set(candidate)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in reference or key not in candidate:
            differences[path] = (reference.get(key), candidate.get(key))
        elif isinstance(reference[key], Mapping) and isinstance(candidate[key], Mapping):
            differences.update(config_differences(reference[key], candidate[key], path))
        elif reference[key] != candidate[key]:
            differences[path] = (reference[key], candidate[key])
    return differences


def without_keys(raw: Mapping[str, Any], keys: set[str]) -> dict[str, Any]:
    """Copy a resolved config with the allowed dotted keys removed."""
    pruned: dict[str, Any] = {}
    for key, value in raw.items():
        if key in keys:
            continue
        nested = {path.split(".", 1)[1] for path in keys if path.startswith(f"{key}.")}
        if nested and isinstance(value, Mapping):
            pruned[key] = without_keys(value, nested)
        else:
            pruned[key] = value
    return pruned


def control_hash(config: Config, keys: set[str]) -> str:
    """Hash every non-experimental value of a resolved config (plan §19)."""
    payload = json.dumps(without_keys(config.to_dict(), keys), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def experiment_context(
    config: Config,
    spec: ExperimentSpec,
    value: Any,
    *,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run-metadata block describing which declared cell a run belongs to."""
    declared = match_experiment_value(spec, value)
    return {
        "name": spec.name,
        "spec": str(spec_path) if spec_path is not None else None,
        "variable": spec.variable,
        "value": declared,
        "seed": config.train.seed,
        "control_hash": control_hash(config, allowed_keys(spec)),
    }
