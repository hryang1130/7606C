"""Training data for takeover-correction fine-tuning (configs/failure_aware/takeover_correction.toml).

Two pieces, both free of ManiSkill:

* :func:`correction_segment` cuts the expert branch of one successful takeover
  into a training episode: the real frames ``tau-1`` and ``tau`` as observation
  history, then the expert's actions up to the end of the action chunk that
  first succeeds. ``train_start = 1`` keeps windows from starting before the
  takeover, so the baseline's pre-takeover states are never decision samples.
  The one pre-takeover action that remains is the action actually executed at
  ``tau-1``; it only fills the past slot of the first window, which inference
  never executes.
* :func:`write_mixed_dataset` copies the base expert demonstrations and either
  the corrections (arm C) or further expert demonstrations (arm D) into one
  file in the ``maniskill-demogen`` schema, so the shared trainer reads it
  unchanged.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from .failure_rollout import RolloutEpisode


def correction_end(tau: int, first_success: int, act_horizon: int, max_steps: int) -> int:
    """Executed-action count at the end of the chunk holding the first success."""
    if not tau < first_success <= max_steps:
        raise ValueError(f"first success {first_success} must follow tau {tau}")
    return min(max_steps, tau + math.ceil((first_success - tau) / act_horizon) * act_horizon)


def correction_segment(
    seed: int,
    rgb: np.ndarray,
    proprio: np.ndarray,
    actions: np.ndarray,
    success: np.ndarray,
    *,
    tau: int,
    first_success: int,
    act_horizon: int,
    max_steps: int,
) -> RolloutEpisode:
    """Slice one full expert-branch episode (``T+1`` frames, ``T`` actions)."""
    if tau < 1:
        raise ValueError("a correction needs the real frame before the takeover")
    end = correction_end(tau, first_success, act_horizon, max_steps)
    return RolloutEpisode(
        seed=seed,
        rgb=np.asarray(rgb[tau - 1 : end + 1], dtype=np.uint8),
        proprio=np.asarray(proprio[tau - 1 : end + 1], dtype=np.float32),
        actions=np.asarray(actions[tau - 1 : end], dtype=np.float32),
        success=np.asarray(success[tau - 1 : end], dtype=bool),
        reward=np.zeros(end - tau + 1, dtype=np.float32),
    )


def ordered_demo_entries(sidecar: Path) -> list[dict[str, Any]]:
    """Demonstration entries in the trainer's canonical seed order."""
    meta = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    return sorted(meta["episodes"], key=lambda entry: int(entry["episode_seed"]))


def matched_extra_demos(entries: Sequence[dict[str, Any]], lengths: dict[int, int], target: int) -> list:
    """Shortest seed-ordered prefix of ``entries`` with at least ``target`` windows."""
    chosen, total = [], 0
    for entry in entries:
        if total >= target:
            break
        chosen.append(entry)
        total += lengths[int(entry["episode_id"])]
    if total < target:
        raise ValueError(f"only {total} demonstration windows available, need {target}")
    return chosen


def write_mixed_dataset(
    path: Path,
    *,
    demo_file: Path,
    demo_entries: Sequence[dict[str, Any]],
    corrections: Sequence[tuple[RolloutEpisode, dict[str, Any]]] = (),
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write demos (``train_start`` 0) then corrections (``train_start`` 1).

    Returns the sidecar. The HDF5 and sidecar are written under ``.partial``
    names and renamed at the end, sidecar last.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    demo_file = Path(demo_file)
    demo_meta = json.loads(demo_file.with_suffix(".json").read_text(encoding="utf-8"))
    partial = path.with_name(path.name + ".partial")
    entries: list[dict[str, Any]] = []
    with h5py.File(demo_file, "r") as source, h5py.File(partial, "w") as target:
        for entry in demo_entries:
            episode_id = len(entries)
            group = target.create_group(f"traj_{episode_id}")
            origin = source[f"traj_{int(entry['episode_id'])}"]
            for key in ("obs_rgb/rgb", "obs_rgb/state", "actions"):
                source.copy(origin[key], group, name=key)
            entries.append(
                {"episode_id": episode_id, "episode_seed": int(entry["episode_seed"]),
                 "source": "demo", "source_episode_id": int(entry["episode_id"]), "train_start": 0}
            )
        for episode, extra in corrections:
            episode_id = len(entries)
            group = target.create_group(f"traj_{episode_id}")
            group.create_dataset("obs_rgb/rgb", data=episode.rgb, compression="gzip", compression_opts=5)
            group.create_dataset("obs_rgb/state", data=episode.proprio)
            group.create_dataset("actions", data=episode.actions)
            entries.append(
                {"episode_id": episode_id, "episode_seed": episode.seed, "source": "correction",
                 "train_start": 1, **extra}
            )
    seeds = [entry["episode_seed"] for entry in entries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("mixed dataset seeds must be unique")
    os.replace(partial, path)
    export = demo_file.with_name(demo_file.stem + ".export_info.json")
    if export.is_file():
        path.with_name(path.stem + ".export_info.json").write_text(export.read_text(encoding="utf-8"))
    sidecar = {"env_info": demo_meta["env_info"], "episodes": entries, "mixed_provenance": provenance}
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path.with_suffix(".json"))
    return sidecar
