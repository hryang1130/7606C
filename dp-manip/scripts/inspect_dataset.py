#!/usr/bin/env python3
"""Inspect a ManiSkill trajectory H5 file and its sidecar JSON metadata."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_H5 = Path("data/pickcube/state/pickcube_batch10.state.pd_joint_pos.physx_cpu.h5")


def print_dataset(name: str, dataset: h5py.Dataset) -> tuple[int, int]:
    print(f"    {name}: shape={dataset.shape}, dtype={dataset.dtype}")
    if dataset.size == 0:
        return 0, 0
    values = dataset[()]
    if np.issubdtype(dataset.dtype, np.number):
        nan_count = int(np.count_nonzero(np.isnan(values)))
        inf_count = int(np.count_nonzero(np.isinf(values)))
        print(f"      NaN={nan_count}, Inf={inf_count}")
        finite = values[np.isfinite(values)]
        if finite.size:
            print(
                "      min={:.6g}, max={:.6g}, mean={:.6g}, std={:.6g}".format(
                    float(finite.min()),
                    float(finite.max()),
                    float(finite.mean()),
                    float(finite.std()),
                )
            )
        else:
            print("      no finite values")
        return nan_count, inf_count
    elif np.issubdtype(dataset.dtype, np.bool_):
        print(f"      true={int(np.count_nonzero(values))}/{values.size}")
    return 0, 0


def datasets_under(group: h5py.Group):
    found = []
    group.visititems(
        lambda name, obj: found.append((name, obj))
        if isinstance(obj, h5py.Dataset)
        else None
    )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", nargs="?", type=Path, default=DEFAULT_H5)
    parser.add_argument("--json", type=Path, help="Sidecar JSON path (defaults to H5 stem + .json)")
    args = parser.parse_args()
    metadata_path = args.json or args.h5.with_suffix(".json")

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    episodes = metadata.get("episodes", [])
    env_info = metadata.get("env_info", {})
    print(f"H5: {args.h5}")
    print(f"JSON: {metadata_path}")
    print(f"Task: {env_info.get('env_id')}")
    print(f"Environment kwargs: {json.dumps(env_info.get('env_kwargs', {}), sort_keys=True)}")
    print(f"Max episode steps: {env_info.get('max_episode_steps')}")
    print(f"Source: {metadata.get('source_type')} — {metadata.get('source_desc')}")
    print(f"JSON episodes: {len(episodes)}")

    with h5py.File(args.h5, "r") as file:
        print(f"H5 top-level keys: {list(file.keys())}")
        trajectory_names = sorted(
            (name for name in file if name.startswith("traj_")),
            key=lambda name: int(name.removeprefix("traj_")),
        )
        print(f"H5 episodes: {len(trajectory_names)}")
        total_actions = 0
        observation_count = 0
        nan_total = 0
        inf_total = 0
        all_actions = []
        h5_episode_ids = {int(name.removeprefix("traj_")) for name in trajectory_names}
        json_episode_ids = {item["episode_id"] for item in episodes}
        print(f"Episode boundaries: H5 groups {trajectory_names}; JSON IDs {sorted(json_episode_ids)}")
        if h5_episode_ids != json_episode_ids:
            print("WARNING: H5 and JSON episode IDs differ")
        for name in trajectory_names:
            group = file[name]
            episode_id = int(name.removeprefix("traj_"))
            episode = next((item for item in episodes if item.get("episode_id") == episode_id), {})
            action_length = len(group["actions"])
            total_actions += action_length
            all_actions.append(group["actions"][()])
            print(
                f"  {name}: actions={action_length}, json_elapsed_steps={episode.get('elapsed_steps')}, "
                f"seed={episode.get('episode_seed')}, success={episode.get('success')}, "
                f"control_mode={episode.get('control_mode')}"
            )
            if episode and action_length != episode.get("elapsed_steps"):
                print("    WARNING: action length differs from JSON elapsed_steps")
            print(f"    group keys: {list(group.keys())}")
            for section in ("obs", "actions", "env_states", "success", "terminated", "truncated"):
                if section not in group:
                    print(f"    {section}: MISSING")
                    continue
                obj = group[section]
                if isinstance(obj, h5py.Dataset):
                    if section == "obs":
                        observation_count += 1
                        print("    observation structure: flat dataset; fields: ['obs']")
                    nan_count, inf_count = print_dataset(section, obj)
                    nan_total += nan_count
                    inf_total += inf_count
                    if section == "obs":
                        relation = (
                            "T" if len(obj) == action_length else
                            "T+1" if len(obj) == action_length + 1 else
                            "neither T nor T+1"
                        )
                        print(f"      time axis relative to actions: {relation}")
                else:
                    entries = datasets_under(obj)
                    if section == "obs":
                        observation_count += len(entries)
                        print(f"    observation fields: {[child_name for child_name, _ in entries]}")
                    if not entries:
                        print(f"    {section}: empty group")
                    for child_name, dataset in entries:
                        nan_count, inf_count = print_dataset(f"{section}/{child_name}", dataset)
                        nan_total += nan_count
                        inf_total += inf_count
                        if section in ("obs", "env_states"):
                            relation = (
                                "T" if len(dataset) == action_length else
                                "T+1" if len(dataset) == action_length + 1 else
                                "neither T nor T+1"
                            )
                            print(f"      time axis relative to actions: {relation}")
        print(f"Total action steps: {total_actions}")
        if all_actions:
            actions = np.concatenate(all_actions, axis=0)
            print("Action dimensions across all episodes (finite values only):")
            for dimension in range(actions.shape[1]):
                values = actions[:, dimension]
                finite = values[np.isfinite(values)]
                if finite.size:
                    print(
                        f"  {dimension}: min={finite.min():.6g}, max={finite.max():.6g}, "
                        f"mean={finite.mean():.6g}, std={finite.std():.6g}, "
                        f"NaN={np.count_nonzero(np.isnan(values))}, Inf={np.count_nonzero(np.isinf(values))}"
                    )
        print(f"Numeric integrity: {'PASS' if nan_total == inf_total == 0 else 'FAIL'} (NaN={nan_total}, Inf={inf_total})")
        if h5_episode_ids == json_episode_ids:
            print("Episode boundaries recoverable from H5 groups and JSON episode IDs: PASS")
        if observation_count == 0:
            print("WARNING: no observation datasets found; this file cannot directly train an observation-conditioned policy.")


if __name__ == "__main__":
    main()
