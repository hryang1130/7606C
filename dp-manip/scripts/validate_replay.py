#!/usr/bin/env python3
"""Validate a state-observation ManiSkill replay against its source trajectories."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def datasets(group: h5py.Group) -> dict[str, h5py.Dataset]:
    result = {}
    group.visititems(
        lambda name, item: result.__setitem__(name, item)
        if isinstance(item, h5py.Dataset)
        else None
    )
    return result


def metadata(path: Path) -> dict:
    with path.with_suffix(".json").open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()

    original_meta = metadata(args.source)
    replay_meta = metadata(args.replay)
    original_episodes = {item["episode_id"]: item for item in original_meta["episodes"]}
    replay_episodes = {item["episode_id"]: item for item in replay_meta["episodes"]}
    assert set(original_episodes) == set(replay_episodes), "A source episode is missing or extra"
    assert replay_meta["env_info"]["env_kwargs"]["obs_mode"] == "state"
    assert replay_meta["env_info"]["env_kwargs"]["control_mode"] == "pd_joint_pos"

    with h5py.File(args.source, "r") as source, h5py.File(args.replay, "r") as replay:
        expected_names = {f"traj_{episode_id}" for episode_id in original_episodes}
        assert set(source.keys()) == expected_names
        assert set(replay.keys()) == expected_names, "H5 trajectory IDs differ"
        observation_schema = None
        total_steps = 0
        for episode_id in sorted(original_episodes):
            name = f"traj_{episode_id}"
            old_episode = original_episodes[episode_id]
            new_episode = replay_episodes[episode_id]
            old_group = source[name]
            new_group = replay[name]
            actions = new_group["actions"][()]
            length = len(actions)
            assert old_episode["episode_seed"] == new_episode["episode_seed"]
            assert old_episode["control_mode"] == new_episode["control_mode"] == "pd_joint_pos"
            assert old_episode["elapsed_steps"] == new_episode["elapsed_steps"] == length
            assert new_episode["success"] is True, f"{name}: replay did not succeed"
            assert bool(new_group["success"][-1]), f"{name}: final H5 success is false"
            assert actions.shape[1] == 8 and actions.dtype == np.float32
            assert np.array_equal(old_group["actions"][()], actions), f"{name}: actions changed"
            assert np.isfinite(actions).all(), f"{name}: non-finite actions"

            obs = new_group["obs"]
            if isinstance(obs, h5py.Dataset):
                fields = {"obs": obs}
            else:
                fields = datasets(obs)
            assert fields, f"{name}: no observations were saved"
            schema = {}
            for field_name, field in fields.items():
                assert field.shape[0] == length + 1, f"{name}/{field_name}: not T+1"
                if np.issubdtype(field.dtype, np.number):
                    assert np.isfinite(field[()]).all(), f"{name}/{field_name}: non-finite obs"
                schema[field_name] = (field.shape[1:], str(field.dtype))
            if observation_schema is None:
                observation_schema = schema
            else:
                assert schema == observation_schema, f"{name}: observation schema differs"

            for path, field in datasets(new_group).items():
                if np.issubdtype(field.dtype, np.number):
                    assert np.isfinite(field[()]).all(), f"{name}/{path}: non-finite values"
            total_steps += length
            print(f"{name}: seed={new_episode['episode_seed']}, T={length}, success=PASS, actions unchanged=PASS")

    print(f"Episodes retained: {len(replay_episodes)}/{len(original_episodes)}")
    print(f"Successful episodes: {sum(item['success'] is True for item in replay_episodes.values())}/{len(replay_episodes)}")
    print(f"Total action steps: {total_steps}")
    print(f"Replay backend: {replay_meta['env_info']['env_kwargs'].get('sim_backend')}")
    print("Observation schema (each field has leading T+1 axis):")
    for name, (suffix, dtype) in observation_schema.items():
        print(f"  {name}: (T+1, {', '.join(map(str, suffix))}), {dtype}")
    print("Numeric integrity, alignment, and source-action identity: PASS")


if __name__ == "__main__":
    main()
