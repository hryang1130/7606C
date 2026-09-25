#!/usr/bin/env python3
"""Check episode-local history/action indexing using saved ManiSkill observations."""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_H5 = Path("data/pickcube/state/pickcube_batch10.state.pd_joint_pos.physx_cpu.h5")


def valid_times(action_count: int, horizon: int) -> range:
    # At time t, state[t-1:t+1] is history and actions[t:t+H] is the target.
    return range(1, max(1, action_count - horizon + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", nargs="?", type=Path, default=DEFAULT_H5)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")

    with h5py.File(args.h5, "r") as file:
        names = sorted(
            (name for name in file if name.startswith("traj_")),
            key=lambda name: int(name.removeprefix("traj_")),
        )
        lengths = [len(file[name]["actions"]) for name in names]
        episode_ids = np.concatenate(
            [np.full(length, episode_id, dtype=np.int32) for episode_id, length in enumerate(lengths)]
        )
        offset = 0
        total_windows = 0
        for episode_id, (name, length) in enumerate(zip(names, lengths)):
            group = file[name]
            observations = group["obs"]
            actions = group["actions"]
            assert isinstance(observations, h5py.Dataset), f"{name}: expected flat state observations"
            assert len(observations) == length + 1, f"{name}: expected T+1 observations"
            times = valid_times(length, args.horizon)
            if times:
                representative = [times[0], times[len(times) // 2], times[-1]]
                for label, t in zip(("first", "middle", "final"), representative):
                    history = observations[t - 1 : t + 1]
                    future = actions[t : t + args.horizon]
                    assert history.shape == (2, observations.shape[1])
                    assert future.shape == (args.horizon, actions.shape[1])
                    assert np.array_equal(history[0], observations[t - 1])
                    assert np.array_equal(history[1], observations[t])
                    assert np.array_equal(future[0], actions[t])
                    assert np.array_equal(future[-1], actions[t + args.horizon - 1])
                    assert t + args.horizon < len(observations)
                    print(
                        f"{name} {label}: t={t}, obs=[{t-1},{t}], "
                        f"actions=[{t},{t+args.horizon}), next obs={t+args.horizon}, "
                        f"shapes={history.shape}/{future.shape}"
                    )
            for t in times:
                flat_start = offset + t
                assert np.all(episode_ids[flat_start : flat_start + args.horizon] == episode_id)
                assert t + args.horizon <= length
                total_windows += 1
            offset += length

        # A complete but shorter-than-horizon slice of real arrays must yield no window.
        short_action_count = min(args.horizon - 1, lengths[0])
        short_actions = file[names[0]]["actions"][:short_action_count]
        short_observations = file[names[0]]["obs"][: short_action_count + 1]
        assert len(short_observations) == len(short_actions) + 1
        assert len(valid_times(len(short_actions), args.horizon)) == 0
        print(f"Short trajectory ({len(short_actions)} actions): 0 valid windows — PASS")
        print(f"Episode-local indexing: {total_windows} windows across {len(names)} episodes — PASS")
        print("Cross-episode action windows: 0 — PASS")
        print("Actual observation-history windows: PASS")


if __name__ == "__main__":
    main()
