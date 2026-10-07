#!/usr/bin/env python3
"""A1: how sharp are the demonstration actions in the PickCube training set?

Offline, CPU only, no simulator and no Slurm job. Prints, per action
dimension, the range and the step-to-step change distribution, then the
gripper's value set and the number of open->close switches per episode.

Run:  cd ~/7606C/dp-manip && PYTHONPATH=. .venv/bin/python ~/7606C/a1_actions.py
"""

from __future__ import annotations

import h5py
import numpy as np

DATASET = (
    "/userhome/cs5/u3684238/7606C/maniskill-demogen/data/dataset/train/"
    "PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
)
NAMES = ["dx", "dy", "dz", "gripper"]


def main() -> int:
    with h5py.File(DATASET, "r") as file:
        keys = list(file.keys())
        actions = [np.asarray(file[key]["actions"], dtype=np.float64) for key in keys]
    total = sum(a.shape[0] for a in actions)
    print(f"episodes: {len(actions)}   steps: {total}   mean length: {total / len(actions):.1f}")

    stacked = np.concatenate(actions, axis=0)
    print(f"\n{'dim':<10}{'min':>9}{'max':>9}{'mean':>9}{'std':>9}")
    for index, name in enumerate(NAMES):
        column = stacked[:, index]
        print(f"{name:<10}{column.min():9.3f}{column.max():9.3f}{column.mean():9.3f}{column.std():9.3f}")

    deltas = np.concatenate([np.abs(np.diff(a, axis=0)) for a in actions], axis=0)
    print(f"\n|a(t+1) - a(t)|:")
    print(f"{'dim':<10}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}{'|d|>0.2':>9}")
    for index, name in enumerate(NAMES):
        column = deltas[:, index]
        print(
            f"{name:<10}{column.mean():9.4f}{np.median(column):9.4f}"
            f"{np.percentile(column, 95):9.4f}{np.percentile(column, 99):9.4f}"
            f"{column.max():9.4f}{(column > 0.2).mean() * 100:8.1f}%"
        )

    gripper = stacked[:, 3]
    values = np.unique(np.round(gripper, 4))
    switches = [int((np.abs(np.diff(a[:, 3])) > 0.5).sum()) for a in actions]
    closing = sum(int((np.diff(a[:, 3]) < -0.5).sum()) for a in actions)
    print(f"\ngripper values: {len(values)} -> {values[:8].tolist()}")
    print(
        f"switches per episode: mean={np.mean(switches):.2f} median={np.median(switches):.0f} "
        f"max={max(switches)}  (all closing: {closing}/{sum(switches)})"
    )
    span = max(np.percentile(deltas[:, :3], 99), 1e-9)
    print(f"\ngripper max single-step jump = {deltas[:, 3].max():.4f} = {deltas[:, 3].max() / span:.1f}x "
          f"the position p99 ({span:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
