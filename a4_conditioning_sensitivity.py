#!/usr/bin/env python3
"""A4: how much does each backbone actually use the observation?

For the same windows and the same initial noise, sample an action chunk with
(a) the true observation features, (b) zeros, (c) another window's features
(a permutation). If a backbone barely changes its actions when the observation
is destroyed, its conditioning path is not carrying the signal.

Reference scale: sampling the *same* observation with a *different* noise seed,
i.e. the policy's own stochasticity. A conditioning sensitivity well below that
means the observation is nearly irrelevant to the sampled action.

Offline, CPU only, no simulator and no Slurm job.

Run: cd ~/7606C/dp-manip && PYTHONPATH=. .venv/bin/python ~/7606C/a4_conditioning_sensitivity.py
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from dp_manip.data import read_dataset_info, RGBWindowDataset
from dp_manip.policy import DiffusionPolicy

DATA = "/userhome/cs5/u3684238/7606C/maniskill-demogen/data/dataset"
VAL = "val/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
RUNS = {
    "unet": "pickcube_rgb_unet_n100_s1",
    "transformer": "pickcube_rgb_transformer_n100_s1",
    "mlp": "pickcube_rgb_mlp_n100_s1",
}
WINDOWS = 12


def select_windows(dataset, info) -> list[int]:
    """Windows whose 16-step action block straddles the gripper switch."""
    switches = {}
    with h5py.File(info.path, "r") as handle:
        for index, episode in enumerate(info.episodes):
            actions = np.asarray(handle[episode.group]["actions"])
            switches[index] = int(np.argmax(np.diff(actions[:, 3]) < -0.5)) + 1
    chosen = []
    for index, (episode_index, timestep) in enumerate(dataset.index):
        switch = switches[episode_index]
        if switch - 8 <= timestep - 1 <= switch - 1:
            chosen.append(index)
        if len(chosen) >= WINDOWS:
            break
    return chosen


def sample(policy, obs, generator) -> np.ndarray:
    with torch.no_grad():
        actions = policy.sample_actions(obs, generator=generator)
    return actions.numpy()


def main() -> int:
    info = read_dataset_info(f"{DATA}/{VAL}", 50)
    dataset = RGBWindowDataset(info, 2, 16, preload=False)
    indices = select_windows(dataset, info)
    rgb = torch.stack([torch.as_tensor(dataset[i]["rgb"]) for i in indices]).float()
    proprio = torch.stack([torch.as_tensor(dataset[i]["proprio"]) for i in indices]).float()
    count = len(indices)
    print(f"windows: {count}", flush=True)

    print(f"\n{'model':<12}{'|dpos| zero':>13}{'|dpos| shuffle':>16}{'|dpos| new-noise':>16}"
          f"{'zero/noise':>12}{'shuffle/noise':>15}{'|dgrip| shuffle':>17}")
    for name, run in RUNS.items():
        checkpoint = torch.load(
            f"/userhome/cs5/u3684238/dp-runs-pickcube/{run}/checkpoints/final.pt",
            map_location="cpu",
            weights_only=False,
        )
        policy = DiffusionPolicy.from_checkpoint(checkpoint, "cpu").eval()
        with torch.no_grad():
            obs = policy.observation_features(rgb, proprio)
            zeros = torch.zeros_like(obs)
            permuted = obs[torch.roll(torch.arange(count), count // 2)]
        reference = torch.Generator().manual_seed(11)
        other_noise = torch.Generator().manual_seed(29)
        print(f"  {name}: obs ready, sampling...", flush=True)
        base = sample(policy, obs, reference)
        zeroed = sample(policy, zeros, torch.Generator().manual_seed(11))
        shuffled = sample(policy, permuted, torch.Generator().manual_seed(11))
        reseeded = sample(policy, obs, other_noise)

        def delta(other: np.ndarray) -> tuple[float, float]:
            position = np.abs(other[..., :3] - base[..., :3]).mean()
            gripper = np.abs(other[..., 3] - base[..., 3]).mean()
            return position, gripper

        zero_pos, _ = delta(zeroed)
        shuffle_pos, shuffle_grip = delta(shuffled)
        noise_pos, _ = delta(reseeded)
        print(f"{name:<12}{zero_pos:13.4f}{shuffle_pos:16.4f}{noise_pos:16.4f}"
              f"{zero_pos / noise_pos:12.2f}{shuffle_pos / noise_pos:15.2f}{shuffle_grip:17.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
