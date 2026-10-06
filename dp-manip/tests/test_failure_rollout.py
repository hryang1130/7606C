"""Phase 1 of the failure-aware plan: rollout collection and datasets.

ManiSkill is cluster-only, so a fake vector environment stands in for it. The
fake reproduces the parts ``evaluate`` relies on (flattened RGB/state
observations, per-step ``info["success"]``, simultaneous truncation) and encodes
the seed and timestep in the observations, so alignment is checkable.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip.data import RGBWindowDataset, read_dataset_info
    from dp_manip.evaluate import evaluate
    from dp_manip.failure_rollout import (
        RAW_HOLDOUT,
        RAW_TRAIN,
        ClassQuota,
        RolloutWriter,
        build_datasets,
        collect,
        failure_storage_length,
        success_storage_length,
    )
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

from dp_manip import config as config_lib
from dp_manip.failure_protocol import DEFAULT_PROTOCOL, load_protocol

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
# The study runs on PlaceSphere; LiftPegUpright is the planned replication and
# must keep working through the same code (plan §12.1).
STUDY_TASKS = ("placesphere", "liftpegupright")

IMAGE = (4, 4, 6)
PROPRIO_DIM = 3
ACTION_DIM = 2
ACT_HORIZON = 3
MAX_STEPS = 10  # not a multiple of ACT_HORIZON: the last chunk is cut short


def success_step(seed: int) -> int | None:
    """Even seeds first succeed after ``2 + seed % 5`` actions; odd seeds never do."""
    return 2 + seed % 5 if seed % 2 == 0 else None


class FakeVectorEnv:
    def __init__(self, num_envs: int, max_steps: int = MAX_STEPS):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.actions: list[np.ndarray] = []

    def _observation(self) -> dict:
        rgb = np.zeros((self.num_envs, *IMAGE), dtype=np.uint8)
        rgb[..., 0] = np.asarray(self.seeds)[:, None, None] % 256
        rgb[..., 1] = self.t
        state = np.zeros((self.num_envs, PROPRIO_DIM), dtype=np.float32)
        state[:, 0] = self.t
        state[:, 1] = self.seeds
        return {"rgb": rgb, "state": state}

    def reset(self, seed):
        self.seeds = list(seed)
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (self.num_envs, ACTION_DIM)
        self.actions.append(action.copy())
        self.t += 1
        success = np.array(
            [success_step(seed) is not None and self.t >= success_step(seed) for seed in self.seeds]
        )
        # Episodes lose success again one step later for seeds divisible by 4,
        # so success_once and success_at_end differ.
        lost = np.array([seed % 4 == 0 and self.t > success_step(seed) for seed in self.seeds])
        success &= ~lost
        reward = action.sum(axis=1)
        truncated = np.full(self.num_envs, self.t >= self.max_steps)
        return self._observation(), reward, np.zeros(self.num_envs, bool), truncated, {"success": success}


class FakePolicy:
    obs_horizon = 2
    training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def get_action(self, rgb, proprio, *, generator):
        return torch.randn((rgb.shape[0], ACT_HORIZON, ACTION_DIM), generator=generator)


def small_protocol(**collection):
    protocol = load_protocol(DEFAULT_PROTOCOL)
    fields = dict(
        dataset_size=3,
        rollout_cap=16,
        holdout_episodes=8,
        pilot_holdout_episodes=4,
        failure_truncation_factor=1.5,
        low_success_min=2,
    )
    fields.update(collection)
    return dataclasses.replace(
        protocol, collection=dataclasses.replace(protocol.collection, **fields)
    )


def env_info() -> dict:
    return {"env_id": "PegInsertionSide-v1", "env_kwargs": {"control_mode": "pd_joint_pos"}}


def raw_provenance(protocol, split: str, checkpoint_sha: str = "abc") -> dict:
    return {
        "dataset_type": "rollout_raw",
        "split": split,
        "act_horizon": ACT_HORIZON,
        "max_episode_steps": MAX_STEPS,
        "source_checkpoint": {"path": "final.pt", "sha256": checkpoint_sha},
        "protocol": {"path": str(protocol.path), "sha256": protocol.sha256},
    }


class ProtocolTest(unittest.TestCase):
    def test_shipped_protocol_fits_every_study_task(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        for task in STUDY_TASKS:
            with self.subTest(task=task):
                cfg = config_lib.load(TASKS / f"{task}.toml")
                protocol.check_against(cfg)
                self.assertEqual(len(protocol.train_seeds()), protocol.collection.rollout_cap)
                self.assertTrue(
                    all(20_000 <= seed < 21_000 for seed in protocol.train_seeds())
                )
                self.assertEqual(protocol.pilot_holdout_end(), 21_050)

    def test_overlapping_ranges_are_rejected(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        seeds = dataclasses.replace(protocol.seeds, collection_holdout=(20_500, 21_500))
        with self.assertRaisesRegex(ValueError, "overlap"):
            dataclasses.replace(protocol, seeds=seeds).validate()

    def test_expert_seed_range_is_rejected(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        seeds = dataclasses.replace(protocol.seeds, collection_train=(3_000, 3_600))
        with self.assertRaisesRegex(ValueError, "5000"):
            dataclasses.replace(protocol, seeds=seeds).validate()

    def test_evaluation_seed_clash_is_rejected(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        cfg = config_lib.load(TASKS / "peginsertionside.toml", ["eval.test_seed_start=20100"])
        with self.assertRaisesRegex(ValueError, "evaluation seeds"):
            protocol.check_against(cfg)

    def test_counts_must_divide_num_envs(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        cfg = config_lib.load(TASKS / "peginsertionside.toml", ["eval.num_envs=7"])
        with self.assertRaisesRegex(ValueError, "divisible"):
            protocol.check_against(cfg)


class TruncationRuleTest(unittest.TestCase):
    @unittest.skipUnless(HAVE_TORCH, "numpy/h5py/torch are not installed")
    def test_success_is_kept_to_the_end_of_its_action_chunk(self) -> None:
        self.assertEqual(success_storage_length(1, 300, 8), 8)
        self.assertEqual(success_storage_length(8, 300, 8), 8)
        self.assertEqual(success_storage_length(9, 300, 8), 16)
        self.assertEqual(success_storage_length(299, 300, 8), 300)

    @unittest.skipUnless(HAVE_TORCH, "numpy/h5py/torch are not installed")
    def test_failure_length_is_one_and_a_half_median_rounded_up_to_a_chunk(self) -> None:
        self.assertEqual(failure_storage_length([96, 96, 104], 1.5, 8, 300), 144)
        self.assertEqual(failure_storage_length([100], 1.5, 8, 300), 152)
        self.assertEqual(failure_storage_length([96, 104], 1.5, 8, 300), 152)
        self.assertEqual(failure_storage_length([240], 1.5, 8, 300), 300)
        self.assertEqual(failure_storage_length([], 1.5, 8, 300), 300)


@unittest.skipUnless(HAVE_TORCH, "numpy/h5py/torch are not installed")
class CollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = small_protocol()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def collect_split(self, split: str, seeds, quota=None, *, sha: str = "abc"):
        path = self.root / (RAW_TRAIN if split == "train" else RAW_HOLDOUT)
        writer = RolloutWriter(path, env_info=env_info(), export_info={"cameras": ["base", "hand"]})
        envs = FakeVectorEnv(num_envs=2)
        result = collect(
            FakePolicy(), envs, seeds, torch.device("cpu"), inference_seed=0, writer=writer, quota=quota
        )
        writer.close(raw_provenance(self.protocol, split, sha))
        return result, envs, path

    def test_observer_does_not_change_evaluation(self) -> None:
        seeds = list(range(20_000, 20_008))
        plain = evaluate(FakePolicy(), FakeVectorEnv(2), seeds, torch.device("cpu"), inference_seed=0)
        observed, _, _ = self.collect_split("holdout", seeds)
        self.assertEqual(plain["episodes"], observed["episodes"])

    def test_recorded_episodes_are_aligned_and_complete(self) -> None:
        seeds = list(range(20_000, 20_004))
        _, envs, path = self.collect_split("holdout", seeds)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.assertFalse(path.with_name(path.name + ".partial").exists())
        with h5py.File(path, "r") as file:
            for index, entry in enumerate(meta["episodes"]):
                group = file[f"traj_{entry['episode_id']}"]
                state = np.asarray(group["obs_rgb/state"])
                actions = np.asarray(group["actions"])
                self.assertEqual(len(state), MAX_STEPS + 1)
                self.assertEqual(len(actions), MAX_STEPS)
                np.testing.assert_array_equal(state[:, 0], np.arange(MAX_STEPS + 1))
                np.testing.assert_array_equal(state[:, 1], entry["episode_seed"])
                wave, env_index = divmod(seeds.index(entry["episode_seed"]), 2)
                sent = np.stack(envs.actions)[wave * MAX_STEPS : (wave + 1) * MAX_STEPS, env_index]
                np.testing.assert_array_equal(actions, sent)
                expected = success_step(entry["episode_seed"])
                self.assertEqual(entry["first_success_step"], expected)
                self.assertEqual(entry["success_once"], expected is not None)
        info = read_dataset_info(path)
        self.assertEqual(info.seeds, seeds)

    def test_quota_stops_after_the_wave_that_reaches_it(self) -> None:
        quota = ClassQuota(3)
        result, _, _ = self.collect_split("train", self.protocol.train_seeds(), quota)
        # Seeds alternate success/failure, so 3 of each need exactly 6 rollouts.
        self.assertEqual(len(result["episodes"]), 6)
        self.assertTrue(quota.reached())

    def test_build_writes_loadable_truncated_datasets(self) -> None:
        self.collect_split("train", self.protocol.train_seeds(), ClassQuota(3))
        self.collect_split("holdout", self.protocol.holdout_seeds())
        summary = build_datasets(self.root, self.protocol)

        self.assertEqual(summary["dataset_size"], 3)
        # Train successes: seeds 20000, 20002, 20004 first succeed after 2, 4, 6
        # actions -> stored to their chunk ends 3, 6, 6; L_fail = ceil3(1.5 * 6) = 9.
        self.assertEqual(summary["l_fail"], 9)
        datasets = self.root / "datasets"
        success = read_dataset_info(datasets / "success_train.h5")
        failure = read_dataset_info(datasets / "failure_train.h5")
        self.assertEqual(success.seeds, [20_000, 20_002, 20_004])
        self.assertEqual(failure.seeds, [20_001, 20_003, 20_005])
        self.assertEqual([episode.length for episode in success.episodes], [3, 6, 6])
        self.assertEqual([episode.length for episode in failure.episodes], [9, 9, 9])
        self.assertEqual(success.cameras, ("base", "hand"))

        pilot = read_dataset_info(datasets / "success_pilot.h5")
        gate = read_dataset_info(datasets / "failure_gate.h5")
        self.assertTrue(all(seed < 21_004 for seed in pilot.seeds))
        self.assertTrue(all(seed >= 21_004 for seed in gate.seeds))

        window = RGBWindowDataset(failure, obs_horizon=2, pred_horizon=4)[0]
        self.assertEqual(window["actions"].shape, (4, ACTION_DIM))
        meta = json.loads((datasets / "failure_train.json").read_text())
        provenance = meta["rollout_provenance"]
        self.assertEqual(provenance["dataset_type"], "rollout_failure")
        self.assertEqual(provenance["source_checkpoint"]["sha256"], "abc")
        self.assertEqual(provenance["l_fail"], 9)
        self.assertEqual([entry["stored_length"] for entry in meta["episodes"]], [9, 9, 9])
        self.assertAlmostEqual(summary["datasets"]["failure_train"]["fraction_truncated"], 0.1)
        self.assertEqual([entry["rollout_length"] for entry in meta["episodes"]], [MAX_STEPS] * 3)

    def test_cap_lowers_k_to_the_smaller_class(self) -> None:
        protocol = small_protocol(dataset_size=5, rollout_cap=6)
        self.protocol = protocol
        self.collect_split("train", protocol.train_seeds(), ClassQuota(5))
        self.collect_split("holdout", protocol.holdout_seeds())
        summary = build_datasets(self.root, protocol)
        self.assertEqual(summary["train_rollouts"], 6)
        self.assertEqual(summary["dataset_size"], 3)

    def test_build_rejects_mixed_checkpoints(self) -> None:
        self.collect_split("train", self.protocol.train_seeds(), ClassQuota(3))
        self.collect_split("holdout", self.protocol.holdout_seeds(), sha="other")
        with self.assertRaisesRegex(ValueError, "different checkpoints"):
            build_datasets(self.root, self.protocol)

    def test_existing_raw_file_is_not_overwritten(self) -> None:
        _, _, path = self.collect_split("holdout", self.protocol.holdout_seeds())
        with self.assertRaises(FileExistsError):
            RolloutWriter(path, env_info=env_info(), export_info={})


if __name__ == "__main__":
    unittest.main()
