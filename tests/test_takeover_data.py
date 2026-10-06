import json
import tempfile
import unittest
from pathlib import Path

try:
    import h5py
    import numpy as np

    from dp_manip.data import ObservationWindowDataset, read_dataset_info
    from dp_manip.finetune import FinetuneSpec
    from dp_manip.takeover_data import (
        correction_end,
        correction_segment,
        matched_extra_demos,
        ordered_demo_entries,
        write_mixed_dataset,
    )
except ModuleNotFoundError:  # torch/h5py are only installed in the cluster environment
    HAVE_DEPS = False
else:
    HAVE_DEPS = True

ENV_INFO = {"env_id": "PlaceSphere-v1", "env_kwargs": {"control_mode": "pd_ee_delta_pos"}}


def write_demos(path, seeds, length=6):
    with h5py.File(path, "w") as file:
        for index, seed in enumerate(seeds):
            group = file.create_group(f"traj_{index}")
            group.create_dataset("obs_rgb/rgb", data=np.full((length + 1, 4, 4, 3), seed % 256, np.uint8))
            group.create_dataset("obs_rgb/state", data=np.full((length + 1, 3), seed, np.float32))
            group.create_dataset("actions", data=np.full((length, 4), seed, np.float32))
            group.create_dataset("success", data=np.ones(length, bool))
    episodes = [{"episode_id": i, "episode_seed": s} for i, s in enumerate(seeds)]
    path.with_suffix(".json").write_text(json.dumps({"env_info": ENV_INFO, "episodes": episodes}))


@unittest.skipUnless(HAVE_DEPS, "requires h5py and torch")
class CorrectionSegmentTests(unittest.TestCase):
    def episode(self, steps=40):
        frames = np.arange(steps + 1)
        rgb = np.broadcast_to(frames[:, None, None, None], (steps + 1, 2, 2, 3)).astype(np.uint8)
        proprio = np.repeat(frames[:, None], 2, axis=1).astype(np.float32)
        actions = np.repeat(np.arange(steps)[:, None], 4, axis=1).astype(np.float32)
        return rgb, proprio, actions, np.zeros(steps, bool)

    def test_end_of_successful_chunk(self):
        self.assertEqual(correction_end(16, 17, 8, 200), 24)
        self.assertEqual(correction_end(16, 24, 8, 200), 24)
        self.assertEqual(correction_end(192, 199, 8, 200), 200)
        with self.assertRaises(ValueError):
            correction_end(16, 16, 8, 200)

    def test_segment_keeps_real_history_and_expert_actions(self):
        rgb, proprio, actions, success = self.episode()
        seg = correction_segment(7, rgb, proprio, actions, success, tau=16, first_success=30,
                                 act_horizon=8, max_steps=40)
        # Frames 15..32, actions 15..31: action 15 is the executed pre-takeover action.
        self.assertEqual(seg.proprio[:, 0].tolist(), list(range(15, 33)))
        self.assertEqual(seg.actions[:, 0].tolist(), list(range(15, 32)))
        self.assertEqual(len(seg.rgb), len(seg.actions) + 1)


@unittest.skipUnless(HAVE_DEPS, "requires h5py and torch")
class MixedDatasetTests(unittest.TestCase):
    def test_corrections_train_from_the_takeover_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            demos = Path(tmp) / "demos.h5"
            write_demos(demos, [30, 10, 20])
            entries = ordered_demo_entries(demos.with_suffix(".json"))
            self.assertEqual([e["episode_seed"] for e in entries], [10, 20, 30])
            steps = 40
            frames = np.arange(steps + 1)
            seg = correction_segment(
                41000,
                np.broadcast_to(frames[:, None, None, None], (steps + 1, 4, 4, 3)).astype(np.uint8),
                np.repeat(frames[:, None], 3, axis=1).astype(np.float32),
                np.repeat(np.arange(steps)[:, None], 4, axis=1).astype(np.float32),
                np.zeros(steps, bool), tau=16, first_success=20, act_horizon=8, max_steps=steps,
            )
            mixed = Path(tmp) / "mixed.h5"
            write_mixed_dataset(mixed, demo_file=demos, demo_entries=entries[:2],
                                corrections=[(seg, {"tau": 16})], provenance={"arm": "C"})
            info = read_dataset_info(mixed)
            self.assertEqual(info.seeds, [10, 20, 41000])
            self.assertEqual([e.start for e in info.episodes], [0, 0, 1])
            windows = ObservationWindowDataset(info, obs_horizon=2, pred_horizon=4)
            self.assertEqual(len(windows), 6 + 6 + 8)
            first = windows[12]  # first correction window
            self.assertEqual(first["proprio"][:, 0].tolist(), [15.0, 16.0])  # real history
            self.assertEqual(first["actions"][:, 0].tolist(), [15.0, 16.0, 17.0, 18.0])
            with self.assertRaises(FileExistsError):
                write_mixed_dataset(mixed, demo_file=demos, demo_entries=entries, provenance={})

    def test_extra_demos_match_window_budget(self):
        entries = [{"episode_id": i, "episode_seed": 100 + i} for i in range(5)]
        lengths = {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}
        self.assertEqual(len(matched_extra_demos(entries, lengths, 25)), 3)
        self.assertEqual(len(matched_extra_demos(entries, lengths, 20)), 2)
        with self.assertRaises(ValueError):
            matched_extra_demos(entries, lengths, 51)


@unittest.skipUnless(HAVE_DEPS, "requires h5py and torch")
class FinetuneSpecRecordTests(unittest.TestCase):
    def test_default_record_is_unchanged(self):
        spec = FinetuneSpec("x.pt", ("observation_encoder",), "constant_with_warmup", (0, 1), (1, 2))
        self.assertNotIn("require_rollout_source", spec.to_dict())
        mixed = FinetuneSpec("x.pt", (), "constant_with_warmup", (0, 1), (1, 2), require_rollout_source=False)
        self.assertFalse(mixed.to_dict()["require_rollout_source"])


if __name__ == "__main__":
    unittest.main()
