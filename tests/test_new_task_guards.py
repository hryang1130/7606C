"""Guards that stop a newly added task from silently producing zero results.

PlaceSphere was trained with ManiSkill's registered 50-step horizon although its
demonstrations need 90-150 steps, so every evaluation failed without an error.
Training now refuses such a horizon. The replication task of the failure-aware
study (plan §12.3) must stop, not fall into low-success mode, when its baseline
is outside the band.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from dp_manip import config as config_lib
from dp_manip import failure_study as study

try:
    from dp_manip.data import DatasetInfo, EpisodeInfo
    from dp_manip.trainer import check_horizon
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

TASKS = Path(__file__).resolve().parents[1] / "configs" / "tasks"


def dataset(*lengths: int) -> "DatasetInfo":
    episodes = tuple(EpisodeInfo(f"traj_{i}", i, i, length) for i, length in enumerate(lengths))
    return DatasetInfo(Path("unused.h5"), "PlaceSphere-v1", "pd_ee_delta_pos", episodes, None, 3, 4, (), None)


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class HorizonGuardTest(unittest.TestCase):
    def test_horizon_shorter_than_a_demonstration_is_refused(self) -> None:
        cfg = config_lib.load(TASKS / "placesphere.toml", ["task.max_episode_steps=50"])
        with self.assertRaisesRegex(ValueError, r"shorter than the longest demonstration \(150 steps\).*e\.g\. 250"):
            check_horizon(cfg, dataset(90, 113, 150), dataset(117))

    def test_horizon_covering_every_demonstration_passes(self) -> None:
        cfg = config_lib.load(TASKS / "placesphere.toml")
        self.assertEqual(cfg.task.max_episode_steps, 200)
        check_horizon(cfg, dataset(90, 113, 150), dataset(145))
        check_horizon(config_lib.load(TASKS / "placesphere.toml", ["task.max_episode_steps=150"]), dataset(150))


class ReplicationCellTest(unittest.TestCase):
    def test_low_success_mode_can_be_disabled(self) -> None:
        pick = study.select_baseline_cell
        self.assertEqual(pick([0.2, 0.3], None, low=0.15, high=0.85, allow_low_success=False)["mode"], "normal")
        normal200 = pick([0.1, 0.1], [0.2, 0.3], low=0.15, high=0.85, allow_low_success=False)
        self.assertEqual((normal200["num_demos"], normal200["mode"]), (200, "normal"))
        self.assertEqual(pick([0.02], [0.04], low=0.15, high=0.85)["mode"], "low-success")
        with self.assertRaisesRegex(ValueError, "study is not run on this task"):
            pick([0.02], [0.04], low=0.15, high=0.85, allow_low_success=False)


if __name__ == "__main__":
    unittest.main()
