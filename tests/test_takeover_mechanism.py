import unittest

import numpy as np

from dp_manip.takeover_mechanism import MechanismParams, episode_metrics, summarize


def episode(steps=200, close=(), grasp=(), success_at=None):
    """Fingers open (0.04 each) except in ``close`` ranges (0); grasped in ``grasp`` ranges."""
    proprio = np.zeros((steps + 1, 29), np.float32)
    proprio[:, 7:9] = 0.04
    for a, b in close:
        proprio[a:b, 7:9] = 0.0
    for a, b in grasp:
        proprio[a:b, 7:9] = 0.0175  # holding the sphere: width 0.035
        proprio[a:b, 18] = 1.0
    success = np.zeros(steps, bool)
    if success_at is not None:
        success[success_at - 1:] = True
    return proprio, success


class EpisodeMetricsTests(unittest.TestCase):
    p = MechanismParams()

    def test_missed_grasp_then_failure(self):
        m = episode_metrics(*episode(close=[(45, 201)]), self.p)
        self.assertEqual((m["missed"], m["first_miss"], m["miss_events"]), (True, 48, 1))
        self.assertEqual(m["stage"], "fail_no_stable_grasp")
        self.assertTrue(m["miss_failure"])

    def test_recovered_after_miss(self):
        # Miss at 45-60, reopen, stable grasp 80-130, success at 140.
        m = episode_metrics(*episode(close=[(45, 60)], grasp=[(80, 130)], success_at=140), self.p)
        self.assertTrue(m["missed"] and m["success"])
        self.assertEqual(m["stable_grasp_at"], 95)
        self.assertFalse(m["miss_failure"])

    def test_two_separate_misses_and_misses_after_success_ignored(self):
        m = episode_metrics(*episode(close=[(45, 60), (100, 120)]), self.p)
        self.assertEqual(m["miss_events"], 2)
        late = episode_metrics(*episode(grasp=[(40, 90)], close=[(150, 201)], success_at=100), self.p)
        self.assertFalse(late["missed"])

    def test_holding_is_not_a_miss(self):
        m = episode_metrics(*episode(grasp=[(40, 201)]), self.p)
        self.assertFalse(m["missed"])
        self.assertEqual(m["stage"], "fail_after_stable_grasp")

    def test_summary_rates(self):
        rows = [episode_metrics(*e, self.p) for e in (
            episode(close=[(45, 201)]),
            episode(close=[(45, 60)], grasp=[(80, 130)], success_at=140),
            episode(grasp=[(40, 90)], success_at=100),
            episode(grasp=[(40, 201)]),
        )]
        s = summarize(rows)
        self.assertEqual((s["missed"], s["success_after_miss"], s["success_without_miss"]), (0.5, 0.5, 0.5))
        self.assertEqual(s["miss_failure"], 0.25)
        self.assertEqual(s["stages"], {"success": 2, "fail_no_stable_grasp": 1, "fail_after_stable_grasp": 1})

    def test_rejects_other_layouts(self):
        with self.assertRaises(ValueError):
            episode_metrics(np.zeros((10, 9)), np.zeros(9, bool), self.p)


if __name__ == "__main__":
    unittest.main()
