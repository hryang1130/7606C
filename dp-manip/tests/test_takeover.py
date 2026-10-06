import unittest
from pathlib import Path

import numpy as np

from dp_manip.takeover import (
    ARM_ACTION_SCALE,
    REST_OFFSET,
    ExpertParams,
    TransportPlaceExpert,
    TriggerParams,
    find_takeover,
    load_probe_config,
)

ROOT = Path(__file__).resolve().parents[1]


def trace(steps=200, grasp_from=None, lift_from=None, success_at=None, slip_at=None):
    """Synthetic per-state arrays: sphere rests at z=0.02 until lifted 5 cm."""
    obj = np.zeros((steps + 1, 3))
    obj[:, 2] = 0.02
    tcp = obj.copy()
    grasped = np.zeros(steps + 1, dtype=bool)
    if grasp_from is not None:
        grasped[grasp_from:] = True
    if lift_from is not None:
        obj[lift_from:, 2] = 0.07
        tcp[lift_from:, 2] = 0.07
    if slip_at is not None:
        obj[slip_at:, 0] += 0.02
    success = np.zeros(steps + 1, dtype=bool)
    if success_at is not None:
        success[success_at:] = True
    return {"obj": obj, "tcp": tcp, "grasped": grasped, "on_bin": np.zeros(steps + 1, dtype=bool),
            "success": success}


class TriggerTests(unittest.TestCase):
    p = TriggerParams()

    def test_earliest_boundary_after_full_window(self):
        # Grasped and lifted from state 30: window 16 first fits at t=45 -> boundary 48.
        result = find_takeover(trace(grasp_from=30, lift_from=30), self.p)
        self.assertEqual(result["tau"], 48)
        self.assertEqual(result["reason"], "triggered")

    def test_window_needs_every_state(self):
        result = find_takeover(trace(grasp_from=33, lift_from=30), self.p)
        self.assertEqual(result["tau"], 48)
        result = find_takeover(trace(grasp_from=34, lift_from=30), self.p)
        self.assertEqual(result["tau"], 56)

    def test_success_before_trigger_blocks_takeover(self):
        result = find_takeover(trace(grasp_from=30, lift_from=30, success_at=40), self.p)
        self.assertIsNone(result["tau"])
        self.assertEqual(result["reason"], "success_before_trigger")

    def test_reasons_without_trigger(self):
        self.assertEqual(find_takeover(trace(), self.p)["reason"], "never_grasped")
        self.assertEqual(find_takeover(trace(grasp_from=10), self.p)["reason"], "not_lifted")
        # Grasped and lifted but the sphere slides 2 cm relative to the TCP every window.
        slipping = trace(grasp_from=10, lift_from=10)
        slipping["obj"][10:, 0] = np.linspace(0, 0.3, 191)
        self.assertEqual(find_takeover(slipping, self.p)["reason"], "slip")
        late = find_takeover(trace(grasp_from=130, lift_from=130), self.p)
        self.assertEqual((late["reason"], late["late_tau"]), ("late", 152))

    def test_future_does_not_change_the_decision(self):
        a = trace(grasp_from=30, lift_from=30)
        b = trace(grasp_from=30, lift_from=30, success_at=120)
        b["grasped"][60:] = False
        self.assertEqual(find_takeover(a, self.p)["tau"], find_takeover(b, self.p)["tau"])


class ExpertTests(unittest.TestCase):
    def simulate(self, tcp, obj, bin_pos, drop_at=None, gain=0.6, steps=150):
        """Kinematic toy: the TCP moves ``gain`` of the command; a grasped sphere follows."""
        expert = TransportPlaceExpert(ExpertParams())
        tcp, obj, bin_pos = map(lambda v: np.array(v, dtype=float), (tcp, obj, bin_pos))
        grasped, velocity, actions = True, np.zeros(3), []
        for t in range(steps):
            if drop_at is not None and t >= drop_at:
                grasped = False
            state = {"tcp": tcp, "obj": obj, "bin": bin_pos, "grasped": grasped, "obj_linvel": velocity}
            action = expert.act(state)
            actions.append(action)
            if action[3] > 0:
                grasped = False
            move = gain * action[:3] * ARM_ACTION_SCALE
            tcp = tcp + move
            velocity = move * 20 if grasped else np.zeros(3)
            if grasped:
                obj = obj + move
            elif obj[2] > bin_pos[2] + REST_OFFSET and np.linalg.norm(obj[:2] - bin_pos[:2]) < 0.005:
                obj = np.array([*obj[:2], bin_pos[2] + REST_OFFSET])  # falls into the bin
        return expert, np.array(actions), obj

    def test_carries_grasped_sphere_into_bin(self):
        expert, actions, obj = self.simulate(tcp=[-0.08, 0.05, 0.08], obj=[-0.08, 0.05, 0.07],
                                             bin_pos=[0.05, -0.06, 0.0025])
        self.assertEqual(expert.phase, "hold")
        self.assertEqual([e["to"] for e in expert.events], ["descend", "release", "retract", "hold"])
        self.assertLess(np.linalg.norm(obj[:2] - [0.05, -0.06]), 0.005)
        self.assertLessEqual(np.abs(actions[:, :3]).max(), 0.2 + 1e-6)  # float32 actions
        first_open = int(np.flatnonzero(actions[:, 3] > 0)[0])
        self.assertTrue((actions[:first_open, 3] == -1).all())
        self.assertTrue((actions[first_open:, 3] == 1).all())

    def test_rises_before_crossing_when_low(self):
        expert, actions, _ = self.simulate(tcp=[-0.08, 0.0, 0.045], obj=[-0.08, 0.0, 0.035],
                                           bin_pos=[0.05, 0.0, 0.0025], steps=1)
        self.assertEqual(actions[0, 0], 0.0)
        self.assertGreater(actions[0, 2], 0.0)

    def test_lost_grasp_stops_and_is_recorded(self):
        expert, _, _ = self.simulate(tcp=[-0.08, 0.05, 0.08], obj=[-0.08, 0.05, 0.07],
                                     bin_pos=[0.05, -0.06, 0.0025], drop_at=3, steps=20)
        self.assertEqual(expert.phase, "hold")
        self.assertEqual(expert.events[-1]["reason"], "lost_grasp")


class ConfigTests(unittest.TestCase):
    def test_probe_config_parses_and_ranges_are_fresh(self):
        cfg = load_probe_config(ROOT / "configs/failure_aware/takeover_probe.toml")
        dev, probe = (range(*cfg["seeds"][k]) for k in ("development", "probe"))
        self.assertFalse(set(dev) & set(probe))
        used = [range(0, 5000), range(10000, 10100), range(20000, 23000), range(32000, 34128)]
        for seeds in (dev, probe):
            self.assertFalse(any(set(seeds) & set(r) for r in used))
        self.assertEqual(cfg["trigger"].max_t + 80, cfg["source"]["max_episode_steps"])


if __name__ == "__main__":
    unittest.main()


from dp_manip.takeover import GraspMissParams, RegraspExpert, RegraspParams, find_grasp_miss  # noqa: E402


def miss_trace(steps=200, close_at=None, grasp=None, success_at=None, lift_from=None):
    """Fingers open (0.08) until ``close_at``, then shut on nothing (0.0)."""
    base = trace(steps, lift_from=lift_from, success_at=success_at)
    base["fingers"] = np.full(steps + 1, 0.08)
    if close_at is not None:
        base["fingers"][close_at:] = 0.0
    if grasp is not None:
        base["grasped"][grasp[0]:grasp[1]] = True
    return base


class GraspMissTriggerTests(unittest.TestCase):
    p = GraspMissParams()

    def test_first_boundary_after_closing_on_nothing(self):
        # Closed from state 45: four closed states first at 48.
        self.assertEqual(find_grasp_miss(miss_trace(close_at=45), self.p)["tau"], 48)
        self.assertEqual(find_grasp_miss(miss_trace(close_at=46), self.p)["tau"], 56)

    def test_recent_contact_delays_trigger(self):
        # A glancing grasp at states 44-45 blocks t=48; 49-56 are all ungrasped.
        result = find_grasp_miss(miss_trace(close_at=40, grasp=(44, 46)), self.p)
        self.assertEqual(result["tau"], 56)

    def test_no_trigger_while_holding_or_after_success(self):
        holding = miss_trace(close_at=None, grasp=(40, 201), lift_from=40)
        self.assertEqual(find_grasp_miss(holding, self.p)["reason"], "never_closed_empty")
        done = miss_trace(close_at=60, success_at=50)
        self.assertEqual(find_grasp_miss(done, self.p)["reason"], "success_before_trigger")
        late = find_grasp_miss(miss_trace(close_at=110), self.p)
        self.assertEqual((late["reason"], late["late_tau"]), ("late", 120))

    def test_lifted_sphere_is_not_a_missed_grasp(self):
        lifted = miss_trace(close_at=60, lift_from=50)
        self.assertEqual(find_grasp_miss(lifted, self.p)["reason"], "closed_elsewhere")


class RegraspExpertTests(unittest.TestCase):
    def simulate(self, grasp_succeeds_after=0, steps=200, gain=0.6, roll=(0.0, 0.0)):
        """Toy: closing within 3 mm of the sphere grasps it (after N failed closes)."""
        expert = RegraspExpert(RegraspParams(), ExpertParams(settle_displacement=0.001))
        tcp = np.array([-0.05, 0.02, 0.025])
        obj = np.array([-0.08, 0.04, 0.02])
        bin_pos = np.array([0.05, -0.05, 0.0025])
        grasped, closes, actions = False, 0, []
        roll = np.array([*roll, 0.0])
        for _ in range(steps):
            rolling = not grasped and obj[2] < 0.021 and closes == 0
            velocity = roll if rolling else np.zeros(3)
            state = {"tcp": tcp, "obj": obj, "bin": bin_pos, "grasped": grasped, "obj_linvel": velocity}
            action = expert.act(state)
            actions.append(action)
            if action[3] < 0 and not grasped and np.linalg.norm(tcp - obj) < 0.006:
                closes += 1
                grasped = closes > grasp_succeeds_after * RegraspParams().close_steps
            if action[3] > 0:
                grasped = False
            move = gain * action[:3] * ARM_ACTION_SCALE
            tcp = tcp + move
            if grasped:
                obj = obj + move
            elif rolling:
                obj = obj + velocity * 0.05
            elif obj[2] > 0.021 and np.linalg.norm(obj[:2] - bin_pos[:2]) < 0.005:
                obj = np.array([*obj[:2], bin_pos[2] + REST_OFFSET])
        return expert, np.array(actions), obj

    def test_regrasps_then_places(self):
        expert, actions, obj = self.simulate()
        reasons = [e["reason"] for e in expert.events]
        self.assertEqual(reasons[:4], ["risen", "aligned", "lowered", "grasped"])
        self.assertEqual(reasons[-1], "retracted")
        self.assertLess(np.linalg.norm(obj[:2] - [0.05, -0.05]), 0.005)
        self.assertTrue((actions[: expert.p.open_steps, 3] == 1).all())  # reopens first
        self.assertLessEqual(np.abs(actions[:, :3]).max(), 0.2 + 1e-6)
        self.assertEqual(expert.attempts, 1)
        steps = [e["step"] for e in expert.events]
        self.assertEqual(steps, sorted(steps))

    def test_catches_a_rolling_sphere(self):
        # 0.06 m/s away from the TCP; the toy TCP covers at most 0.24 m/s.
        expert, _, obj = self.simulate(roll=(-0.06, 0.0))
        self.assertEqual(expert.attempts, 1)
        self.assertEqual(expert.events[-1]["reason"], "retracted")
        self.assertLess(np.linalg.norm(obj[:2] - [0.05, -0.05]), 0.005)

    def test_retries_once_then_stops(self):
        expert, _, _ = self.simulate(grasp_succeeds_after=1)
        self.assertEqual(expert.attempts, 2)
        self.assertIn("missed", [e["reason"] for e in expert.events])
        self.assertEqual(expert.phase, "hold")
        expert, _, _ = self.simulate(grasp_succeeds_after=5)
        self.assertEqual((expert.phase, expert.events[-1]["reason"]), ("stopped", "grasp_failed"))


class ConfigV2Tests(unittest.TestCase):
    def test_v2_ranges_are_fresh_and_values_parse(self):
        v1 = load_probe_config(ROOT / "configs/failure_aware/takeover_probe.toml")
        v2 = load_probe_config(ROOT / "configs/failure_aware/takeover_probe_v2.toml")
        self.assertEqual((v1["trigger_kind"], v2["trigger_kind"]), ("held", "grasp_miss"))
        self.assertIsNone(v1["expert"].settle_displacement)
        used = [range(0, 5000), range(10000, 10100), range(20000, 23000), range(32000, 34128)]
        used += [range(*v1["seeds"][k]) for k in ("development", "probe")]
        for key in ("development", "probe"):
            seeds = set(range(*v2["seeds"][key]))
            self.assertFalse(any(seeds & set(r) for r in used))
