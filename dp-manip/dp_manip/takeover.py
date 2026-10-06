"""Expert mid-episode takeover for PlaceSphere (docs/failure-expert-takeover.zh-CN.md).

A baseline rollout is forked at a takeover boundary ``tau``: the baseline branch
(B) keeps running the policy, the expert branch (E) replays the identical action
prefix in a fresh environment and then hands control to
:class:`TransportPlaceExpert`. The trigger and the expert read privileged task
state; neither changes what the policy observes.

Everything above :func:`make_single_env` is pure NumPy so it can be unit-tested
without ManiSkill.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

# PlaceSphere geometry (ManiSkill 3.0.1 place_sphere.py): a resting sphere's
# centre sits radius + bottom half-thickness above the bin's pose.
SPHERE_RADIUS = 0.02
BIN_BOTTOM_HALF = 0.0025
REST_OFFSET = SPHERE_RADIUS + BIN_BOTTOM_HALF
# pd_ee_delta_pos maps a normalised arm action of 1.0 to 0.1 m.
ARM_ACTION_SCALE = 0.1
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0

TASK_KEYS = ("tcp", "obj", "obj_linvel", "bin", "grasped", "on_bin", "static", "success")


@dataclass(frozen=True)
class TriggerParams:
    act_horizon: int = 8
    window: int = 16
    min_lift: float = 0.03
    max_slip: float = 0.01
    max_t: int = 120


@dataclass(frozen=True)
class GraspMissParams:
    """Oracle trigger for a missed grasp: the fingers closed on nothing."""

    act_horizon: int = 8
    closed_window: int = 4  # states with finger width below max_finger_width
    max_finger_width: float = 0.02  # metres; a held sphere keeps them ~0.035 apart
    ungrasped_window: int = 8  # states without is_obj_grasped
    max_obj_lift: float = 0.01  # sphere still within this of its reset height
    max_t: int = 96


@dataclass(frozen=True)
class RegraspParams:
    max_step: float = 0.02
    approach_height: float = 0.06  # TCP above the sphere centre while aligning
    grasp_z_offset: float = 0.0  # TCP minus sphere centre height when closing
    xy_tolerance: float = 0.003
    z_tolerance: float = 0.004
    open_steps: int = 4
    close_steps: int = 8
    confirm_steps: int = 3
    max_attempts: int = 2
    rise_timeout: int = 20
    align_timeout: int = 30
    lower_timeout: int = 25
    # A knocked sphere rolls on (0.1-0.2 m/s seen in development). Aim at where
    # it will be after the TCP could cover the gap at lead_speed, at most
    # max_lead seconds ahead; and start moving sideways during the rise once
    # the TCP is rise_clearance above the sphere centre.
    lead_speed: float = 0.15
    max_lead: float = 0.5
    rise_clearance: float = 0.03


@dataclass(frozen=True)
class ExpertParams:
    max_step: float = 0.02
    transport_clearance: float = 0.03
    release_height: float = 0.015
    min_tcp_above_bin: float = 0.025
    xy_tolerance: float = 0.004
    z_tolerance: float = 0.004
    transport_z_tolerance: float = 0.01
    settle_speed: float = 0.05
    release_steps: int = 6
    retract_height: float = 0.05
    lost_grasp_steps: int = 3
    transport_timeout: int = 60
    descend_timeout: int = 30
    retract_timeout: int = 20
    # When set, "settled" means the sphere moved less than this between two
    # states instead of using its reported linear velocity, which stays at
    # 0.06-0.18 m/s for a held sphere that is not moving (probe v1).
    settle_displacement: float | None = None


def _section(cls, values: dict[str, Any]):
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_probe_config(path: str | Path) -> dict[str, Any]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    trigger = dict(raw["trigger"])
    raw["trigger_kind"] = trigger.pop("kind", "held")
    trigger_cls = {"held": TriggerParams, "grasp_miss": GraspMissParams}[raw["trigger_kind"]]
    raw["trigger"] = _section(trigger_cls, trigger)
    raw["expert"] = _section(ExpertParams, raw["expert"])
    if raw["trigger_kind"] == "grasp_miss":
        raw["regrasp"] = _section(RegraspParams, raw["regrasp"])
    return raw


# --------------------------------------------------------------------------- #
# Trigger
# --------------------------------------------------------------------------- #


def _window_ok(trace: dict[str, np.ndarray], t: int, p: TriggerParams) -> dict[str, bool]:
    window = slice(t - p.window + 1, t + 1)
    relative = trace["obj"][window] - trace["tcp"][window]
    slip = float(np.max(np.linalg.norm(relative - relative[-1], axis=1)))
    return {
        "grasped": bool(trace["grasped"][window].all()),
        "lifted": bool((trace["obj"][window, 2] >= trace["obj"][0, 2] + p.min_lift).all()),
        "no_slip": slip <= p.max_slip,
        "not_on_bin": not bool(trace["on_bin"][t]),
    }


def find_takeover(trace: dict[str, np.ndarray], p: TriggerParams) -> dict[str, Any]:
    """Earliest boundary satisfying the trigger, or why there is none.

    ``trace`` holds per-state arrays indexed by executed actions (index 0 is the
    reset state). Only states up to the candidate boundary are read, so the
    decision never uses the episode's future.
    """
    steps = len(trace["grasped"]) - 1
    flags: dict[str, bool] = {"grasped": False, "lifted": False, "no_slip": False}
    first_success = np.flatnonzero(trace["success"][1:])
    first_success = int(first_success[0]) + 1 if first_success.size else None
    late = None
    for t in range(p.window, steps + 1):
        if t % p.act_horizon:
            continue
        if first_success is not None and first_success <= t:
            if t <= p.max_t:
                return {"tau": None, "reason": "success_before_trigger", "first_success": first_success}
            break
        checks = _window_ok(trace, t, p)
        if all(checks.values()):
            if t <= p.max_t:
                return {"tau": t, "reason": "triggered", "first_success": first_success}
            late = t
            break
        if t <= p.max_t:
            # Track the furthest the rule got before max_t, in trigger order.
            flags["grasped"] |= checks["grasped"]
            flags["lifted"] |= checks["grasped"] and checks["lifted"]
            flags["no_slip"] |= checks["grasped"] and checks["lifted"] and checks["no_slip"]
    if late is not None:
        reason = "late"
    elif not trace["grasped"][: p.max_t + 1].any():
        reason = "never_grasped"
    elif not flags["grasped"]:
        reason = "grasp_not_stable"
    elif not flags["lifted"]:
        reason = "not_lifted"
    elif not flags["no_slip"]:
        reason = "slip"
    else:
        reason = "on_bin"
    return {"tau": None, "reason": reason, "first_success": first_success, "late_tau": late}


def find_grasp_miss(trace: dict[str, np.ndarray], p: GraspMissParams) -> dict[str, Any]:
    """Earliest boundary at which the gripper has closed on nothing.

    For states up to the boundary t: the finger width stayed below
    ``max_finger_width`` for the last ``closed_window`` states, the sphere was
    not grasped in the last ``ungrasped_window`` states and is still near its
    reset height, and the task never succeeded. Like :func:`find_takeover` it
    reads no state after t.
    """
    steps = len(trace["grasped"]) - 1
    first_success = np.flatnonzero(trace["success"][1:])
    first_success = int(first_success[0]) + 1 if first_success.size else None
    start = max(p.closed_window, p.ungrasped_window) - 1
    closed_seen = False
    for t in range(p.act_horizon, steps + 1, p.act_horizon):
        if t < start:
            continue
        if first_success is not None and first_success <= t:
            reason = "success_before_trigger" if t <= p.max_t else "none_by_max_t"
            return {"tau": None, "reason": reason, "first_success": first_success}
        closed = bool((trace["fingers"][t - p.closed_window + 1 : t + 1] < p.max_finger_width).all())
        ungrasped = not trace["grasped"][t - p.ungrasped_window + 1 : t + 1].any()
        low = trace["obj"][t, 2] < trace["obj"][0, 2] + p.max_obj_lift
        closed_seen |= closed
        if closed and ungrasped and low:
            if t <= p.max_t:
                return {"tau": t, "reason": "triggered", "first_success": first_success}
            return {"tau": None, "reason": "late", "first_success": first_success, "late_tau": t}
    reason = "closed_elsewhere" if closed_seen else "never_closed_empty"
    return {"tau": None, "reason": reason, "first_success": first_success}


# --------------------------------------------------------------------------- #
# Expert
# --------------------------------------------------------------------------- #


class TransportPlaceExpert:
    """Carry an already grasped sphere over the bin, lower it and let go.

    Closed loop on privileged state: each step it commands the TCP translation
    that moves the sphere toward the phase's goal, clipped to ``max_step`` per
    axis. It never reopens the gripper before the release phase, never
    regrasps, and keeps the orientation the baseline left (the controller is
    position-only).
    """

    PHASES = ("transport", "descend", "release", "retract", "hold")

    def __init__(self, params: ExpertParams):
        self.p = params
        self.phase = "transport"
        self.phase_steps = 0
        self.ungrasped = 0
        self.events: list[dict[str, Any]] = []
        self.release_tcp_z: float | None = None
        self.previous_obj: np.ndarray | None = None
        self.steps = 0

    def _goto(self, phase: str, reason: str) -> None:
        self.events.append({"step": self.steps, "from": self.phase, "to": phase, "reason": reason})
        self.phase = phase
        self.phase_steps = 0

    def _arm(self, tcp_delta: np.ndarray) -> np.ndarray:
        clipped = np.clip(tcp_delta, -self.p.max_step, self.p.max_step)
        return clipped / ARM_ACTION_SCALE

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        """Return one normalised 4-D action for ``state`` and advance the phase."""
        p = self.p
        tcp, obj, bin_pos = state["tcp"], state["obj"], state["bin"]
        rest_z = bin_pos[2] + REST_OFFSET
        min_tcp_z = bin_pos[2] + p.min_tcp_above_bin

        if self.phase in ("transport", "descend"):
            self.ungrasped = 0 if bool(state["grasped"]) else self.ungrasped + 1
            if self.ungrasped >= p.lost_grasp_steps:
                self._goto("hold", "lost_grasp")

        def toward(goal_obj: np.ndarray) -> np.ndarray:
            delta = goal_obj - obj
            # The sphere moves with the TCP while grasped; never drive the
            # fingers into the bin walls.
            delta[2] = max(delta[2], min_tcp_z - tcp[2])
            return delta

        arm = np.zeros(3)
        gripper = GRIPPER_CLOSED
        if self.phase == "transport":
            goal = np.array([bin_pos[0], bin_pos[1], rest_z + p.transport_clearance])
            delta = toward(goal)
            if obj[2] < goal[2] - p.transport_z_tolerance and np.linalg.norm(delta[:2]) > p.xy_tolerance:
                delta[:2] = 0.0  # rise to clearance height before crossing the bin wall
            arm = self._arm(delta)
            if np.linalg.norm(goal[:2] - obj[:2]) < p.xy_tolerance and abs(goal[2] - obj[2]) < p.transport_z_tolerance:
                self._goto("descend", "aligned")
            elif self.phase_steps + 1 >= p.transport_timeout:
                self._goto("descend", "timeout")
        elif self.phase == "descend":
            goal = np.array([bin_pos[0], bin_pos[1], rest_z + p.release_height])
            arm = self._arm(toward(goal))
            if p.settle_displacement is None:
                still = float(np.linalg.norm(state["obj_linvel"])) < p.settle_speed
            else:
                moved = np.inf if self.previous_obj is None else np.linalg.norm(obj - self.previous_obj)
                still = bool(moved < p.settle_displacement)
            if (
                np.linalg.norm(goal[:2] - obj[:2]) < p.xy_tolerance
                and abs(goal[2] - obj[2]) < p.z_tolerance
                and still
            ):
                self._goto("release", "settled")
            elif self.phase_steps + 1 >= p.descend_timeout:
                self._goto("release", "timeout")
        elif self.phase == "release":
            gripper = GRIPPER_OPEN
            if self.release_tcp_z is None:
                self.release_tcp_z = float(tcp[2])
            if self.phase_steps + 1 >= p.release_steps:
                self._goto("retract", "opened")
        elif self.phase == "retract":
            gripper = GRIPPER_OPEN
            target = (self.release_tcp_z if self.release_tcp_z is not None else tcp[2]) + p.retract_height
            arm = self._arm(np.array([0.0, 0.0, target - tcp[2]]))
            if target - tcp[2] < p.z_tolerance:
                self._goto("hold", "retracted")
            elif self.phase_steps + 1 >= p.retract_timeout:
                self._goto("hold", "timeout")
        else:  # hold
            gripper = GRIPPER_OPEN

        self.previous_obj = np.array(obj, dtype=float)
        self.phase_steps += 1
        self.steps += 1
        return np.array([*arm, gripper], dtype=np.float32)


class RegraspExpert:
    """Reopen after a missed grasp, grasp the sphere from above, then place it.

    Phases rise (open, lift to the approach height) -> align (over the
    sphere) -> lower (to the grasp height) -> close (until the grasp holds for
    ``confirm_steps``). A close that does not hold retries from rise, up to
    ``max_attempts``; afterwards the expert stops (``grasp_failed``). Once the
    grasp holds, :class:`TransportPlaceExpert` takes over unchanged.
    """

    def __init__(self, params: RegraspParams, place: ExpertParams):
        self.p = params
        self.place_params = place
        self.place: TransportPlaceExpert | None = None
        self.phase = "rise"
        self.phase_steps = 0
        self.attempts = 1
        self.held = 0
        self.events: list[dict[str, Any]] = []
        self.steps = 0
        self.grasped_at: int | None = None
        self._handoff_events = 0

    def _goto(self, phase: str, reason: str) -> None:
        self.events.append({"step": self.steps, "from": self.phase, "to": phase, "reason": reason})
        self.phase = phase
        self.phase_steps = 0

    def _arm(self, tcp_delta: np.ndarray) -> np.ndarray:
        return np.clip(tcp_delta, -self.p.max_step, self.p.max_step) / ARM_ACTION_SCALE

    def act(self, state: dict[str, np.ndarray]) -> np.ndarray:
        if self.place is not None:
            action = self.place.act(state)
            for event in self.place.events[len(self.events) - self._handoff_events :]:
                self.events.append({**event, "step": event["step"] + self.grasped_at})
            self.phase = self.place.phase
            self.steps += 1
            return action

        p = self.p
        tcp, obj = state["tcp"], np.array(state["obj"], dtype=float)
        approach_z = obj[2] + p.approach_height
        if self.phase in ("rise", "align", "lower"):
            velocity = np.asarray(state["obj_linvel"], dtype=float)[:2]
            lead = min(np.linalg.norm(obj[:2] - tcp[:2]) / p.lead_speed, p.max_lead)
            obj[:2] = obj[:2] + velocity * lead
        arm, gripper = np.zeros(3), GRIPPER_OPEN
        if self.phase == "rise":
            sideways = obj[:2] - tcp[:2] if tcp[2] >= obj[2] + p.rise_clearance else np.zeros(2)
            arm = self._arm(np.array([*sideways, approach_z - tcp[2]]))
            if self.phase_steps + 1 >= p.open_steps and abs(approach_z - tcp[2]) < p.z_tolerance:
                self._goto("align", "risen")
            elif self.phase_steps + 1 >= p.rise_timeout:
                self._goto("align", "timeout")
        elif self.phase == "align":
            arm = self._arm(np.array([obj[0] - tcp[0], obj[1] - tcp[1], approach_z - tcp[2]]))
            if np.linalg.norm(obj[:2] - tcp[:2]) < p.xy_tolerance:
                self._goto("lower", "aligned")
            elif self.phase_steps + 1 >= p.align_timeout:
                self._goto("lower", "timeout")
        elif self.phase == "lower":
            target_z = obj[2] + p.grasp_z_offset
            arm = self._arm(np.array([obj[0] - tcp[0], obj[1] - tcp[1], target_z - tcp[2]]))
            if abs(target_z - tcp[2]) < p.z_tolerance and np.linalg.norm(obj[:2] - tcp[:2]) < p.xy_tolerance:
                self._goto("close", "lowered")
            elif self.phase_steps + 1 >= p.lower_timeout:
                self._goto("close", "timeout")
        elif self.phase == "close":
            gripper = GRIPPER_CLOSED
            self.held = self.held + 1 if bool(state["grasped"]) else 0
            if self.held >= p.confirm_steps:
                self._goto("place", "grasped")
                self.grasped_at = self.steps + 1
                self.place = TransportPlaceExpert(self.place_params)
                self._handoff_events = len(self.events)
            elif self.phase_steps + 1 >= p.close_steps:
                if self.attempts < p.max_attempts:
                    self.attempts += 1
                    self.held = 0
                    self._goto("rise", "missed")
                else:
                    self._goto("stopped", "grasp_failed")
        # "stopped": stay open and still.

        self.phase_steps += 1
        self.steps += 1
        return np.array([*arm, gripper], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Environment helpers (ManiSkill imports deferred)
# --------------------------------------------------------------------------- #


def make_single_env(cfg, render_backend: str | None = "cpu"):
    """One unvectorised env built exactly like :func:`dp_manip.envs.make_eval_envs`."""
    from .envs import environment_kwargs, ensure_render_icd

    if cfg.task.sim_backend != "physx_cpu":
        raise ValueError("takeover replay requires physx_cpu")
    ensure_render_icd()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers import CPUGymWrapper
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    env = gym.make(cfg.task.env_id, **environment_kwargs(cfg, render_backend))
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def read_task_state(env, info: dict[str, Any]) -> dict[str, Any]:
    """Privileged per-step record; task flags come from the env's own ``info``."""
    unwrapped = env.unwrapped

    def vector(value) -> np.ndarray:
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64).reshape(-1)

    return {
        "tcp": vector(unwrapped.agent.tcp.pose.p),
        "obj": vector(unwrapped.obj.pose.p),
        "obj_linvel": vector(unwrapped.obj.linear_velocity),
        "bin": vector(unwrapped.bin.pose.p),
        "fingers": float(vector(unwrapped.agent.robot.get_qpos())[7:9].sum()),
        "grasped": bool(np.asarray(info["is_obj_grasped"]).reshape(-1)[0]),
        "on_bin": bool(np.asarray(info["is_obj_on_bin"]).reshape(-1)[0]),
        "static": bool(np.asarray(info["is_obj_static"]).reshape(-1)[0]),
        "success": bool(np.asarray(info["success"]).reshape(-1)[0]),
        "sim_state": vector(unwrapped.get_state()),
    }


def stack_states(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.stack([np.asarray(state[key]) for state in states]) for key in states[0]}
