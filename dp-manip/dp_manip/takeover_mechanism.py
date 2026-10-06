"""Grasp-stage metrics of evaluation episodes, from proprioception only
(configs/failure_aware/takeover_curve.toml [mechanism]).

PlaceSphere's flattened RGB state is qpos (9), qvel (9), is_grasped (1),
tcp_pose (7), bin_pos (3); the two finger joints are qpos 7 and 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

PROPRIO_DIM = 29
FINGER_COLUMNS = (7, 8)
GRASPED_COLUMN = 18


@dataclass(frozen=True)
class MechanismParams:
    act_horizon: int = 8
    closed_window: int = 4
    max_finger_width: float = 0.02
    ungrasped_window: int = 8
    stable_grasp_window: int = 16


def episode_metrics(proprio: np.ndarray, success: np.ndarray, p: MechanismParams) -> dict[str, Any]:
    """``proprio`` (T+1, 29) indexed by executed actions; ``success`` (T,) after each action."""
    proprio = np.asarray(proprio)
    if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM:
        raise ValueError(f"expected (T+1, {PROPRIO_DIM}) PlaceSphere proprio, got {proprio.shape}")
    fingers = proprio[:, FINGER_COLUMNS[0]] + proprio[:, FINGER_COLUMNS[1]]
    grasped = proprio[:, GRASPED_COLUMN] > 0.5
    steps = len(proprio) - 1
    hits = np.flatnonzero(np.asarray(success, dtype=bool))
    first_success = int(hits[0]) + 1 if hits.size else None

    start = max(p.closed_window, p.ungrasped_window) - 1
    boundaries = [
        t for t in range(p.act_horizon, steps + 1, p.act_horizon)
        if t >= start
        and (fingers[t - p.closed_window + 1 : t + 1] < p.max_finger_width).all()
        and not grasped[t - p.ungrasped_window + 1 : t + 1].any()
    ]
    if first_success is not None:
        boundaries = [t for t in boundaries if t < first_success]
    # Consecutive boundaries belong to one missed grasp.
    events = sum(1 for i, t in enumerate(boundaries) if i == 0 or t - boundaries[i - 1] != p.act_horizon)

    run, stable_at = 0, None
    for t, held in enumerate(grasped):
        run = run + 1 if held else 0
        if run >= p.stable_grasp_window:
            stable_at = t
            break
    succeeded = first_success is not None
    stage = "success" if succeeded else ("fail_no_stable_grasp" if stable_at is None else "fail_after_stable_grasp")
    return {
        "success": succeeded,
        "first_success": first_success,
        "missed": bool(boundaries),
        "first_miss": boundaries[0] if boundaries else None,
        "miss_events": events,
        "stable_grasp_at": stable_at,
        "stage": stage,
        "miss_failure": bool(boundaries) and not succeeded,
    }


def summarize(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    missed = [e for e in episodes if e["missed"]]
    clean = [e for e in episodes if not e["missed"]]

    def rate(rows, key):
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else None

    return {
        "episodes": n,
        "success": rate(episodes, "success"),
        "missed": rate(episodes, "missed"),
        "success_after_miss": rate(missed, "success"),
        "success_without_miss": rate(clean, "success"),
        "miss_failure": rate(episodes, "miss_failure"),
        "mean_miss_events": float(np.mean([e["miss_events"] for e in episodes])) if n else None,
        "stages": {s: sum(e["stage"] == s for e in episodes) for s in
                   ("success", "fail_no_stable_grasp", "fail_after_stable_grasp")},
    }
