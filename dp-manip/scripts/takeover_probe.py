#!/usr/bin/env python3
"""Expert mid-episode takeover feasibility probe (configs/failure_aware/takeover_probe.toml).

For every seed of the chosen split the locked baseline runs a full 200-step
episode (the B branch, recorded with privileged task state). If the trigger
fires at ``tau``, a fresh environment replays the identical action prefix,
checks that state, RGB and proprio at ``tau`` equal the source episode, and
hands control to the scripted expert for the remaining ``200 - tau`` steps (E).
A third replay continues the baseline from the restored observation history
and generator state to confirm the B branch is reproducible from the fork.

Per-seed records are write-once, so an interrupted run resumes. The probe
split's decision is computed from all records and is write-once too.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dp_manip.failure_round2 import write_once
from dp_manip.metadata import file_sha256, git_revision
from dp_manip.takeover import (
    RegraspExpert,
    TransportPlaceExpert,
    find_grasp_miss,
    find_takeover,
    load_probe_config,
    make_single_env,
    read_task_state,
    stack_states,
)


class Branch:
    """Everything one branch observed and did, indexed by executed actions."""

    def __init__(self):
        self.rgb, self.proprio, self.states, self.actions = [], [], [], []
        self.phases: list[str] = []

    def record(self, env, observation, info):
        from dp_manip.evaluate import _adapt_environment_observation

        rgb, proprio = _adapt_environment_observation(
            {"rgb": np.asarray(observation["rgb"])[None], "state": np.asarray(observation["state"])[None]}, 1
        )
        self.rgb.append(rgb[0])
        self.proprio.append(proprio[0])
        self.states.append(read_task_state(env, info))

    def history(self, obs_horizon):
        """The policy's observation history after the latest state (evaluate's padding)."""
        rgb = [self.rgb[max(0, len(self.rgb) - obs_horizon + i)] for i in range(obs_horizon)]
        proprio = [self.proprio[max(0, len(self.proprio) - obs_horizon + i)] for i in range(obs_horizon)]
        return np.stack(rgb), np.stack(proprio)

    @property
    def success(self):
        return np.array([state["success"] for state in self.states[1:]], dtype=bool)


def step(env, branch, action, t, max_steps):
    observation, _, _, truncated, info = env.step(action)
    branch.actions.append(np.asarray(action, dtype=np.float32))
    branch.record(env, observation, info)
    if bool(truncated) != (t + 1 == max_steps):
        raise RuntimeError(f"truncation at step {t + 1} disagrees with max_episode_steps={max_steps}")


def start(env, seed, prefix, max_steps):
    branch = Branch()
    observation, info = env.reset(seed=seed)
    branch.record(env, observation, info)
    for t, action in enumerate(prefix):
        step(env, branch, action, t, max_steps)
        branch.phases.append("prefix")
    return branch


def run_policy(env, policy, device, seed, gen_seed, max_steps, prefix=(), gen_state=None):
    import torch

    branch = start(env, seed, prefix, max_steps)
    generator = torch.Generator(device=device).manual_seed(gen_seed)
    if gen_state is not None:
        generator.set_state(gen_state)
    gen_states = {}
    t = len(prefix)
    while t < max_steps:
        if t % policy.act_horizon:
            raise RuntimeError("policy branches start on action-chunk boundaries")
        gen_states[t] = generator.get_state().clone()
        rgb, proprio = branch.history(policy.obs_horizon)
        rgb_tensor = torch.as_tensor(np.transpose(rgb[None], (0, 1, 4, 2, 3)), device=device, dtype=torch.uint8)
        proprio_tensor = torch.as_tensor(proprio[None], device=device, dtype=torch.float32)
        chunk = policy.get_action(rgb_tensor, proprio_tensor, generator=generator).cpu().numpy()[0]
        for action in chunk:
            step(env, branch, action, t, max_steps)
            branch.phases.append("policy")
            t += 1
            if t == max_steps:
                break
    return branch, gen_states


def make_expert(cfg):
    if cfg["trigger_kind"] == "grasp_miss":
        return RegraspExpert(cfg["regrasp"], cfg["expert"])
    return TransportPlaceExpert(cfg["expert"])


def run_expert(env, seed, prefix, cfg, max_steps):
    branch = start(env, seed, prefix, max_steps)
    expert = make_expert(cfg)
    for t in range(len(prefix), max_steps):
        phase = expert.phase
        action = expert.act(branch.states[-1])
        step(env, branch, action, t, max_steps)
        branch.phases.append(phase)
    return branch, expert


def first_success(success, offset=0):
    hits = np.flatnonzero(success[offset:])
    return int(hits[0]) + offset + 1 if hits.size else None


def save_video(target, left, right, tau):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    frames = np.concatenate([left[..., :3], right[..., :3]], axis=2)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(target, frames, fps=20, codec="libx264")
    picks = sorted({*np.linspace(0, len(frames) - 1, 11, dtype=int).tolist(), tau})
    height, width = frames.shape[1:3]
    sheet = Image.new("RGB", (3 * width, 4 * (height + 16)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, t in enumerate(picks[:12]):
        x, y = (index % 3) * width, (index // 3) * (height + 16)
        sheet.paste(Image.fromarray(frames[t]), (x, y + 16))
        draw.text((x + 2, y + 2), f"t={t}{' tau' if t == tau else ''}  B | E", fill="black")
    sheet.save(target.with_suffix(".png"))


def probe_seed(seed, ctx):
    policy, device, env, cfg, out = ctx["policy"], ctx["device"], ctx["env"], ctx["cfg"], ctx["out"]
    max_steps = cfg["source"]["max_episode_steps"]
    gen_seed = cfg["source"]["inference_seed"] + seed
    tick = time.time()
    source, gen_states = run_policy(env, policy, device, seed, gen_seed, max_steps)
    trace = stack_states(source.states)
    find = find_grasp_miss if cfg["trigger_kind"] == "grasp_miss" else find_takeover
    trigger = find(trace, cfg["trigger"])
    record = {
        "seed": seed,
        "identity": ctx["identity"],
        "source": {
            "success_once": bool(source.success.any()),
            "first_success": first_success(source.success),
            "steps": len(source.actions),
        },
        "trigger": trigger,
    }
    arrays = {f"B_{key}": value for key, value in trace.items()}
    arrays["B_actions"] = np.stack(source.actions)
    tau = trigger["tau"]
    if tau is not None:
        prefix = source.actions[:tau]
        expert_branch, expert = run_expert(env, seed, prefix, cfg, max_steps)
        state_diff = float(np.max(np.abs(expert_branch.states[tau]["sim_state"] - source.states[tau]["sim_state"])))
        replay = {
            "sim_state_equal": state_diff == 0.0,
            "max_sim_state_diff": state_diff,
            "rgb_equal": all(np.array_equal(a, b) for a, b in zip(expert_branch.rgb[: tau + 1], source.rgb[: tau + 1])),
            "proprio_equal": all(
                np.array_equal(a, b) for a, b in zip(expert_branch.proprio[: tau + 1], source.proprio[: tau + 1])
            ),
            "prefix_task_state_equal": all(
                expert_branch.states[i][key] == source.states[i][key]
                for i in range(tau + 1)
                for key in ("grasped", "on_bin", "static", "success")
            ),
        }
        e_trace = stack_states(expert_branch.states)
        e_success = expert_branch.success
        record["replay"] = replay
        record["expert"] = {
            "success_once": bool(e_success[tau:].any()),
            "first_success": first_success(e_success, tau),
            "success_at_end": bool(e_success[-1]),
            "budget": max_steps - tau,
            "steps": len(expert_branch.actions) - tau,
            "final_phase": expert.phase,
            "events": expert.events,
            "lost_grasp": any(event["reason"] == "lost_grasp" for event in expert.events),
            "timeouts": [event["from"] for event in expert.events if event["reason"] == "timeout"],
            "attempts": getattr(expert, "attempts", None),
            "grasped_at": getattr(expert, "grasped_at", None),
            "grasp_failed": any(event["reason"] == "grasp_failed" for event in expert.events),
        }
        b_branch, _ = run_policy(env, policy, device, seed, gen_seed, max_steps, prefix, gen_states[tau])
        b_actions = np.stack(b_branch.actions)
        source_actions = np.stack(source.actions)
        record["baseline_replay"] = {
            "actions_equal": bool(np.array_equal(b_actions, source_actions)),
            "max_action_diff": float(np.max(np.abs(b_actions - source_actions))),
            "success_once": bool(b_branch.success.any()),
            "outcome_equal": bool(b_branch.success.any()) == record["source"]["success_once"],
        }
        arrays.update({f"E_{key}": value for key, value in e_trace.items()})
        arrays["E_actions"] = np.stack(expert_branch.actions)
        arrays["E_phase"] = np.array(expert_branch.phases)
        arrays["E_rgb"] = np.stack(expert_branch.rgb)
        arrays["E_proprio"] = np.stack(expert_branch.proprio)
        arrays["B_rgb"] = np.stack(source.rgb)
        arrays["B_proprio"] = np.stack(source.proprio)
        save_video(out / "videos" / f"seed{seed}.mp4", arrays["B_rgb"], arrays["E_rgb"], tau)
    record["wall_s"] = time.time() - tick
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "episodes" / f"seed{seed}.npz", **arrays)
    return write_once(out / "records" / f"seed{seed}.json", record)


def summarize(records, gates):
    triggered = [r for r in records if r["trigger"]["tau"] is not None]
    failures = [r for r in triggered if not r["source"]["success_once"]]
    reasons = {}
    for r in records:
        reasons[r["trigger"]["reason"]] = reasons.get(r["trigger"]["reason"], 0) + 1

    def rate(rows, value):
        return sum(value(r) for r in rows) / len(rows) if rows else None

    replay_ok = lambda r: all(r["replay"][k] for k in ("sim_state_equal", "rgb_equal", "proprio_equal", "prefix_task_state_equal"))
    b_ok = lambda r: r["baseline_replay"]["actions_equal"] and r["baseline_replay"]["outcome_equal"]
    e_ok = lambda r: r["expert"]["success_once"]
    metrics = {
        "seeds": len(records),
        "source_successes": sum(r["source"]["success_once"] for r in records),
        "trigger_reasons": reasons,
        "triggered": len(triggered),
        "triggered_baseline_failures": len(failures),
        "triggered_baseline_successes": len(triggered) - len(failures),
        "baseline_failures": sum(not r["source"]["success_once"] for r in records),
        "tau": sorted(r["trigger"]["tau"] for r in triggered),
        "replay_state_match": rate(triggered, replay_ok),
        "baseline_replay_match": rate(triggered, b_ok),
        "expert_success": rate(triggered, e_ok),
        "expert_success_on_failures": rate(failures, e_ok),
        "expert_success_on_successes": rate([r for r in triggered if r["source"]["success_once"]], e_ok),
        "expert_lost_grasp": sum(r["expert"]["lost_grasp"] for r in triggered),
        "expert_timeouts": sum(bool(r["expert"]["timeouts"]) for r in triggered),
        "expert_grasp_failed": sum(bool(r["expert"].get("grasp_failed")) for r in triggered),
        "expert_retried": sum((r["expert"].get("attempts") or 1) > 1 for r in triggered),
        "budget_respected": all(r["expert"]["steps"] == r["expert"]["budget"] for r in triggered),
        "rescued": sorted(r["seed"] for r in failures if e_ok(r)),
        "expert_failed": sorted(r["seed"] for r in triggered if not e_ok(r)),
        "wall_s": sum(r["wall_s"] for r in records),
    }
    checks = {
        "replay_state_match": (metrics["replay_state_match"] or 0) >= gates["replay_state_match"],
        "baseline_replay_match": (metrics["baseline_replay_match"] or 0) >= gates["baseline_replay_match"],
        "min_triggered_baseline_failures": len(failures) >= gates["min_triggered_baseline_failures"],
        "min_expert_success": (metrics["expert_success"] or 0) >= gates["min_expert_success"],
        "min_expert_success_on_failures": (metrics["expert_success_on_failures"] or 0)
        >= gates["min_expert_success_on_failures"],
        "budget_respected": metrics["budget_respected"],
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/failure_aware/takeover_probe.toml")
    parser.add_argument("--split", choices=("development", "probe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="development only: run the first N seeds")
    args = parser.parse_args()
    if args.limit is not None and args.split == "probe":
        parser.error("--limit is for development; the probe runs its whole range")

    import tomllib

    import torch

    from dp_manip.config import from_recorded
    from dp_manip.policy import DiffusionPolicy

    cfg = load_probe_config(args.config)
    lock = tomllib.loads((ROOT / cfg["source"]["lock"]).read_text())
    entry = lock["checkpoints"][cfg["source"]["checkpoint"]]
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("baseline checkpoint hash changed")
    identity = {
        "config_sha256": file_sha256(args.config),
        "lock_sha256": file_sha256(ROOT / cfg["source"]["lock"]),
        "checkpoint": entry["path"],
        "checkpoint_sha256": entry["sha256"],
        "takeover_module_sha256": file_sha256(ROOT / "dp_manip/takeover.py"),
        "runner_sha256": file_sha256(Path(__file__)),
    }
    out = args.output / args.split
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "identity.json", identity)
    (out / "run_meta.json").write_text(json.dumps({"git": git_revision(ROOT)}, indent=2) + "\n")

    device = torch.device("cuda")
    payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
    run_cfg = from_recorded(payload["config"])
    run_cfg.task.max_episode_steps = cfg["source"]["max_episode_steps"]
    policy = DiffusionPolicy.from_checkpoint(payload, device).eval()
    del payload
    if policy.act_horizon != cfg["trigger"].act_horizon:
        raise ValueError("trigger act_horizon differs from the policy's")
    env = make_single_env(run_cfg, "cpu")
    ctx = {"policy": policy, "device": device, "env": env, "cfg": cfg, "out": out, "identity": identity}
    seeds = list(range(*cfg["seeds"][args.split]))[: args.limit]
    records = []
    try:
        for seed in seeds:
            path = out / "records" / f"seed{seed}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["identity"] != identity:
                    raise ValueError(f"record from a different probe identity: {path}")
            else:
                record = probe_seed(seed, ctx)
            records.append(record)
            brief = {k: record[k] for k in ("seed",)} | {
                "B": record["source"]["success_once"],
                "tau": record["trigger"]["tau"],
                "reason": record["trigger"]["reason"],
                "E": record.get("expert", {}).get("success_once"),
                "events": [e["to"] + ":" + e["reason"] for e in record.get("expert", {}).get("events", [])],
                "replay": record.get("replay", {}).get("sim_state_equal"),
                "B_replay": record.get("baseline_replay", {}).get("actions_equal"),
                "s": round(record["wall_s"], 1),
            }
            print(json.dumps(brief), flush=True)
    finally:
        env.close()
    summary = summarize(records, cfg["gates"])
    summary.update(identity=identity, split=args.split, seeds=seeds)
    if args.split == "probe":
        write_once(out / "decision.json", summary)
    else:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
