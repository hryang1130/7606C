#!/usr/bin/env python3
"""Takeover-correction fine-tuning study (configs/failure_aware/takeover_correction.toml).

Stages, each per baseline checkpoint and resumable (records are write-once):

    collect   baseline rollouts on collection seeds; at a missed grasp the
              frozen v2 expert takes over in a replayed environment; successful
              expert tails become corrections
    build     arm C = base demos + corrections, arm D = base demos + extra demos
              with as many trainable windows
    finetune  arm C or D from the baseline final.pt with identical settings
    evaluate  arm B (baseline), C or D on the evaluation seeds
    analyze   paired comparison over all checkpoints (no --checkpoint)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dp_manip.failure_round2 import paired_changes, write_once
from dp_manip.metadata import file_sha256

CONFIG = ROOT / "configs/failure_aware/takeover_correction.toml"


def read(path):
    return json.loads(Path(path).read_text())


def load(config_path):
    from dp_manip.takeover import load_probe_config

    cfg = tomllib.loads(Path(config_path).read_text())
    probe_path = ROOT / cfg["source"]["probe_config"]
    if file_sha256(probe_path) != cfg["source"]["probe_config_sha256"]:
        raise ValueError("the frozen v2 probe config changed")
    probe = load_probe_config(probe_path)
    if probe["trigger_kind"] != "grasp_miss":
        raise ValueError("corrections use the v2 grasp-miss takeover")
    lock = tomllib.loads((ROOT / cfg["source"]["lock"]).read_text())
    return cfg, probe, lock


def baseline_path(lock, checkpoint):
    entry = lock["checkpoints"][checkpoint]
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError(f"baseline hash changed: {entry['path']}")
    return Path(entry["path"]), entry["sha256"]


def identity(config_path, extra=None, runner=True):
    """Hashes of the code a stage depends on. Collection excludes this runner so a
    later fix to build/analyze keeps collected records resumable; each record
    still notes the runner hash it was written with."""
    files = {
        "config": config_path,
        "takeover": ROOT / "dp_manip/takeover.py",
        "takeover_data": ROOT / "dp_manip/takeover_data.py",
        "probe_runner": ROOT / "scripts/takeover_probe.py",
    }
    if runner:
        files["runner"] = Path(__file__)
    return {f"{name}_sha256": file_sha256(path) for name, path in files.items()} | (extra or {})


def load_policy(path, device, max_steps):
    import torch

    from dp_manip.config import from_recorded
    from dp_manip.policy import DiffusionPolicy

    payload = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = from_recorded(payload["config"])
    run_cfg.task.max_episode_steps = max_steps
    policy = DiffusionPolicy.from_checkpoint(payload, device).eval()
    return policy, run_cfg


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #


def collect(args, cfg, probe, lock):
    import torch

    from dp_manip.takeover import find_grasp_miss, make_single_env, stack_states
    from dp_manip.takeover_data import correction_end, correction_segment
    from scripts.takeover_probe import first_success, run_expert, run_policy

    path, sha = baseline_path(lock, args.checkpoint)
    out = args.output / args.checkpoint / "collection"
    ident = identity(args.config, {"checkpoint_sha256": sha}, runner=False)
    write_once(out / "identity.json", ident)
    max_steps = cfg["source"]["max_episode_steps"]
    device = torch.device("cuda")
    policy, run_cfg = load_policy(path, device, max_steps)
    env = make_single_env(run_cfg, "cpu")
    target = cfg["collection"]["target_corrections"]
    records = []
    try:
        for seed in range(*cfg["collection"]["seeds"]):
            if sum("correction" in r for r in records) >= target:
                break
            record_path = out / "records" / f"seed{seed}.json"
            if record_path.exists():
                record = read(record_path)
                if record["identity"] != ident:
                    raise ValueError(f"record from another identity: {record_path}")
                records.append(record)
                continue
            tick = time.time()
            gen_seed = cfg["source"]["inference_seed"] + seed
            source, _ = run_policy(env, policy, device, seed, gen_seed, max_steps)
            trigger = find_grasp_miss(stack_states(source.states), probe["trigger"])
            record = {"seed": seed, "identity": ident, "runner_sha256": file_sha256(Path(__file__)),
                      "trigger": trigger,
                      "baseline_success": bool(source.success.any())}
            tau = trigger["tau"]
            if tau is not None:
                branch, expert = run_expert(env, seed, source.actions[:tau], probe, max_steps)
                aligned = (
                    np.array_equal(branch.states[tau]["sim_state"], source.states[tau]["sim_state"])
                    and all(np.array_equal(a, b) for a, b in zip(branch.rgb[: tau + 1], source.rgb[: tau + 1]))
                    and all(np.array_equal(a, b) for a, b in zip(branch.proprio[: tau + 1], source.proprio[: tau + 1]))
                )
                if not aligned:
                    raise RuntimeError(f"seed {seed}: replayed prefix differs from the baseline rollout")
                hit = first_success(branch.success, tau)
                record["expert"] = {"success": hit is not None, "first_success": hit,
                                    "events": expert.events, "attempts": expert.attempts}
                if hit is not None:
                    act_horizon = probe["trigger"].act_horizon
                    segment = correction_segment(
                        seed, np.stack(branch.rgb), np.stack(branch.proprio), np.stack(branch.actions),
                        branch.success, tau=tau, first_success=hit, act_horizon=act_horizon,
                        max_steps=max_steps,
                    )
                    (out / "corrections").mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / "corrections" / f"seed{seed}.npz", rgb=segment.rgb,
                                        proprio=segment.proprio, actions=segment.actions,
                                        success=segment.success)
                    record["correction"] = {
                        "tau": tau, "end": correction_end(tau, hit, act_horizon, max_steps),
                        "windows": len(segment.actions) - 1,
                        "sha256": file_sha256(out / "corrections" / f"seed{seed}.npz"),
                    }
            record["wall_s"] = time.time() - tick
            records.append(write_once(record_path, record))
            print(json.dumps({"seed": seed, "B": record["baseline_success"], "tau": tau,
                              "E": record.get("expert", {}).get("success"),
                              "n": sum("correction" in r for r in records)}), flush=True)
    finally:
        env.close()
    triggered = [r for r in records if r["trigger"]["tau"] is not None]
    summary = {
        "seeds_run": len(records),
        "baseline_successes": sum(r["baseline_success"] for r in records),
        "triggered": len(triggered),
        "triggered_baseline_failures": sum(not r["baseline_success"] for r in triggered),
        "expert_successes": sum(r["expert"]["success"] for r in triggered),
        "corrections": sum("correction" in r for r in records),
        "correction_windows": sum(r["correction"]["windows"] for r in records if "correction" in r),
        "trigger_reasons": {k: sum(r["trigger"]["reason"] == k for r in records)
                            for k in sorted({r["trigger"]["reason"] for r in records})},
        "reached_target": sum("correction" in r for r in records) >= target,
        "wall_s": sum(r["wall_s"] for r in records),
    }
    if not summary["reached_target"]:
        raise RuntimeError(f"collection range exhausted before {target} corrections: {summary}")
    write_once(out / "summary.json", summary | {"identity": ident})
    print(json.dumps(summary, indent=2), flush=True)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def build(args, cfg, probe, lock):
    import h5py

    from dp_manip.failure_rollout import RolloutEpisode
    from dp_manip.takeover_data import matched_extra_demos, ordered_demo_entries, write_mixed_dataset

    _, sha = baseline_path(lock, args.checkpoint)
    base = args.output / args.checkpoint
    summary = read(base / "collection/summary.json")
    records = sorted((read(p) for p in (base / "collection/records").glob("seed*.json")), key=lambda r: r["seed"])
    corrections = [r for r in records if "correction" in r]
    if len(corrections) != summary["corrections"]:
        raise ValueError("collection records and summary disagree")
    demo_file = Path(cfg["data"]["root"]) / cfg["data"]["train_path"]
    entries = ordered_demo_entries(demo_file.with_suffix(".json"))
    n = cfg["data"]["base_demos"]
    with h5py.File(demo_file, "r") as file:
        lengths = {int(e["episode_id"]): len(file[f"traj_{int(e['episode_id'])}/actions"]) for e in entries}
    windows = sum(r["correction"]["windows"] for r in corrections)
    extra = matched_extra_demos(entries[n:], lengths, windows)
    out = base / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"identity": identity(args.config), "baseline_sha256": sha,
                  "demo_file": str(demo_file), "demo_sidecar_sha256": file_sha256(demo_file.with_suffix(".json"))}

    def episodes():
        for r in corrections:
            path = base / "collection/corrections" / f"seed{r['seed']}.npz"
            if file_sha256(path) != r["correction"]["sha256"]:
                raise ValueError(f"correction changed: {path}")
            arrays = np.load(path)
            yield (RolloutEpisode(r["seed"], arrays["rgb"], arrays["proprio"], arrays["actions"],
                                  arrays["success"], np.zeros(len(arrays["actions"]), np.float32)),
                   {"tau": r["correction"]["tau"], "baseline_success": r["baseline_success"],
                    "source_checkpoint_sha256": sha})

    if not (out / "C.h5").exists():
        write_mixed_dataset(out / "C.h5", demo_file=demo_file, demo_entries=entries[:n],
                            corrections=list(episodes()), provenance={**provenance, "arm": "C"})
    if not (out / "D.h5").exists():
        write_mixed_dataset(out / "D.h5", demo_file=demo_file, demo_entries=entries[:n] + extra,
                            provenance={**provenance, "arm": "D"})
    manifest = {
        "base_demo_seeds": [int(e["episode_seed"]) for e in entries[:n]],
        "correction_seeds": [r["seed"] for r in corrections],
        "correction_windows": windows,
        "extra_demo_seeds": [int(e["episode_seed"]) for e in extra],
        "extra_demo_windows": sum(lengths[int(e["episode_id"])] for e in extra),
        "C_sha256": file_sha256(out / "C.h5"),
        "D_sha256": file_sha256(out / "D.h5"),
        "C_sidecar_sha256": file_sha256(out / "C.json"),
        "D_sidecar_sha256": file_sha256(out / "D.json"),
    }
    write_once(out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("seeds")}, indent=2), flush=True)


# --------------------------------------------------------------------------- #
# finetune
# --------------------------------------------------------------------------- #


def finetune(args, cfg, probe, lock):
    import torch

    from dp_manip import config as config_lib
    from dp_manip.finetune import FinetuneSpec
    from dp_manip.trainer import run_training

    path, _ = baseline_path(lock, args.checkpoint)
    base = args.output / args.checkpoint
    manifest = read(base / "datasets/manifest.json")
    dataset = base / "datasets" / f"{args.arm}.h5"
    if file_sha256(dataset) != manifest[f"{args.arm}_sha256"]:
        raise ValueError(f"dataset changed: {dataset}")
    episodes = len(read(dataset.with_suffix(".json"))["episodes"])
    ft = cfg["finetune"]
    overrides = [
        f"data.root={json.dumps(str(dataset.parent))}",
        f"data.train_path={json.dumps(dataset.name)}",
        f"data.val_path={json.dumps(str(Path(cfg['data']['root']) / cfg['data']['val_path']))}",
        f"data.num_demos={episodes}",
        f"data.val_num_demos={cfg['data']['val_demos']}",
        f"train.lr={ft['lr']!r}",
        f"train.total_iters={ft['steps']}",
        f"train.warmup_steps={ft['warmup_steps']}",
        "train.checkpoint_steps=[]",
        "train.validation_steps=[]",
        *args.overrides,
    ]
    baseline = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = config_lib.from_recorded(baseline["config"], overrides)
    del baseline
    spec = FinetuneSpec(
        init_checkpoint=str(path),
        frozen_modules=tuple(ft["frozen_modules"]),
        lr_schedule="constant_with_warmup",
        train_seed_range=(0, cfg["collection"]["seeds"][1]),
        val_seed_range=(4000, 5000),
        require_rollout_source=False,
    )
    return run_training(run_cfg, output_root=base / "models", run_name=args.arm, finetune=spec)


# --------------------------------------------------------------------------- #
# evaluate / analyze
# --------------------------------------------------------------------------- #


def model_for(args, lock):
    if args.arm == "B":
        return baseline_path(lock, args.checkpoint)[0]
    return args.output / args.checkpoint / "models" / args.arm / "checkpoints" / "final.pt"


def evaluate_arm(args, cfg, probe, lock):
    import torch

    from dp_manip.envs import make_eval_envs
    from dp_manip.evaluate import evaluate

    model = model_for(args, lock)
    seeds = list(range(*cfg["evaluation"]["seeds"]))
    max_steps = cfg["source"]["max_episode_steps"]
    ident = {"model": str(model), "model_sha256": file_sha256(model), "seeds": seeds,
             "horizon": max_steps, "config_sha256": file_sha256(args.config)}
    output = args.output / args.checkpoint / "eval" / f"{args.arm}.json"
    if output.exists():
        if read(output)["identity"] != ident:
            raise ValueError(f"evaluation provenance changed: {output}")
        return
    device = torch.device("cuda")
    policy, run_cfg = load_policy(model, device, max_steps)
    envs = make_eval_envs(run_cfg, run_cfg.eval.num_envs, "cpu")
    try:
        result = evaluate(policy, envs, seeds, device, inference_seed=run_cfg.eval.inference_seed)
    finally:
        envs.close()
    result.update(identity=ident, num_envs=run_cfg.eval.num_envs, inference_seed=run_cfg.eval.inference_seed)
    write_once(output, result)
    print(f"{args.checkpoint} {args.arm}: {sum(e['success_once'] for e in result['episodes'])}/{len(seeds)}",
          flush=True)


def bootstrap(per_checkpoint, samples, seed):
    """Two-level paired bootstrap of the mean success difference."""
    rng = np.random.default_rng(seed)
    diffs = [np.asarray(d, dtype=float) for d in per_checkpoint]
    means = np.empty(samples)
    for i in range(samples):
        picks = rng.integers(0, len(diffs), len(diffs))
        means[i] = np.mean([diffs[k][rng.integers(0, len(diffs[k]), len(diffs[k]))].mean() for k in picks])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze(args, cfg, probe, lock):
    results = {
        ck: {arm: read(args.output / ck / "eval" / f"{arm}.json") for arm in ("B", "C", "D")}
        for ck in cfg["source"]["checkpoints"]
    }
    report: dict = {"success": {ck: {arm: sum(e["success_once"] for e in r["episodes"]) for arm, r in arms.items()}
                          for ck, arms in results.items()}}
    a = cfg["analysis"]
    for left, right in (("C", "B"), ("D", "B"), ("C", "D")):
        per_ck, changes = [], {}
        for ck, arms in results.items():
            x = {e["seed"]: e["success_once"] for e in arms[left]["episodes"]}
            y = {e["seed"]: e["success_once"] for e in arms[right]["episodes"]}
            per_ck.append([float(x[s]) - float(y[s]) for s in sorted(x)])
            changes[ck] = paired_changes(arms[left], arms[right])
        report[f"{left}-{right}"] = {
            "difference": float(np.mean([np.mean(d) for d in per_ck])),
            "ci95": bootstrap(per_ck, a["bootstrap_samples"], a["bootstrap_seed"]),
            "only_left_succeeds": sum(len(c["gained"]) for c in changes.values()),
            "only_right_succeeds": sum(len(c["lost"]) for c in changes.values()),
        }
    report["corrections_help"] = report["C-B"]["ci95"][0] > 0
    report["beyond_extra_demos"] = report["corrections_help"] and report["C-D"]["ci95"][0] > 0
    report["identity"] = identity(args.config)
    write_once(args.output / "decision.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("collect", "build", "finetune", "evaluate", "analyze"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("s1", "s2", "s3"))
    parser.add_argument("--arm", choices=("B", "C", "D"))
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="machine-only trainer overrides such as train.num_workers")
    args = parser.parse_args()
    if args.stage != "analyze" and args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.stage == "finetune" and args.arm not in ("C", "D"):
        parser.error("finetune needs --arm C or D")
    if args.stage == "evaluate" and args.arm is None:
        parser.error("evaluate needs --arm")
    cfg, probe, lock = load(args.config)
    stage = {"collect": collect, "build": build, "finetune": finetune,
             "evaluate": evaluate_arm, "analyze": analyze}[args.stage]
    result = stage(args, cfg, probe, lock)
    raise SystemExit(result or 0)


if __name__ == "__main__":
    main()
