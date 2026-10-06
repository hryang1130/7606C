#!/usr/bin/env python3
"""Mechanism check and exchange-rate curve (configs/failure_aware/takeover_curve.toml).

Stages (per checkpoint unless noted; outputs are write-once and resumable):

    mechanism  re-evaluate an existing arm (B, C or D) with proprio-only grasp
               metrics; its episodes must equal the recorded evaluation
    build      datasets C<k> (first k corrections) and D<n> (next n demos)
    finetune   one curve arm, with the correction study's fine-tuning settings
    evaluate   one curve arm on the correction study's evaluation seeds
    analyze    (no --checkpoint) curve and mechanism comparisons against B
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from dp_manip.failure_round2 import write_once
from dp_manip.metadata import file_sha256
from dp_manip.takeover_mechanism import MechanismParams, episode_metrics, summarize
from scripts.takeover_correction import baseline_path, bootstrap, load, load_policy, read

CONFIG = ROOT / "configs/failure_aware/takeover_curve.toml"


def setup(config_path):
    curve = tomllib.loads(Path(config_path).read_text())
    corr_path = ROOT / curve["source"]["correction_config"]
    if file_sha256(corr_path) != curve["source"]["correction_config_sha256"]:
        raise ValueError("the correction study config changed")
    cfg, probe, lock = load(corr_path)
    params = MechanismParams(**curve["mechanism"])
    return curve, cfg, lock, params


def arms(curve):
    return [f"C{k}" for k in curve["curve"]["corrections"]] + [f"D{n}" for n in curve["curve"]["extra_demos"]]


def identity(args):
    files = {"config": args.config, "runner": Path(__file__),
             "mechanism": ROOT / "dp_manip/takeover_mechanism.py",
             "takeover_data": ROOT / "dp_manip/takeover_data.py"}
    return {f"{k}_sha256": file_sha256(v) for k, v in files.items()}


def evaluate_with_metrics(model, cfg, params):
    """``evaluate`` on the correction study's seeds plus per-episode grasp metrics."""
    import torch

    from dp_manip.envs import make_eval_envs
    from dp_manip.evaluate import evaluate
    from dp_manip.failure_rollout import RolloutRecorder

    seeds = list(range(*cfg["evaluation"]["seeds"]))
    device = torch.device("cuda")
    policy, run_cfg = load_policy(model, device, cfg["source"]["max_episode_steps"])
    metrics = {}

    def sink(episode):
        metrics[episode.seed] = episode_metrics(episode.proprio, episode.success, params)

    envs = make_eval_envs(run_cfg, run_cfg.eval.num_envs, "cpu")
    try:
        result = evaluate(policy, envs, seeds, device, inference_seed=run_cfg.eval.inference_seed,
                          observer=RolloutRecorder(sink))
    finally:
        envs.close()
    for episode in result["episodes"]:
        if metrics[episode["seed"]]["success"] != episode["success_once"]:
            raise RuntimeError(f"seed {episode['seed']}: metrics disagree with evaluate")
    return result, metrics, seeds


def mechanism(args, curve, cfg, lock, params):
    corr = Path(curve["source"]["correction_output"]) / args.checkpoint
    model = (baseline_path(lock, args.checkpoint)[0] if args.arm == "B"
             else corr / "models" / args.arm / "checkpoints" / "final.pt")
    recorded = read(corr / "eval" / f"{args.arm}.json")
    if recorded["identity"]["model_sha256"] != file_sha256(model):
        raise ValueError(f"model differs from the evaluated one: {model}")
    output = args.output / args.checkpoint / "mechanism" / f"{args.arm}.json"
    if output.exists():
        return
    result, metrics, _ = evaluate_with_metrics(model, cfg, params)
    if result["episodes"] != recorded["episodes"]:
        raise RuntimeError(f"{args.checkpoint} {args.arm}: re-evaluation does not reproduce the recorded episodes")
    write_once(output, {"identity": identity(args), "model_sha256": recorded["identity"]["model_sha256"],
                        "summary": summarize(list(metrics.values())),
                        "episodes": {str(s): m for s, m in sorted(metrics.items())}})
    print(f"{args.checkpoint} {args.arm} mechanism: {json.dumps(summarize(list(metrics.values())))}", flush=True)


def build(args, curve, cfg, lock, params):
    import h5py

    from dp_manip.failure_rollout import RolloutEpisode
    from dp_manip.takeover_data import ordered_demo_entries, write_mixed_dataset

    _, sha = baseline_path(lock, args.checkpoint)
    corr = Path(curve["source"]["correction_output"]) / args.checkpoint
    manifest_c = read(corr / "datasets/manifest.json")
    records = {r["seed"]: r for r in (read(p) for p in (corr / "collection/records").glob("seed*.json"))}
    corrections = [records[s] for s in manifest_c["correction_seeds"]]  # seed order, as in arm C
    demo_file = Path(cfg["data"]["root"]) / cfg["data"]["train_path"]
    entries = ordered_demo_entries(demo_file.with_suffix(".json"))
    n = cfg["data"]["base_demos"]
    if [int(e["episode_seed"]) for e in entries[:n]] != manifest_c["base_demo_seeds"]:
        raise ValueError("base demonstrations differ from the correction study")
    with h5py.File(demo_file, "r") as file:
        lengths = {int(e["episode_id"]): len(file[f"traj_{int(e['episode_id'])}/actions"]) for e in entries}
    out = args.output / args.checkpoint / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"identity": identity(args), "baseline_sha256": sha, "demo_file": str(demo_file)}
    manifest = {}
    for k in curve["curve"]["corrections"]:
        chosen = corrections[:k]
        if len(chosen) != k:
            raise ValueError(f"only {len(chosen)} corrections")
        path = out / f"C{k}.h5"
        if not path.exists():
            episodes = []
            for r in chosen:
                arrays = np.load(corr / "collection/corrections" / f"seed{r['seed']}.npz")
                if file_sha256(corr / "collection/corrections" / f"seed{r['seed']}.npz") != r["correction"]["sha256"]:
                    raise ValueError(f"correction changed: {r['seed']}")
                episodes.append((RolloutEpisode(r["seed"], arrays["rgb"], arrays["proprio"], arrays["actions"],
                                                arrays["success"], np.zeros(len(arrays["actions"]), np.float32)),
                                 {"tau": r["correction"]["tau"], "baseline_success": r["baseline_success"],
                                  "source_checkpoint_sha256": sha}))
            write_mixed_dataset(path, demo_file=demo_file, demo_entries=entries[:n], corrections=episodes,
                                provenance={**provenance, "arm": f"C{k}"})
        manifest[f"C{k}"] = {"added_windows": sum(r["correction"]["windows"] for r in chosen),
                             "correction_seeds": [r["seed"] for r in chosen], "sha256": file_sha256(path)}
    for m in curve["curve"]["extra_demos"]:
        extra = entries[n:n + m]
        if len(extra) != m:
            raise ValueError(f"only {len(extra)} extra demonstrations")
        path = out / f"D{m}.h5"
        if not path.exists():
            write_mixed_dataset(path, demo_file=demo_file, demo_entries=entries[:n] + extra,
                                provenance={**provenance, "arm": f"D{m}"})
        manifest[f"D{m}"] = {"added_windows": sum(lengths[int(e["episode_id"])] for e in extra),
                             "extra_demo_seeds": [int(e["episode_seed"]) for e in extra], "sha256": file_sha256(path)}
    write_once(out / "manifest.json", manifest)
    print(json.dumps({k: v["added_windows"] for k, v in manifest.items()}), flush=True)


def finetune(args, curve, cfg, lock, params):
    import torch

    from dp_manip import config as config_lib
    from dp_manip.finetune import FinetuneSpec
    from dp_manip.trainer import run_training

    path, _ = baseline_path(lock, args.checkpoint)
    base = args.output / args.checkpoint
    dataset = base / "datasets" / f"{args.arm}.h5"
    if file_sha256(dataset) != read(base / "datasets/manifest.json")[args.arm]["sha256"]:
        raise ValueError(f"dataset changed: {dataset}")
    ft = cfg["finetune"]
    overrides = [
        f"data.root={json.dumps(str(dataset.parent))}",
        f"data.train_path={json.dumps(dataset.name)}",
        f"data.val_path={json.dumps(str(Path(cfg['data']['root']) / cfg['data']['val_path']))}",
        f"data.num_demos={len(read(dataset.with_suffix('.json'))['episodes'])}",
        f"data.val_num_demos={cfg['data']['val_demos']}",
        f"train.lr={ft['lr']!r}",
        f"train.total_iters={ft['steps']}",
        f"train.warmup_steps={ft['warmup_steps']}",
        "train.checkpoint_steps=[]",
        "train.validation_steps=[]",
    ]
    baseline = torch.load(path, map_location="cpu", weights_only=False)
    run_cfg = config_lib.from_recorded(baseline["config"], overrides)
    del baseline
    spec = FinetuneSpec(
        init_checkpoint=str(path), frozen_modules=tuple(ft["frozen_modules"]),
        lr_schedule="constant_with_warmup", train_seed_range=(0, cfg["collection"]["seeds"][1]),
        val_seed_range=(4000, 5000), require_rollout_source=False,
    )
    status = run_training(run_cfg, output_root=base / "models", run_name=args.arm, finetune=spec)
    checkpoints = base / "models" / args.arm / "checkpoints"
    if status == 0 and curve["curve"]["drop_resume"] and (checkpoints / "final.pt").is_file():
        (checkpoints / "resume.pt").unlink(missing_ok=True)
    return status


def evaluate_arm(args, curve, cfg, lock, params):
    model = args.output / args.checkpoint / "models" / args.arm / "checkpoints" / "final.pt"
    output = args.output / args.checkpoint / "eval" / f"{args.arm}.json"
    ident = {"model": str(model), "model_sha256": file_sha256(model),
             "seeds": list(range(*cfg["evaluation"]["seeds"])), "horizon": cfg["source"]["max_episode_steps"],
             "config_sha256": file_sha256(args.config)}
    if output.exists():
        if read(output)["identity"] != ident:
            raise ValueError(f"evaluation provenance changed: {output}")
        return
    result, metrics, _ = evaluate_with_metrics(model, cfg, params)
    result["identity"] = ident
    write_once(output, result)
    write_once(args.output / args.checkpoint / "mechanism" / f"{args.arm}.json",
               {"identity": identity(args), "model_sha256": ident["model_sha256"],
                "summary": summarize(list(metrics.values())),
                "episodes": {str(s): m for s, m in sorted(metrics.items())}})
    print(f"{args.checkpoint} {args.arm}: {sum(e['success_once'] for e in result['episodes'])}/{len(ident['seeds'])}",
          flush=True)


def analyze(args, curve, cfg, lock, params):
    a = curve["analysis"]
    corr = Path(curve["source"]["correction_output"])
    checkpoints = cfg["source"]["checkpoints"]

    def evaluation(ck, arm):
        if arm in ("B", "C", "D"):
            return read(corr / ck / "eval" / f"{arm}.json")
        return read(args.output / ck / "eval" / f"{arm}.json")

    def windows(ck, arm):
        if arm == "B":
            return 0
        if arm in ("C", "D"):
            m = read(corr / ck / "datasets/manifest.json")
            return m["correction_windows"] if arm == "C" else m["extra_demo_windows"]
        return read(args.output / ck / "datasets/manifest.json")[arm]["added_windows"]

    def paired(values):  # {ck: {arm: {seed: value}}} -> difference vs B
        per_ck = [[float(values[ck]["X"][s]) - float(values[ck]["B"][s]) for s in sorted(values[ck]["B"])]
                  for ck in checkpoints]
        return {"difference": float(np.mean([np.mean(d) for d in per_ck])),
                "ci95": bootstrap(per_ck, a["bootstrap_samples"], a["bootstrap_seed"])}

    report = {"identity": identity(args), "curve": {}, "mechanism": {}}
    success = {ck: {arm: {e["seed"]: e["success_once"] for e in evaluation(ck, arm)["episodes"]}
                    for arm in ["B", "C", "D", *arms(curve)]} for ck in checkpoints}
    for arm in ["C", "D", *arms(curve)]:
        entry = {"windows": {ck: windows(ck, arm) for ck in checkpoints},
                 "success": {ck: sum(success[ck][arm].values()) for ck in checkpoints}}
        entry["vs_B"] = paired({ck: {"B": success[ck]["B"], "X": success[ck][arm]} for ck in checkpoints})
        report["curve"]["C100" if arm == "C" else ("D63" if arm == "D" else arm)] = entry
    report["curve"]["B"] = {"windows": {ck: 0 for ck in checkpoints},
                            "success": {ck: sum(success[ck]["B"].values()) for ck in checkpoints}}

    def mech(ck, arm):
        return read(args.output / ck / "mechanism" / f"{arm}.json")

    for arm in ["B", "C", "D", *arms(curve)]:
        rows = {ck: mech(ck, arm) for ck in checkpoints}
        entry = {"summary": {ck: rows[ck]["summary"] for ck in checkpoints}}
        if arm != "B":
            miss_fail = {ck: {"B": {int(s): m["miss_failure"] for s, m in mech(ck, "B")["episodes"].items()},
                              "X": {int(s): m["miss_failure"] for s, m in rows[ck]["episodes"].items()}}
                         for ck in checkpoints}
            entry["miss_failure_vs_B"] = paired(miss_fail)
        report["mechanism"]["C100" if arm == "C" else ("D63" if arm == "D" else arm)] = entry
    d300 = report["curve"].get("D300", {}).get("vs_B")
    report["Q1_extra_demos_help"] = bool(d300 and d300["ci95"][0] > 0)
    report["M1_corrections_cut_miss_failures"] = report["mechanism"]["C100"]["miss_failure_vs_B"]["ci95"][1] < 0
    write_once(args.output / "curve_decision.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "identity"}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("mechanism", "build", "finetune", "evaluate", "analyze"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("s1", "s2", "s3"))
    parser.add_argument("--arm")
    args = parser.parse_args()
    curve, cfg, lock, params = setup(args.config)
    if args.stage != "analyze" and args.checkpoint is None:
        parser.error("--checkpoint is required")
    valid = {"mechanism": ("B", "C", "D"), "finetune": tuple(arms(curve)), "evaluate": tuple(arms(curve))}
    if args.stage in valid and args.arm not in valid[args.stage]:
        parser.error(f"{args.stage} needs --arm in {valid[args.stage]}")
    stage = {"mechanism": mechanism, "build": build, "finetune": finetune,
             "evaluate": evaluate_arm, "analyze": analyze}[args.stage]
    raise SystemExit(stage(args, curve, cfg, lock, params) or 0)


if __name__ == "__main__":
    main()
