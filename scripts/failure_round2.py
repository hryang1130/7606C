#!/usr/bin/env python3
"""Audit, replay and explore round 2 without modifying the first study.

Explicit validation seeds are part of the immutable round2 manifest, rather
than an override to the original study's test seeds. No formal test is run here.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.failure_round2 import choose_candidate, paired_changes, write_once
from dp_manip.metadata import file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint(lock, seed):
    entry = lock["checkpoints"][f"s{seed}"]
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"baseline hash changed: {path}")
    return path


def audit(args, lock):
    import h5py
    import numpy as np

    reports, all_pairs = {}, []
    for seed in (1, 2, 3):
        base = checkpoint(lock, seed)
        old = args.source / f"s{seed}"
        raw = old / "raw_train.h5"
        meta = read(raw.with_suffix(".json"))
        prov = meta["rollout_provenance"]
        if (prov["source_checkpoint"]["sha256"] != file_sha256(base)
                or prov["protocol"]["sha256"] != lock["task"]["protocol_sha256"]
                or prov["max_episode_steps"] != 200):
            raise ValueError(f"raw provenance mismatch: {raw}")
        entries = meta["episodes"]
        ids = [int(e["episode_seed"]) for e in entries]
        if len(ids) != len(set(ids)) or not all(20000 <= s < 21000 for s in ids):
            raise ValueError("raw train seed collision or invalid range")
        with h5py.File(raw) as file:
            for e in entries:
                group = file[f"traj_{e['episode_id']}"]
                n = len(group["actions"])
                success = np.asarray(group["rollout_success"], dtype=bool)
                if (n != e["rollout_length"] or len(success) != n
                        or len(group["obs_rgb/rgb"]) != n + 1
                        or len(group["obs_rgb/state"]) != n + 1
                        or bool(success.any()) != e["success_once"]
                        or bool(success[-1]) != e["success_at_end"]):
                    raise ValueError(f"raw episode inconsistent: {raw}, {e['episode_id']}")
        b = read(next((old / "eval/test").glob("*_B.json")))
        f = read(next((old / "eval/test").glob("*_F_a0.5.json")))
        change = paired_changes(f, b)
        common = sorted(e["seed"] for e in b["episodes"]
                        if not e["success_once"] and not next(
                            x["success_once"] for x in f["episodes"] if x["seed"] == e["seed"]))[:4]
        for direction, seeds in (("gained", change["gained"]), ("lost", change["lost"]),
                                 ("common_failure", common)):
            for episode_seed in seeds:
                all_pairs.append({"checkpoint_seed": seed, "episode_seed": episode_seed,
                                  "category": direction, "annotation": "pending"})
        reports[f"s{seed}"] = {"raw_sha256": file_sha256(raw),
            "raw_json_sha256": file_sha256(raw.with_suffix(".json")),
            "raw_export_sha256": file_sha256(raw.with_suffix(".export_info.json")),
            "episodes": len(entries), "successes": sum(e["success_once"] for e in entries),
            "failures": sum(not e["success_once"] for e in entries), "paired": change}
    write_once(args.output / "audit.json", reports)
    write_once(args.output / "diagnostic_pairs.json", {"pairs": all_pairs})
    print(json.dumps(reports, indent=2), flush=True)


def load_base(lock, seed, device):
    import torch
    from dp_manip.config import from_recorded
    from dp_manip.policy import DiffusionPolicy

    payload = torch.load(checkpoint(lock, seed), map_location="cpu", weights_only=False)
    cfg = from_recorded(payload["config"])
    cfg.task.max_episode_steps = 200
    return DiffusionPolicy.from_checkpoint(payload, device).eval(), cfg


def write_episode_video(target, rgb):
    """Write a replay video and contact sheet into a possibly new directory."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(target, rgb[..., :3], fps=20, codec="libx264")
    frames = np.linspace(0, len(rgb) - 1, 12, dtype=int)
    height, width = rgb.shape[1:3]
    sheet = Image.new("RGB", (4 * width, 3 * (height + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, step in enumerate(frames):
        x, y = (index % 4) * width, (index // 4) * (height + 20)
        sheet.paste(Image.fromarray(rgb[step, ..., :3]), (x, y + 20))
        draw.text((x + 2, y + 2), f"step {step}", fill="black")
    sheet.save(target.with_suffix(".png"))


def run_eval(lock, seed, negative, alpha, seeds, output, videos=None):
    import torch
    from dp_manip.envs import make_eval_envs
    from dp_manip.evaluate import evaluate
    from dp_manip.failure_guidance import FailureGuidedPolicy, GuidanceDiagnostics
    from dp_manip.failure_rollout import RolloutRecorder
    from dp_manip.policy import DiffusionPolicy

    source_hash = file_sha256(negative) if negative else None
    identity = {"baseline_sha256": lock["checkpoints"][f"s{seed}"]["sha256"],
                "negative_sha256": source_hash, "alpha": alpha, "seeds": seeds,
                "horizon": 200}
    if output.exists():
        result = read(output)
        if result["identity"] != identity:
            raise ValueError(f"evaluation provenance changed: {output}")
        return result
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    base, cfg = load_base(lock, seed, device)
    diagnostics = GuidanceDiagnostics() if negative else None
    if negative:
        payload = torch.load(negative, map_location="cpu", weights_only=False)
        model = DiffusionPolicy.from_checkpoint(payload, device).eval()
        del payload
        policy = FailureGuidedPolicy(base, model, mode="fixed", alpha=alpha, diagnostics=diagnostics)
    else:
        policy = base
    observer = None
    if videos:
        def sink(episode):
            if episode.seed in videos:
                target = output.parent / f"{output.stem}_seed{episode.seed}.mp4"
                write_episode_video(target, episode.rgb)

        observer = RolloutRecorder(sink)
    envs = make_eval_envs(cfg, cfg.eval.num_envs, "cpu")
    try:
        result = evaluate(policy, envs, seeds, device,
                          inference_seed=cfg.eval.inference_seed, observer=observer)
    finally:
        envs.close()
    result.update(identity=identity, diagnostics=diagnostics.summary() if diagnostics else None,
                  inference_seed=cfg.eval.inference_seed, num_envs=cfg.eval.num_envs)
    write_once(output, result)
    print(f"{output.name}: {sum(e['success_once'] for e in result['episodes'])}/{len(seeds)}", flush=True)
    del policy, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def diagnose(args, lock):
    audit(args, lock)
    model = Path(lock["models"]["s1"]["failure"])
    base = run_eval(lock, 1, None, 0.0, list(range(33000, 33008)), args.output / "smoke/B.json")
    zero = run_eval(lock, 1, model, 0.0, list(range(33000, 33008)), args.output / "smoke/F_alpha0.json")
    if base["episodes"] != zero["episodes"]:
        raise ValueError("alpha=0 fails to reproduce baseline on new seeds")
    pairs = read(args.output / "diagnostic_pairs.json")["pairs"]
    for seed in (1, 2, 3):
        selected = {e["episode_seed"] for e in pairs if e["checkpoint_seed"] == seed}
        old = args.source / f"s{seed}"
        negative = Path(lock["models"][f"s{seed}"]["failure"])
        if file_sha256(negative) != lock["models"][f"s{seed}"]["failure_sha256"]:
            raise ValueError("first-round failure model hash changed")
        for arm, model in (("B", None), ("F", negative)):
            expected = read(next((old / "eval/test").glob(f"*_{arm}*.json")))
            seeds = [e["seed"] for e in expected["episodes"]]
            result = run_eval(lock, seed, model, 0.5 if model else 0.0, seeds,
                              args.output / "diagnostics" / f"s{seed}_{arm}.json", selected)
            if result["episodes"] != expected["episodes"]:
                raise ValueError(f"diagnostic replay does not reproduce s{seed} {arm}")
    write_once(args.output / "diagnostics_reproduced.json", {"reproduced": True})


def build_subset(args, lock, k, protocol):
    import h5py
    from dp_manip.failure_rollout import RolloutWriter, _load_episode, _stored_length

    output = args.output / f"k{k}" / "datasets"
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / f"k{k}" / "dataset_manifest.json"
    sources = {"train": args.source / "s1/raw_train.h5",
               "holdout": args.output / "holdout/raw_holdout.h5"}
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    if manifest.exists():
        recorded = read(manifest)
        if recorded["source_hashes"] != source_hashes:
            raise ValueError("subset sources changed")
        for name, digest in recorded["dataset_hashes"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"dataset changed: {name}")
        return
    subsets = {}
    for split, source in sources.items():
        meta = read(source.with_suffix(".json"))
        prov = meta["rollout_provenance"]
        expected_protocol = lock["task"]["protocol_sha256"] if split == "train" else protocol.sha256
        if (prov["source_checkpoint"]["sha256"] != lock["checkpoints"]["s1"]["sha256"]
                or prov["protocol"]["sha256"] != expected_protocol
                or prov["max_episode_steps"] != 200):
            raise ValueError("subset source provenance mismatch")
        entries = sorted(meta["episodes"], key=lambda e: e["episode_seed"])
        export = read(source.with_suffix(".export_info.json"))
        for label in ("failure", "success"):
            matching = [e for e in entries if bool(e["success_once"]) == (label == "success")]
            if split == "train":
                size = k if label == "failure" else 150
                if len(matching) < size:
                    raise ValueError(f"not enough {label} episodes for K={size}")
                groups = {f"{label}_train": matching[:size]}
            else:
                groups = {f"{label}_pilot": [e for e in matching if e["episode_seed"] < protocol.pilot_holdout_end()],
                          f"{label}_gate": [e for e in matching if e["episode_seed"] >= protocol.pilot_holdout_end()]}
            for name, selected in groups.items():
                if not selected:
                    raise ValueError(f"empty holdout class: {name}")
                writer = RolloutWriter(output / f"{name}.h5", env_info=meta["env_info"],
                    export_info={**export, "dataset_type": f"rollout_{label}"}, overwrite=True)
                try:
                    with h5py.File(source) as file:
                        for e in selected:
                            writer.add(_load_episode(file, e),
                                stored_length=_stored_length(e, int(prov["act_horizon"]), 184),
                                extra={"source_episode_id": e["episode_id"]})
                    writer.close({**prov, "round2_protocol_sha256": protocol.sha256,
                        "source_raw": str(source), "source_raw_sha256": source_hashes[split],
                        "dataset_type": f"rollout_{label}", "l_fail": 184,
                        "K_fail": k, "K_success": 150})
                except BaseException:
                    writer.abort()
                    raise
                subsets[name] = [e["episode_seed"] for e in selected]
    write_once(manifest, {"source_hashes": source_hashes, "subsets": subsets,
                         "dataset_hashes": {p.name: file_sha256(p) for p in output.iterdir()}})


def explore(args, lock):
    import torch
    import collect_rollouts
    import finetune_dp
    from dp_manip.data import read_dataset_info
    from dp_manip.failure_protocol import load_protocol
    from dp_manip.failure_study import offline_gaps
    from dp_manip.policy import DiffusionPolicy

    started = time.monotonic()
    if not (args.output / "diagnostics_reproduced.json").exists():
        raise ValueError("complete and review the diagnostic replay before exploration")
    protocol_path = ROOT / "configs/failure_aware/round2_explore.toml"
    protocol = load_protocol(protocol_path)
    base_path = checkpoint(lock, 1)
    screen_seeds, validation_seeds = list(range(33000, 33048)), list(range(34000, 34128))
    write_once(args.output / "explore_manifest.json", {
        "baseline": lock["checkpoints"]["s1"], "protocol_sha256": protocol.sha256,
        "code": {str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__), ROOT / "dp_manip/failure_round2.py")},
        "K_fail": [150, 300], "K_success": 150, "steps": [5000, 10000],
        "alpha": [0.25, 0.5, 1.0], "screen_seeds": screen_seeds,
        "validation_seeds": validation_seeds, "advance_min_net_successes": 7,
        "budget_gpu_hours": 8, "horizon": 200, "l_fail": 184})

    def budget():
        # Slurm bounds the job as well; include time from prior attempts.
        if (time.monotonic() - started) / 3600 >= 7.5:
            raise RuntimeError("exploration budget reached; resume only after cost accounting")

    holdout = args.output / "holdout"
    if not (holdout / "raw_holdout.json").exists():
        collect_rollouts.main(["collect", str(base_path), "--split", "holdout", "--protocol",
            str(protocol_path), "--output-dir", str(holdout), "--max-episode-steps", "200", "--render-backend", "cpu"])
    for k in (150, 300):
        budget()
        build_subset(args, lock, k, protocol)
    small = read(args.output / "k150/dataset_manifest.json")["subsets"]["failure_train"]
    large = read(args.output / "k300/dataset_manifest.json")["subsets"]["failure_train"]
    if small != large[:150]:
        raise ValueError("failure subsets are not nested")

    trained = {}
    for k in (150, 300):
        budget()
        directory = args.output / f"k{k}"
        final = directory / "failure_model_lr1e-05_it10000/checkpoints/final.pt"
        if not final.exists():
            status = finetune_dp.main([str(base_path), "--label", "failure", "--lr", "1e-5",
                "--steps", "10000", "--checkpoint-steps", "5000", "--protocol", str(protocol_path),
                "--rollout-dir", str(directory), "--set", "train.num_workers=3"])
            if status:
                raise SystemExit(status)
        for steps in (5000, 10000):
            model = final if steps == 10000 else final.with_name("step_005000.pt")
            if not model.exists():
                raise FileNotFoundError(model)
            trained[k, steps] = model
    write_once(args.output / "trained_models.json", {
        f"k{k}_steps{steps}": {"path": str(p), "sha256": file_sha256(p)}
        for (k, steps), p in trained.items()})

    gates = {}
    for (k, steps), model in trained.items():
        budget()
        path = args.output / "gates" / f"k{k}_steps{steps}.json"
        if not path.exists():
            device = torch.device("cuda")
            base, cfg = load_base(lock, 1, device)
            payload = torch.load(model, map_location="cpu", weights_only=False)
            negative = DiffusionPolicy.from_checkpoint(payload, device).eval()
            directory = args.output / "k150/datasets"
            failures = read_dataset_info(directory / "failure_gate.h5")
            successes = read_dataset_info(directory / "success_gate.h5")
            values = offline_gaps(base, negative, failures, successes, device)
            values["passed"] = values["gap_fail"] > 0 and values["margin"] > 0
            write_once(path, values)
            del base, negative, payload
            gc.collect()
            torch.cuda.empty_cache()
        gates[k, steps] = read(path)
    # Screen all four candidates; gate is required for advancement, not tuning.
    run_eval(lock, 1, None, 0.0, screen_seeds, args.output / "screen/B.json")
    rows = []
    for (k, steps), model in trained.items():
        for alpha in (0.25, 0.5, 1.0):
            budget()
            result = run_eval(lock, 1, model, alpha, screen_seeds,
                             args.output / "screen" / f"F_k{k}_steps{steps}_a{alpha}.json")
            rows.append({"k": k, "steps": steps, "alpha": alpha,
                         "successes": sum(e["success_once"] for e in result["episodes"])})
    selection = {str(k): choose_candidate([r for r in rows if r["k"] == k]) for k in (150, 300)}
    write_once(args.output / "selection.json", {"rows": rows, "selected": selection})
    baseline = run_eval(lock, 1, None, 0.0, validation_seeds, args.output / "validation/B.json")
    validation = {}
    for k in (150, 300):
        budget()
        selected = selection[str(k)]
        result = run_eval(lock, 1, trained[k, selected["steps"]], selected["alpha"],
                          validation_seeds, args.output / "validation" / f"F_k{k}.json")
        validation[str(k)] = {"successes": sum(e["success_once"] for e in result["episodes"]),
                             "paired_vs_B": paired_changes(result, baseline),
                             "gate": gates[k, selected["steps"]]}
    net = validation["300"]["paired_vs_B"]
    advance = (len(net["gained"]) - len(net["lost"]) >= 7
               and validation["300"]["successes"] > validation["150"]["successes"]
               and validation["300"]["gate"]["passed"])
    write_once(args.output / "exploration_decision.json", {
        "advance": advance, "validation": validation,
        "note": "Exploratory decision only; diagnostic annotations must be reviewed before confirmation.",
        "formal_test_started": False})
    print(f"EXPLORATION COMPLETE: advance={advance}; formal test not started", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "diagnose", "explore", "run"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-lock", type=Path, default=ROOT / "configs/failure_aware/placesphere.toml")
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.output == args.source or args.source in args.output.parents:
        raise ValueError("round2 output must be outside the first study directory")
    args.output.mkdir(parents=True, exist_ok=True)
    import tomllib
    lock = tomllib.loads(args.first_lock.read_text())
    if args.stage != "audit":
        inventory = read(args.output / "seed_inventory.json")
        if inventory["conflicts"]:
            raise ValueError("new evaluation seeds overlap prior data")
    if args.stage == "run":
        diagnose(args, lock)
        explore(args, lock)
    else:
        {"audit": audit, "diagnose": diagnose, "explore": explore}[args.stage](args, lock)


if __name__ == "__main__":
    main()
