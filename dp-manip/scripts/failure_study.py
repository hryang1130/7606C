#!/usr/bin/env python3
"""Run the failure-aware study stage by stage (plan §8 Phase 5).

Every stage reads its inputs from and writes its result to the task's lock file
``configs/failure_aware/<task>.toml``; a stage refuses to run before its
prerequisites and a written section is never overwritten. Closed-loop stages
skip results that already exist, so a preempted job can simply be resubmitted.

    select-cell        §1    baseline cell + checkpoints        -> [task], [checkpoints]
    (collect_rollouts.py collect/build per checkpoint)
    record-collection  §3.4  one checkpoint's datasets         -> [collection.sN]
    (finetune_dp.py: pilot runs on checkpoint 1)
    pilot              §4.4  lr/steps from offline margins      -> [finetune]
    (finetune_dp.py: remaining failure/success models)
    record-models      §4    one checkpoint's two models       -> [models.sN]
    gate               §6.7  offline discriminability gate     -> [gate]
    dry-run            §5.5  m, and alpha = 0 == baseline      -> [guidance.dry_run]
    eval --split tuning      F, A (then C1) over the alpha grid
    select-alpha       §5.5  F/A alphas and the guidance arm   -> [guidance.selection]
    select-c1          §5.5  C1's alpha                        -> [guidance.c1]
    eval --split test        B, F, A, C1, C2 with locked alphas
    status                   stages done and GPU-hours spent
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip import failure_study as study  # noqa: E402
from dp_manip.failure_lock import LockFile, lock_path  # noqa: E402
from dp_manip.failure_protocol import DEFAULT_PROTOCOL, FailureProtocol, load_protocol  # noqa: E402
from dp_manip.failure_rollout import RAW_HOLDOUT, RAW_TRAIN, SUMMARY  # noqa: E402
from dp_manip.metadata import file_sha256, git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str, *, torch_device: bool = False, envs: bool = False):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--task", required=True)
        sub.add_argument("--lock", type=Path, help="default: configs/failure_aware/<task>.toml")
        sub.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        if torch_device:
            sub.add_argument("--device", default="cuda")
        if envs:
            sub.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
        return sub

    cell = command("select-cell", "apply the baseline cell rule (plan §1)")
    cell.add_argument("--run-root", type=Path, required=True, help="main-track run root")
    cell.add_argument(
        "--rollout-root",
        type=Path,
        help="root of failure_aware/<task>/s<seed>/ (default: the run root); a smoke run uses a scratch root",
    )
    cell.add_argument(
        "--max-episode-steps",
        type=positive_int,
        help="horizon of every closed-loop stage (plan §1, §13); reads eval/val_final_h<N>.json. "
        "Default: the checkpoints' recorded horizon and eval/val_final.json",
    )
    cell.add_argument(
        "--no-low-success",
        action="store_true",
        help="stop instead of entering low-success mode when no cell is in the band (plan §12.3, replication task)",
    )
    collection = command("record-collection", "lock one checkpoint's built datasets")
    collection.add_argument("--seed", type=int, required=True)
    command("pilot", "choose lr/steps from the offline pilot (plan §4.4)", torch_device=True)
    models = command("record-models", "lock one checkpoint's failure and success models")
    models.add_argument("--seed", type=int, required=True)
    command("gate", "offline discriminability gate (plan §6.7)", torch_device=True)
    command("dry-run", "measure m and check alpha = 0 reproduces the baseline", torch_device=True, envs=True)
    evaluation = command("eval", "closed-loop evaluation of one arm", torch_device=True, envs=True)
    evaluation.add_argument("--split", choices=study.SPLITS, required=True)
    evaluation.add_argument("--arm", choices=study.ARMS, required=True)
    evaluation.add_argument("--seeds", type=int, nargs="*", help="default: every locked checkpoint seed")
    evaluation.add_argument("--grid-values", type=float, nargs="*", help="tuning only; default: the whole grid")
    command("select-alpha", "choose the F and A alphas and the guidance arm (plan §5.5)")
    command("select-c1", "choose C1's alpha (plan §5.5)")
    command("status", "print the locked stages and the GPU-hours spent")
    return parser.parse_args(argv)


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not positive")
    return value


def open_lock(args: argparse.Namespace) -> LockFile:
    return LockFile(args.lock or lock_path(args.task))


def torch_device(name: str):
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# ------------------------------------------------------------------ stages


def select_cell(args, lock: LockFile, protocol: FailureProtocol) -> None:
    cell = protocol.baseline_cell
    horizon = args.max_episode_steps
    n100 = study.read_val_success(args.run_root, args.task, 100, cell.val_seeds, horizon)
    if n100 is None:
        raise FileNotFoundError(
            f"N=100 {study.val_result_name(horizon)} of seeds {cell.val_seeds} are missing under {args.run_root}"
        )
    n200 = study.read_val_success(args.run_root, args.task, 200, cell.val_seeds, horizon)
    choice = study.select_baseline_cell(
        n100, n200, low=cell.min_val_success, high=cell.max_val_success, allow_low_success=not args.no_low_success
    )
    checkpoints = {}
    for seed in cell.checkpoint_seeds:
        path = study.baseline_run_dir(args.run_root, args.task, choice["num_demos"], seed) / "checkpoints" / "final.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[f"s{seed}"] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if horizon is None:
        horizon = recorded_horizon(next(iter(checkpoints.values()))["path"])
    lock.write(
        "task",
        {
            "name": args.task,
            **choice,
            "val_seeds": list(cell.val_seeds),
            "val_success_n100": n100,
            "val_success_n200": n200,
            "max_episode_steps": horizon,
            "val_results": study.val_result_name(args.max_episode_steps),
            "low_success_allowed": not args.no_low_success,
            "protocol_sha256": protocol.sha256,
            "rollout_root": str(args.rollout_root.resolve()) if args.rollout_root else None,
        },
    )
    lock.write("checkpoints", checkpoints)
    print(
        f"{args.task}: N={choice['num_demos']} ({choice['mode']}), mean val success "
        f"{choice['mean_val_success']:.3f}, horizon {horizon}"
    )


def recorded_horizon(checkpoint_path: str) -> int:
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return int(payload["config"]["task"]["max_episode_steps"])


def record_collection(args, lock: LockFile, protocol: FailureProtocol) -> None:
    summary_path = study.dataset_dir(lock, args.seed) / SUMMARY
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = lock.require(f"checkpoints.s{args.seed}", "select-cell")
    if summary["source_checkpoint"]["sha256"] != checkpoint["sha256"]:
        raise ValueError(f"{summary_path} was collected from another checkpoint")
    if summary["protocol"]["sha256"] != protocol.sha256:
        raise ValueError(f"{summary_path} was built under another protocol file")
    horizon = study.study_horizon(lock)
    if horizon is not None and int(summary["max_episode_steps"]) != horizon:
        raise ValueError(
            f"{summary_path} was collected at {summary['max_episode_steps']} steps, the locked horizon is {horizon}; "
            "collect with --max-episode-steps"
        )
    lock.write(
        f"collection.s{args.seed}",
        {
            "train_rollouts": summary["train_rollouts"],
            "train_successes": summary["train_successes"],
            "train_failures": summary["train_failures"],
            "dataset_size": summary["dataset_size"],
            "l_fail": summary["l_fail"],
            "low_success": summary["low_success"],
            "failure_fraction_truncated": summary["datasets"]["failure_train"]["fraction_truncated"],
            "summary_sha256": file_sha256(summary_path),
        },
    )
    print(f"s{args.seed}: K={summary['dataset_size']}, L_fail={summary['l_fail']}, low_success={summary['low_success']}")


def reference_infos(lock: LockFile, seed: int, part: str):
    """Failure and success datasets of the pilot or gate holdout part.

    In low-success mode the success side is the baseline's expert validation
    demonstrations instead of held-out successful rollouts (plan §1).
    """
    import torch

    from dp_manip.data import read_dataset_info

    directory = study.dataset_dir(lock, seed)
    failure = read_dataset_info(directory / f"failure_{part}.h5")
    if study.low_success(lock):
        checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
        payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
        return failure, read_dataset_info(payload["val_data"]["path"]), "expert_validation"
    return failure, read_dataset_info(directory / f"success_{part}.h5"), f"success_{part}"


def pilot(args, lock: LockFile, protocol: FailureProtocol) -> None:
    seed = study.checkpoint_seeds(lock)[0]
    lock.require(f"collection.s{seed}", "record-collection")
    device = torch_device(args.device)
    checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
    base = study.load_policy(checkpoint["path"], checkpoint["sha256"], device)[0]
    failure_info, success_info, success_source = reference_infos(lock, seed, "pilot")
    rollout = study.rollout_dir(lock, seed)
    longest = max(protocol.pilot.steps)
    candidates: list[dict[str, Any]] = []
    for lr in protocol.pilot.learning_rates:
        for steps in protocol.pilot.steps:
            path = study.model_checkpoint(rollout, "failure", lr, longest, steps)
            model, payload = study.load_policy(path, file_sha256(path), device)
            if payload["finetune"]["init_checkpoint_sha256"] != checkpoint["sha256"]:
                raise ValueError(f"{path} was not fine-tuned from checkpoint s{seed}")
            gaps = study.offline_gaps(base, model, failure_info, success_info, device)
            candidates.append({"lr": lr, "steps": steps, "path": str(path), **gaps})
            print(f"lr={lr:g} steps={steps}: margin={gaps['margin']:.5f}")
    chosen = study.select_pilot(candidates, protocol.pilot.relative_tie)
    lock.write(
        "finetune",
        {
            "lr": chosen["lr"],
            "steps": chosen["steps"],
            "source": "pilot",
            "pilot_seed": seed,
            "pilot_checkpoint": chosen["path"],
            "pilot_checkpoint_sha256": file_sha256(chosen["path"]),
            "pilot_success_reference": success_source,
            "candidate_lr": [candidate["lr"] for candidate in candidates],
            "candidate_steps": [candidate["steps"] for candidate in candidates],
            "candidate_margin": [candidate["margin"] for candidate in candidates],
            "candidate_gap_fail": [candidate["gap_fail"] for candidate in candidates],
            "candidate_gap_succ": [candidate["gap_succ"] for candidate in candidates],
        },
    )
    print(f"chosen: lr={chosen['lr']:g}, steps={chosen['steps']} (margin {chosen['margin']:.5f})")


def record_models(args, lock: LockFile, protocol: FailureProtocol) -> None:
    import torch

    finetune = lock.require("finetune", "pilot")
    lock.require(f"collection.s{args.seed}", "record-collection")
    checkpoint = lock.require(f"checkpoints.s{args.seed}", "select-cell")
    rollout = study.rollout_dir(lock, args.seed)
    lr, steps = float(finetune["lr"]), int(finetune["steps"])
    # The pilot's chosen checkpoint is checkpoint 1's failure model (plan §4.4).
    if args.seed == int(finetune["pilot_seed"]):
        failure = Path(finetune["pilot_checkpoint"])
    else:
        failure = study.model_checkpoint(rollout, "failure", lr, steps)
    entry: dict[str, Any] = {}
    labels = {"failure": failure}
    if not study.low_success(lock):
        labels["success"] = study.model_checkpoint(rollout, "success", lr, steps)
    for label, path in labels.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = payload.get("finetune") or {}
        if record.get("init_checkpoint_sha256") != checkpoint["sha256"]:
            raise ValueError(f"{path} was not fine-tuned from checkpoint s{args.seed}")
        if float(payload["config"]["train"]["lr"]) != lr:
            raise ValueError(f"{path} was trained with lr {payload['config']['train']['lr']}, locked lr is {lr}")
        if int(payload["step"]) != steps:
            raise ValueError(f"{path} is at step {payload['step']}, locked steps are {steps}")
        expected = Path(payload["train_data"]["path"]).name
        if expected != f"{label}_train.h5":
            raise ValueError(f"{path} was trained on {expected}, not {label}_train.h5")
        entry[label] = str(Path(path).resolve())
        entry[f"{label}_sha256"] = file_sha256(path)
    lock.write(f"models.s{args.seed}", entry)
    print(f"s{args.seed}: " + ", ".join(f"{label}={entry[label]}" for label in labels))


def gate(args, lock: LockFile, protocol: FailureProtocol) -> None:
    device = torch_device(args.device)
    seeds = study.checkpoint_seeds(lock)
    results: dict[str, Any] = {}
    for seed in seeds:
        checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
        models = lock.require(f"models.s{seed}", "record-models")
        base = study.load_policy(checkpoint["path"], checkpoint["sha256"], device)[0]
        failure_info, success_info, success_source = reference_infos(lock, seed, "gate")
        failure_model = study.load_policy(models["failure"], models["failure_sha256"], device)[0]
        gaps = study.offline_gaps(base, failure_model, failure_info, success_info, device)
        entry = {**gaps, "passed": gaps["gap_fail"] > gaps["gap_succ"], "success_reference": success_source}
        if "success" in models:
            success_model = study.load_policy(models["success"], models["success_sha256"], device)[0]
            mirror = study.offline_gaps(base, success_model, failure_info, success_info, device)
            entry.update({f"success_model_{key}": value for key, value in mirror.items()})
        results[f"s{seed}"] = entry
        print(f"s{seed}: gap_fail={gaps['gap_fail']:.5f} gap_succ={gaps['gap_succ']:.5f} passed={entry['passed']}")
    passed = all(entry["passed"] for entry in results.values())
    lock.write("gate", {"passed": passed, **results})
    print("gate " + ("PASSED" if passed else "FAILED: stop and report the offline losses (plan §6.7)"))


def baseline_config(lock: LockFile, seed: int):
    """The checkpoint's recorded config at the locked horizon, and the recorded horizon."""
    import torch

    from dp_manip.config import from_recorded

    checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=False)
    cfg = from_recorded(payload["config"])
    recorded = cfg.task.max_episode_steps
    horizon = study.study_horizon(lock)
    if horizon is not None:
        cfg.task.max_episode_steps = horizon
    return cfg, recorded


def default_envs_factory(cfg, num_envs: int, render_backend: str | None):
    from dp_manip.envs import make_eval_envs

    return make_eval_envs(cfg, num_envs, render_backend)


def run_policy(policy, cfg, seeds, device, envs_factory, render_backend) -> dict:
    from dp_manip.evaluate import evaluate

    envs = envs_factory(cfg, cfg.eval.num_envs, render_backend)
    try:
        return evaluate(policy, envs, seeds, device, inference_seed=cfg.eval.inference_seed)
    finally:
        envs.close()


def context(lock: LockFile, protocol: FailureProtocol, cfg, recorded_horizon: int, device) -> dict[str, Any]:
    return {
        "lock": {"path": str(lock.path), "sha256": lock.sha256},
        "protocol_sha256": protocol.sha256,
        "git": git_revision(ROOT),
        "num_envs": cfg.eval.num_envs,
        "inference_seed": cfg.eval.inference_seed,
        "max_episode_steps": cfg.task.max_episode_steps,
        "checkpoint_max_episode_steps": recorded_horizon,
        "device": str(device),
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def dry_run(args, lock: LockFile, protocol: FailureProtocol, envs_factory: Callable = default_envs_factory) -> None:
    from dp_manip.failure_guidance import GuidanceDiagnostics

    if not lock.require("gate", "gate").get("passed"):
        raise RuntimeError("the offline gate failed: the study stops here (plan §6.7)")
    device = torch_device(args.device)
    seed = study.checkpoint_seeds(lock)[0]
    cfg, recorded = baseline_config(lock, seed)
    seeds = study.dry_run_seeds(protocol)
    checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
    models = lock.require(f"models.s{seed}", "record-models")
    baseline = study.load_policy(checkpoint["path"], checkpoint["sha256"], device)[0]
    spec = study.ArmSpec(
        task=args.task,
        arm="A",
        split="dry",
        seed=seed,
        name="dry_run_alpha0",
        base_path=checkpoint["path"],
        base_sha256=checkpoint["sha256"],
        negative_path=models["failure"],
        negative_sha256=models["failure_sha256"],
        mode="adaptive",
        grid_value=0.0,
        alpha=0.0,
    )
    diagnostics = GuidanceDiagnostics()
    guided = study.build_arm_policy(spec, device, diagnostics)
    expected = run_policy(baseline, cfg, seeds, device, envs_factory, args.render_backend)
    observed = run_policy(guided, cfg, seeds, device, envs_factory, args.render_backend)
    reproduces = observed["episodes"] == expected["episodes"]
    summary = diagnostics.summary()
    output = study.rollout_dir(lock, seed) / "eval" / "dry"
    write_json(output / "baseline.json", {**expected, **context(lock, protocol, cfg, recorded, device)})
    write_json(
        output / "alpha0_guided.json",
        {**observed, "arm": spec.to_dict(), "diagnostics": summary, **context(lock, protocol, cfg, recorded, device)},
    )
    if not reproduces:
        raise RuntimeError(
            f"alpha = 0 guidance did not reproduce the baseline episodes; see {output}. "
            "Paired comparisons are invalid until this is fixed (plan §8 Phase 3)."
        )
    lock.write(
        "guidance.dry_run",
        {
            "seed": seed,
            "episodes": len(seeds),
            "seed_range": [seeds[0], seeds[-1] + 1],
            "max_episode_steps": cfg.task.max_episode_steps,
            "m": summary["mean_half_one_minus_cos"],
            "mean_cosine": summary["mean"]["cosine"],
            "alpha0_reproduces_baseline": True,
        },
    )
    print(f"m = {summary['mean_half_one_minus_cos']:.6f}; alpha = 0 reproduces the baseline on {len(seeds)} episodes")


def evaluate_arms(args, lock: LockFile, protocol: FailureProtocol, envs_factory: Callable = default_envs_factory) -> None:
    from dp_manip.failure_guidance import GuidanceDiagnostics

    device = torch_device(args.device)
    seeds = args.seeds or study.checkpoint_seeds(lock)
    grid = [None] if args.split == "test" else (args.grid_values or list(protocol.guidance.alpha_grid))
    for seed in seeds:
        cfg, recorded = baseline_config(lock, seed)
        episode_seeds = study.tuning_seeds(protocol) if args.split == "tuning" else protocol.test_seeds(cfg)
        for grid_value in grid:
            spec = study.resolve_arm(lock, protocol, arm=args.arm, split=args.split, seed=seed, grid_value=grid_value)
            output = study.eval_output_path(lock, spec)
            if output.is_file():
                print(f"{spec.name}: {output} exists; skipped")
                continue
            diagnostics = GuidanceDiagnostics() if spec.mode is not None else None
            policy = study.build_arm_policy(spec, device, diagnostics)
            result = run_policy(policy, cfg, episode_seeds, device, envs_factory, args.render_backend)
            write_json(
                output,
                {
                    **result,
                    "arm": spec.to_dict(),
                    "diagnostics": diagnostics.summary() if diagnostics is not None else None,
                    **context(lock, protocol, cfg, recorded, device),
                },
            )
            successes = sum(episode["success_once"] for episode in result["episodes"])
            print(f"{spec.name}: {successes}/{len(episode_seeds)} success_once; {output}")


def select_alpha(args, lock: LockFile, protocol: FailureProtocol) -> None:
    lock.require("guidance.dry_run", "dry-run")
    selections = {}
    for arm in ("F", "A"):
        pooled = study.pooled_tuning_successes(lock, protocol, arm)
        selections[arm] = {**study.select_alpha(pooled, protocol.guidance.tie_episodes), "pooled": pooled}
    guidance_arm = study.select_guidance_arm(selections["F"], selections["A"])
    m = float(lock.require("guidance.dry_run", "dry-run")["m"])
    values: dict[str, Any] = {"guidance_arm": guidance_arm, "grid": list(protocol.guidance.alpha_grid)}
    for arm, selection in selections.items():
        values[f"grid_value_{arm}"] = selection["grid_value"]
        values[f"alpha_{arm}"] = selection["grid_value"] / m if arm == "A" else selection["grid_value"]
        values[f"successes_{arm}"] = selection["successes"]
        values[f"at_grid_edge_{arm}"] = selection["at_grid_edge"]
        values[f"pooled_{arm}"] = [selection["pooled"][value] for value in protocol.guidance.alpha_grid]
    lock.write("guidance.selection", values)
    print(json.dumps(values, indent=2))


def select_c1(args, lock: LockFile, protocol: FailureProtocol) -> None:
    pooled = study.pooled_tuning_successes(lock, protocol, "C1")
    selection = study.select_alpha(pooled, protocol.guidance.tie_episodes)
    spec = study.resolve_arm(
        lock, protocol, arm="C1", split="tuning", seed=study.checkpoint_seeds(lock)[0], grid_value=selection["grid_value"]
    )
    lock.write(
        "guidance.c1",
        {
            **selection,
            "mode": spec.mode,
            "alpha": spec.alpha,
            "pooled": [pooled[value] for value in protocol.guidance.alpha_grid],
        },
    )
    print(f"C1: grid value {selection['grid_value']:g} ({spec.mode}, alpha {spec.alpha:g})")


def status(args, lock: LockFile, protocol: FailureProtocol) -> None:
    stages = [
        "task", "checkpoints", "finetune", "gate", "guidance.dry_run", "guidance.selection", "guidance.c1",
    ]
    for stage in stages:
        print(f"[{stage}] " + ("locked" if lock.has(stage) else "-"))
    if not lock.has("checkpoints"):
        return
    seconds = 0.0
    for seed in study.checkpoint_seeds(lock):
        print(
            f"s{seed}: collection {'locked' if lock.has(f'collection.s{seed}') else '-'}, "
            f"models {'locked' if lock.has(f'models.s{seed}') else '-'}"
        )
        rollout = study.rollout_dir(lock, seed)
        for raw in (RAW_TRAIN, RAW_HOLDOUT):
            sidecar = (rollout / raw).with_suffix(".json")
            if sidecar.is_file():
                seconds += json.loads(sidecar.read_text())["rollout_provenance"].get("wall_time_s", 0.0)
        for result in rollout.glob("eval/*/*.json"):
            seconds += json.loads(result.read_text()).get("summary", {}).get("wall_time_s", 0.0)
        for summary in rollout.glob("*_model_*/summary.json"):
            seconds += json.loads(summary.read_text()).get("wall_time_s_this_invocation", 0.0)
    print(f"GPU-hours recorded (1 GPU per job; last invocation of each fine-tune): {seconds / 3600:.2f} / 24")


STAGES: dict[str, Callable] = {
    "select-cell": select_cell,
    "record-collection": record_collection,
    "pilot": pilot,
    "record-models": record_models,
    "gate": gate,
    "dry-run": dry_run,
    "eval": evaluate_arms,
    "select-alpha": select_alpha,
    "select-c1": select_c1,
    "status": status,
}


def main(argv: list[str] | None = None, envs_factory: Callable = default_envs_factory) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    stage = STAGES[args.command]
    if stage in (dry_run, evaluate_arms):
        stage(args, open_lock(args), protocol, envs_factory)
    else:
        stage(args, open_lock(args), protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
