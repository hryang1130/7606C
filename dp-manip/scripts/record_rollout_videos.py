#!/usr/bin/env python3
"""Record 1080p success and failure rollout videos of one RGB checkpoint.

Rolls the checkpoint out on its held-out seeds exactly like ``eval_dp.py``
(same environments, seed order, ``num_envs`` and inference seed), saving each
step's simulation state, and keeps the first ``--success`` successful and
``--failure`` failed episodes. Each kept episode is then replayed, in its own
freshly spawned process, through a new environment's render camera into an MP4
file, so rendering never touches the environments the policy acts in::

    record_rollout_videos.py <run>/checkpoints/final.pt --split test \
        --reference <run>/eval/test_final.json

The recording reuses a saved ``eval_dp.py`` result for the same checkpoint
and split (its ``num_envs`` and horizon) and checks every recorded episode's
``success_once`` against it, so each video is one of the episodes behind the
reported success rate. A mismatch is reported in ``videos.json`` and makes the
script exit non-zero after writing it. Without ``--reference`` the script uses
``<run>/eval/<split>_<checkpoint>[_h<N>].json`` when exactly one such file
exists, and stops when there are several (for example a 50-step and a 200-step
evaluation); ``--no-reference`` records without the check.

Videos land in ``<run>/videos/<split>_<checkpoint>[_h<N>]/`` by default, named
``<task>_<split>_seed<seed>_<success|failure>.mp4``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import multiprocessing
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dp_manip.config import from_recorded  # noqa: E402
from dp_manip.metadata import git_revision  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--success", type=int, default=3, help="successful episodes to keep (default 3)")
    parser.add_argument("--failure", type=int, default=3, help="failed episodes to keep (default 3)")
    parser.add_argument("--episodes", type=int, help="search at most the split's first N seeds")
    parser.add_argument("--reference", type=Path, help="eval_dp.py result JSON to reproduce and check against")
    parser.add_argument("--no-reference", action="store_true", help="record without checking a saved evaluation")
    parser.add_argument("--num-envs", type=int, help="default: the reference's num_envs, else the config's")
    parser.add_argument("--max-episode-steps", type=int, help="horizon override (default: the reference's)")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, help="default: the task's control frequency (real time)")
    parser.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better (default 16)")
    parser.add_argument(
        "--shader",
        default="default",
        help="render camera shader pack: default, rt-fast or rt (ray traced, slower)",
    )
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="freeze the last frame this long")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing video directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-backend", help="for example 'cpu' to force lavapipe")
    args = parser.parse_args(argv)
    if args.success < 0 or args.failure < 0 or args.success + args.failure == 0:
        parser.error("--success and --failure must be non-negative and not both zero")
    for name in ("width", "height"):
        if getattr(args, name) <= 0 or getattr(args, name) % 2:
            parser.error(f"--{name} must be a positive even number (yuv420p)")
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    if args.hold_seconds < 0:
        parser.error("--hold-seconds must be non-negative")
    if args.reference is not None and args.no_reference:
        parser.error("--reference and --no-reference are mutually exclusive")
    return args


def find_reference(checkpoint: Path, split: str, max_episode_steps: int | None) -> Path:
    """The run's saved evaluation of this checkpoint and split, if it is unambiguous."""
    eval_dir = checkpoint.resolve().parents[1] / "eval"
    if max_episode_steps is not None:
        candidates = [eval_dir / f"{split}_{checkpoint.stem}_h{max_episode_steps}.json"]
    else:
        candidates = [eval_dir / f"{split}_{checkpoint.stem}.json"]
        candidates += sorted(eval_dir.glob(f"{split}_{checkpoint.stem}_h*.json"))
    found = [path for path in candidates if path.is_file()]
    if not found:
        raise FileNotFoundError(
            f"no saved {split} evaluation of {checkpoint.name} in {eval_dir}; "
            "pass --reference, or --no-reference to record without the check"
        )
    if len(found) > 1:
        names = ", ".join(path.name for path in found)
        raise ValueError(f"several saved {split} evaluations ({names}); choose one with --reference")
    return found[0]


def load_reference(path: Path, split: str) -> dict:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if reference.get("split") not in (None, split):
        raise ValueError(f"{path} is a {reference['split']} evaluation, not {split}")
    if "episodes" not in reference:
        raise ValueError(f"{path} has no per-episode results")
    return reference


def resolve_horizon(cli: int | None, reference: dict | None) -> int | None:
    recorded = None if reference is None else reference.get("max_episode_steps")
    if cli is not None and recorded is not None and cli != recorded:
        raise ValueError(f"--max-episode-steps {cli} differs from the reference's horizon {recorded}")
    return cli if cli is not None else recorded


def default_output_dir(checkpoint: Path, split: str, horizon_override: int | None) -> Path:
    suffix = "" if horizon_override is None else f"_h{horizon_override}"
    return checkpoint.resolve().parents[1] / "videos" / f"{split}_{checkpoint.stem}{suffix}"


def main() -> None:
    args = parse_args()
    import torch

    from dp_manip.evaluate import evaluate
    from dp_manip.policy import DiffusionPolicy
    from dp_manip.envs import make_eval_envs
    from dp_manip.rollout_video import StateRecorder, reference_mismatches, render_videos, video_name

    if args.no_reference:
        reference_path = None
    else:
        reference_path = args.reference or find_reference(args.checkpoint, args.split, args.max_episode_steps)
        print(f"reference evaluation: {reference_path}")
    reference = None if reference_path is None else load_reference(reference_path, args.split)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = from_recorded(checkpoint["config"])
    if getattr(cfg.task, "obs_mode", "rgb") != "rgb":
        raise ValueError("rollout videos support RGB checkpoints only")
    if reference is not None:
        for key, ours in (("checkpoint_step", int(checkpoint["step"])), ("inference_seed", cfg.eval.inference_seed)):
            if key in reference and reference[key] != ours:
                raise ValueError(f"{reference_path} has {key}={reference[key]}, this checkpoint {ours}")
    horizon = resolve_horizon(args.max_episode_steps, reference)
    horizon_override = None
    if horizon is not None and horizon != cfg.task.max_episode_steps:
        horizon_override = horizon
        cfg.task.max_episode_steps = horizon
    policy = DiffusionPolicy.from_checkpoint(checkpoint, device)
    policy.eval()

    seeds = cfg.test_seeds() if args.split == "test" else cfg.val_seeds()
    if args.episodes is not None:
        if not 1 <= args.episodes <= len(seeds):
            raise ValueError(f"--episodes must be in [1, {len(seeds)}] for split {args.split}")
        seeds = seeds[: args.episodes]
    requested_envs = args.num_envs or (reference or {}).get("num_envs") or cfg.eval.num_envs
    if reference is not None and requested_envs != reference.get("num_envs", requested_envs):
        print(f"WARNING: num_envs {requested_envs} differs from the reference's; outcomes may not match")
    num_envs = math.gcd(len(seeds), requested_envs)
    if num_envs != requested_envs:
        print(f"adjusting num_envs from {requested_envs} to {num_envs} so all waves are full")

    output_dir = args.output_dir or default_output_dir(args.checkpoint, args.split, horizon_override)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = make_eval_envs(cfg, num_envs, args.render_backend)
    try:
        recorder = StateRecorder(envs, quota={"success": args.success, "failure": args.failure})
        result = evaluate(policy, envs, seeds, device, inference_seed=cfg.eval.inference_seed, observer=recorder)
    finally:
        envs.close()

    jobs = []
    for episode in recorder.episodes:
        episode["video"] = None
        if episode["kept"]:
            episode["video"] = video_name(cfg.task.name, args.split, episode["seed"], episode["outcome"])
            jobs.append((episode["seed"], recorder.kept_states[episode["seed"]], episode["video"]))
    # One fresh interpreter per video, so no render process builds a third scene
    # (see dp_manip.rollout_video).
    fps = args.fps
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context, max_tasks_per_child=1) as pool:
        for job in jobs:
            rendered = pool.submit(
                render_videos,
                cfg,
                [job],
                str(output_dir),
                width=args.width,
                height=args.height,
                shader=args.shader,
                fps=args.fps,
                crf=args.crf,
                hold_seconds=args.hold_seconds,
                render_backend=args.render_backend,
            ).result()
            fps = rendered["fps"]

    mismatches = None if reference is None else reference_mismatches(recorder.episodes, reference["episodes"])
    manifest = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "task": cfg.task.name,
        "env_id": cfg.task.env_id,
        "control_mode": cfg.task.control_mode,
        "split": args.split,
        "max_episode_steps": cfg.task.max_episode_steps,
        "num_envs": num_envs,
        "inference_seed": cfg.eval.inference_seed,
        "label_metric": "success_once",
        "quota": {"success": args.success, "failure": args.failure},
        "kept": recorder.kept,
        "video": {
            "camera": "render_camera (replayed from saved simulation states)",
            "width": args.width,
            "height": args.height,
            "fps": fps,
            "codec": "libx264",
            "crf": args.crf,
            "shader": args.shader,
            "hold_seconds": args.hold_seconds,
        },
        "reference": None if reference_path is None else str(reference_path.resolve()),
        "reference_mismatches": mismatches,
        "episodes_searched": len(recorder.episodes),
        "episodes": recorder.episodes,
        "summary_of_searched_episodes": result["summary"],
        "code": git_revision(ROOT),
    }
    manifest_path = output_dir / "videos.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"{cfg.task.name} {args.split}: kept {recorder.kept['success']}/{args.success} success, "
        f"{recorder.kept['failure']}/{args.failure} failure from {len(recorder.episodes)} episodes; {output_dir}"
    )
    for outcome, wanted in recorder.quota.items():
        if recorder.kept[outcome] < wanted:
            print(f"NOTE: only {recorder.kept[outcome]} {outcome} episodes in the searched seeds")
    if reference_path is None:
        print("NOTE: --no-reference; outcomes were not checked against a saved evaluation")
    elif mismatches:
        raise SystemExit(f"ERROR: {len(mismatches)} episodes differ from {reference_path}; see {manifest_path}")
    else:
        print(f"all {len(recorder.episodes)} recorded outcomes match {reference_path.name}")


if __name__ == "__main__":
    main()
