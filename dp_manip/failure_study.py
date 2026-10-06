"""Stages of the failure-aware study (plan §1, §4.4, §5.5, §6, §8 Phase 4-5).

Pure rules (baseline cell, pilot choice, alpha choice) and the arm resolver
are stdlib-only and read the per-task lock file; the offline losses and the
closed-loop arm evaluation import torch lazily. ``scripts/failure_study.py``
is the command-line front end, one subcommand per stage.

Arms (plan §6.1): ``B`` baseline, ``F``/``A`` fixed/adaptive guidance away from
the failure model, ``C1`` guidance away from the success-rollout model, ``C2``
the success-rollout model used as the policy.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .failure_lock import LockFile
from .failure_protocol import FailureProtocol
from .failure_rollout import DATASET_DIR, rollout_dir_for
from .finetune import model_run_name

ARMS = ("B", "F", "A", "C1", "C2")
GUIDED_ARMS = {"F": "fixed", "A": "adaptive"}
SPLITS = ("tuning", "test")


# --------------------------------------------------------------------- rules


def baseline_run_dir(run_root: Path, task: str, num_demos: int, seed: int) -> Path:
    """Main-track run directory (``config.default_run_name`` for the UNet arm)."""
    return Path(run_root) / f"{task}_rgb_unet_n{num_demos}_s{seed}"


def val_result_name(max_episode_steps: int | None) -> str:
    """``scripts/eval_dp.py``'s validation result name; an overridden horizon adds ``_h<N>``."""
    return "val_final.json" if max_episode_steps is None else f"val_final_h{max_episode_steps}.json"


def read_val_success(
    run_root: Path, task: str, num_demos: int, seeds: Sequence[int], max_episode_steps: int | None = None
) -> list[float] | None:
    """``success_once`` of each seed's validation result; ``None`` if any is missing.

    With ``max_episode_steps`` the results evaluated at that horizon are read,
    and a result that records another horizon is rejected.
    """
    values = []
    for seed in seeds:
        path = baseline_run_dir(run_root, task, num_demos, seed) / "eval" / val_result_name(max_episode_steps)
        if not path.is_file():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        recorded = result.get("max_episode_steps")
        if max_episode_steps is not None and recorded is not None and int(recorded) != max_episode_steps:
            raise ValueError(f"{path} was evaluated at {recorded} steps, not {max_episode_steps}")
        values.append(float(result["summary"]["success_once"]))
    return values


def select_baseline_cell(
    n100: Sequence[float], n200: Sequence[float] | None, *, low: float, high: float, allow_low_success: bool = True
) -> dict[str, Any]:
    """Plan §1: N=100 if its mean is in [low, high), else N=200, else low-success mode.

    Without ``allow_low_success`` the last case is an error: the replication
    task is not run outside the band (plan §12.3).
    """
    mean100 = statistics.fmean(n100)
    if mean100 >= high:
        raise ValueError(f"N=100 mean validation success {mean100:.3f} >= {high}: too few failures")
    if mean100 >= low:
        return {"num_demos": 100, "mode": "normal", "mean_val_success": mean100}
    if n200 is None:
        raise ValueError(
            f"N=100 mean validation success {mean100:.3f} < {low}: the N=200 validation results are needed"
        )
    mean200 = statistics.fmean(n200)
    if low <= mean200 < high:
        return {"num_demos": 200, "mode": "normal", "mean_val_success": mean200}
    if mean200 >= high:
        raise ValueError(f"N=200 mean validation success {mean200:.3f} >= {high}")
    if not allow_low_success:
        raise ValueError(
            f"mean validation success N=100 {mean100:.3f}, N=200 {mean200:.3f}: neither is in [{low}, {high}); "
            "low-success mode is disabled, so the study is not run on this task (plan §12.3)"
        )
    best = (200, mean200) if mean200 > mean100 else (100, mean100)
    return {"num_demos": best[0], "mode": "low-success", "mean_val_success": best[1]}


def select_pilot(candidates: Sequence[Mapping[str, Any]], relative_tie: float) -> Mapping[str, Any]:
    """Plan §4.4: largest margin; within ``relative_tie`` prefer fewer steps, then lower lr."""
    if not candidates:
        raise ValueError("no pilot candidates")
    best = max(candidate["margin"] for candidate in candidates)
    threshold = best - abs(best) * relative_tie
    eligible = [candidate for candidate in candidates if candidate["margin"] >= threshold]
    return min(eligible, key=lambda candidate: (candidate["steps"], candidate["lr"]))


def select_alpha(pooled: Mapping[float, int], tie_episodes: int) -> dict[str, Any]:
    """Plan §5.5: most pooled successes; within ``tie_episodes`` the smaller value wins."""
    if not pooled:
        raise ValueError("no tuning results")
    best = max(pooled.values())
    value = min(grid for grid, successes in pooled.items() if successes >= best - tie_episodes)
    return {
        "grid_value": value,
        "successes": pooled[value],
        "at_grid_edge": value in (min(pooled), max(pooled)),
    }


def select_guidance_arm(fixed: Mapping[str, Any], adaptive: Mapping[str, Any]) -> str:
    """The arm whose selected alpha has more pooled successes; ties go to F (plan §5.5)."""
    return "F" if fixed["successes"] >= adaptive["successes"] else "A"


# ------------------------------------------------------------------ locking


def checkpoint_seeds(lock: LockFile) -> list[int]:
    checkpoints = lock.require("checkpoints", "select-cell")
    return sorted(int(key.removeprefix("s")) for key in checkpoints if key.startswith("s"))


def rollout_dir(lock: LockFile, seed: int) -> Path:
    """``<rollout root>/failure_aware/<task>/s<seed>``; the root is locked by select-cell."""
    entry = lock.require(f"checkpoints.s{seed}", "select-cell")
    task = lock.require("task", "select-cell")
    root = task.get("rollout_root")
    return rollout_dir_for(Path(entry["path"]), task["name"], seed, Path(root) if root else None)


def study_horizon(lock: LockFile) -> int | None:
    """``max_episode_steps`` of every closed-loop stage (plan §1, §13), locked by select-cell.

    ``None`` (lock files written before the horizon was locked) keeps each
    checkpoint's recorded horizon.
    """
    value = lock.require("task", "select-cell").get("max_episode_steps")
    return None if value is None else int(value)


def dataset_dir(lock: LockFile, seed: int) -> Path:
    return rollout_dir(lock, seed) / DATASET_DIR


def low_success(lock: LockFile) -> bool:
    """Low-success mode (plan §1): the cell rule fell through, or a checkpoint
    produced fewer successes than ``collection.low_success_min``."""
    if lock.require("task", "select-cell").get("mode") == "low-success":
        return True
    return any(bool(entry.get("low_success")) for entry in (lock.get("collection") or {}).values())


def model_checkpoint(rollout: Path, label: str, lr: float, steps: int, checkpoint_steps: int | None = None) -> Path:
    """Checkpoint of a ``finetune_dp.py`` run; ``checkpoint_steps`` selects a pilot step."""
    run = rollout / model_run_name(label, lr, steps)
    if checkpoint_steps is None or checkpoint_steps == steps:
        return run / "checkpoints" / "final.pt"
    return run / "checkpoints" / f"step_{checkpoint_steps:06d}.pt"


@dataclass(frozen=True)
class ArmSpec:
    """Everything one arm evaluation needs, resolved from the lock file only."""

    task: str
    arm: str
    split: str
    seed: int
    name: str
    base_path: str
    base_sha256: str
    policy_path: str | None = None  # C2: the success model evaluated on its own
    policy_sha256: str | None = None
    negative_path: str | None = None
    negative_sha256: str | None = None
    mode: str | None = None
    grid_value: float | None = None
    alpha: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value is not None}


def resolve_arm(
    lock: LockFile,
    protocol: FailureProtocol,
    *,
    arm: str,
    split: str,
    seed: int,
    grid_value: float | None = None,
) -> ArmSpec:
    """Resolve one (arm, split, checkpoint, alpha) cell, enforcing stage order.

    Tuning takes ``grid_value`` from the protocol grid; the test split takes
    it only from the lock file's selections, after the offline gate passed.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    task = lock.require("task", "select-cell")
    checkpoint = lock.require(f"checkpoints.s{seed}", "select-cell")
    if arm in ("C1", "C2") and low_success(lock):
        raise RuntimeError("low-success mode: C1/C2 are not run (plan §1)")
    if split == "tuning":
        if arm not in ("F", "A", "C1"):
            raise ValueError("only F, A and C1 are tuned (plan §5.5)")
        if grid_value is None or grid_value not in protocol.guidance.alpha_grid:
            raise ValueError(f"tuning needs a grid value from {protocol.guidance.alpha_grid}")
    else:
        if grid_value is not None:
            raise ValueError("test alphas come from the lock file, not the command line")
        gate = lock.require("gate", "gate")
        if not gate.get("passed"):
            raise RuntimeError("the offline gate failed: the study stops before closed-loop tests (plan §6.7)")

    models = lock.require(f"models.s{seed}", "record-models") if arm != "B" else None
    mode = GUIDED_ARMS.get(arm)
    negative_path = negative_sha = policy_path = policy_sha = None
    if arm in ("F", "A"):
        negative_path, negative_sha = models["failure"], models["failure_sha256"]
    if arm == "C1":
        negative_path, negative_sha = models["success"], models["success_sha256"]
    if arm == "C2":
        policy_path, policy_sha = models["success"], models["success_sha256"]

    if split == "test" and arm in ("F", "A", "C1"):
        selection = lock.require("guidance.selection", "select-alpha")
        if arm == "C1":
            grid_value = lock.require("guidance.c1", "select-c1")["grid_value"]
        else:
            grid_value = selection[f"grid_value_{arm}"]
    if arm == "C1":
        mode = GUIDED_ARMS[lock.require("guidance.selection", "select-alpha")["guidance_arm"]]

    alpha = None
    if mode is not None:
        assert grid_value is not None
        if mode == "adaptive":
            m = float(lock.require("guidance.dry_run", "dry-run")["m"])
            alpha = float(grid_value) / m
        else:
            alpha = float(grid_value)
    suffix = f"_a{grid_value:g}" if mode is not None else ""
    return ArmSpec(
        task=task["name"],
        arm=arm,
        split=split,
        seed=seed,
        name=f"{task['name']}_unet_n{task['num_demos']}_s{seed}_{arm}{suffix}",
        base_path=checkpoint["path"],
        base_sha256=checkpoint["sha256"],
        policy_path=policy_path,
        policy_sha256=policy_sha,
        negative_path=negative_path,
        negative_sha256=negative_sha,
        mode=mode,
        grid_value=None if grid_value is None else float(grid_value),
        alpha=alpha,
    )


def eval_output_path(lock: LockFile, spec: ArmSpec) -> Path:
    return rollout_dir(lock, spec.seed) / "eval" / spec.split / f"{spec.name}.json"


def tuning_seeds(protocol: FailureProtocol) -> list[int]:
    start = protocol.seeds.guidance_tuning[0]
    return list(range(start, start + protocol.guidance.tuning_episodes))


def dry_run_seeds(protocol: FailureProtocol) -> list[int]:
    return tuning_seeds(protocol)[: protocol.guidance.dry_run_episodes]


def pooled_tuning_successes(
    lock: LockFile, protocol: FailureProtocol, arm: str
) -> dict[float, int]:
    """Pool ``success_once`` over checkpoints per grid value, checking provenance."""
    pooled: dict[float, int] = {}
    missing: list[str] = []
    expected_seeds = tuning_seeds(protocol)
    for grid_value in protocol.guidance.alpha_grid:
        total = 0
        for seed in checkpoint_seeds(lock):
            spec = resolve_arm(lock, protocol, arm=arm, split="tuning", seed=seed, grid_value=grid_value)
            path = eval_output_path(lock, spec)
            if not path.is_file():
                missing.append(str(path))
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            recorded = result["arm"]
            for key in ("arm", "seed", "base_sha256", "negative_sha256", "mode", "grid_value", "alpha"):
                if recorded.get(key) != spec.to_dict().get(key):
                    raise ValueError(f"{path}: {key} {recorded.get(key)!r} != lock {spec.to_dict().get(key)!r}")
            if [episode["seed"] for episode in result["episodes"]] != expected_seeds:
                raise ValueError(f"{path}: not the declared tuning seeds")
            total += sum(bool(episode["success_once"]) for episode in result["episodes"])
        pooled[float(grid_value)] = total
    if missing:
        raise FileNotFoundError("missing tuning results:\n  " + "\n  ".join(missing))
    return pooled


# ----------------------------------------------------------- torch helpers


def denoising_loss(policy, info, device, *, seed: int = 0, batch_size: int = 64) -> float:
    """Mean denoising loss over every window of ``info`` with fixed noise.

    The generator is re-seeded per call and batches are in dataset order, so
    two models evaluated on the same dataset see identical noise and
    timesteps: their loss difference reflects only the models (plan §6.7).
    """
    import torch
    from torch.utils.data import DataLoader

    from .data import RGBWindowDataset

    # Windows are read in order in the main process; lazily, each one would
    # re-inflate gzip chunks spanning 23 frames. Preloading reads each chunk once
    # and yields bit-identical windows.
    dataset = RGBWindowDataset(info, policy.obs_horizon, policy.pred_horizon, preload=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = policy.training
    policy.eval()
    total, count = 0.0, 0
    try:
        with torch.no_grad():
            for batch in loader:
                rgb, proprio, actions = (batch[key].to(device) for key in ("rgb", "proprio", "actions"))
                loss = policy.compute_loss(rgb, proprio, actions, generator=generator)
                total += float(loss) * len(rgb)
                count += len(rgb)
    finally:
        dataset.close()
        policy.train(was_training)
    return total / count


def offline_gaps(base, model, failure_info, success_info, device) -> dict[str, float]:
    """``gap = L_base - L_model`` on failures and successes, and their margin."""
    losses = {
        "base_failure": denoising_loss(base, failure_info, device),
        "model_failure": denoising_loss(model, failure_info, device),
        "base_success": denoising_loss(base, success_info, device),
        "model_success": denoising_loss(model, success_info, device),
    }
    gap_fail = losses["base_failure"] - losses["model_failure"]
    gap_succ = losses["base_success"] - losses["model_success"]
    return {**losses, "gap_fail": gap_fail, "gap_succ": gap_succ, "margin": gap_fail - gap_succ}


def load_policy(path: str | Path, expected_sha256: str, device):
    """Load a checkpoint after checking it is the file the lock file names."""
    import torch

    from .metadata import file_sha256
    from .policy import DiffusionPolicy

    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path} has sha256 {actual}, but the lock file records {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return DiffusionPolicy.from_checkpoint(payload, device).eval(), payload


def build_arm_policy(spec: ArmSpec, device, diagnostics=None):
    """The policy an arm evaluates, loaded strictly from the lock file's hashes."""
    from .failure_guidance import FailureGuidedPolicy
    from .metadata import file_sha256

    if spec.mode is None:
        path, sha = (spec.policy_path, spec.policy_sha256) if spec.arm == "C2" else (spec.base_path, spec.base_sha256)
        assert path is not None and sha is not None
        return load_policy(path, sha, device)[0]
    assert spec.negative_path is not None and spec.alpha is not None
    if file_sha256(spec.negative_path) != spec.negative_sha256:
        raise ValueError(f"{spec.negative_path} does not match the lock file's sha256")
    policy = FailureGuidedPolicy.from_checkpoints(
        spec.base_path, spec.negative_path, mode=spec.mode, alpha=spec.alpha, device=device, diagnostics=diagnostics
    )
    if policy.provenance["base_checkpoint"]["sha256"] != spec.base_sha256:
        raise ValueError(f"{spec.base_path} does not match the lock file's sha256")
    return policy.eval()
