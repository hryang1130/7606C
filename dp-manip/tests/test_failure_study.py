"""Phase 4 of the failure-aware plan: the staged study and its lock file.

Unit tests cover the pre-registered rules and the lock file. The end-to-end
test runs the real command-line stages on synthetic data with a fake vector
environment, from the baseline cell rule to the test evaluation, and checks the
Phase 4 acceptance criteria: every arm uses the same tuning and test seeds,
the dry-run m and the alpha selections are locked before any test run, test
evaluation reads checkpoints and alphas only from the lock file, and results
record both checkpoint hashes and the guidance mode.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from dp_manip import failure_study as study
from dp_manip.failure_lock import LockFile, dumps
from dp_manip.failure_protocol import DEFAULT_PROTOCOL, load_protocol

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip import config as config_lib
    from dp_manip.trainer import run_training
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASK = "placesphere"
# The study task and its planned replication (plan §12); the end-to-end study
# runs on both so that neither depends on a task-specific path.
STUDY_TASKS = ("placesphere", "liftpegupright")
# Synthetic shapes, not PlaceSphere's. The pipeline test expects this data to
# fail the offline gate; changing the shapes changes the random data.
PROPRIO_DIM = 5
ACTION_DIM = 8
# The study horizon. The synthetic baselines record a horizon too short for any
# success (fake successes start at step 3), like PlaceSphere's checkpoints
# recording 50 steps; the study only works if every stage applies the lock's
# horizon (plan §13).
MAX_STEPS = 12
RECORDED_STEPS = 2


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LockFileTest(unittest.TestCase):
    def test_dumps_round_trips_nested_tables(self) -> None:
        data = {
            "task": {"name": "peg", "num_demos": 100, "mean": 0.25, "flags": [True, False], "text": 'a "b"'},
            "gate": {"passed": False, "s1": {"gap_fail": -1.5e-05, "values": [0.5, 1.0]}},
            "empty": {},
        }
        self.assertEqual(tomllib.loads(dumps(data)), data)

    def test_sections_are_written_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = LockFile(Path(directory) / "task.toml")
            lock.write("guidance.dry_run", {"m": 0.02, "seed_range": (22000, 22008), "unused": None})
            reread = LockFile(lock.path)
            self.assertEqual(reread.get("guidance.dry_run.m"), 0.02)
            self.assertEqual(reread.get("guidance.dry_run.seed_range"), [22000, 22008])
            self.assertIsNone(reread.get("guidance.dry_run.unused"))
            self.assertIn("recorded_utc", reread.require("guidance.dry_run", "dry-run"))
            with self.assertRaises(FileExistsError):
                reread.write("guidance.dry_run", {"m": 0.5})
            reread.write("guidance.selection", {"guidance_arm": "A"})
            self.assertEqual(LockFile(lock.path).get("guidance.dry_run.m"), 0.02)
            with self.assertRaisesRegex(RuntimeError, "run the 'gate' stage"):
                reread.require("gate", "gate")


class RuleTest(unittest.TestCase):
    def test_baseline_cell_rule(self) -> None:
        pick = study.select_baseline_cell
        self.assertEqual(pick([0.2, 0.3], None, low=0.15, high=0.85)["num_demos"], 100)
        self.assertEqual(pick([0.1, 0.1], [0.2, 0.3], low=0.15, high=0.85)["num_demos"], 200)
        low = pick([0.1, 0.12], [0.02, 0.04], low=0.15, high=0.85)
        self.assertEqual((low["num_demos"], low["mode"]), (100, "low-success"))
        low = pick([0.02, 0.04], [0.1, 0.12], low=0.15, high=0.85)
        self.assertEqual((low["num_demos"], low["mode"]), (200, "low-success"))
        with self.assertRaisesRegex(ValueError, "N=200 validation results are needed"):
            pick([0.1, 0.1], None, low=0.15, high=0.85)
        with self.assertRaisesRegex(ValueError, "too few failures"):
            pick([0.9, 0.95], None, low=0.15, high=0.85)

    def test_validation_results_are_read_at_the_study_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for seed, success in ((1, 0.2), (2, 0.4)):
                results = study.baseline_run_dir(run_root, TASK, 100, seed) / "eval"
                results.mkdir(parents=True)
                (results / "val_final.json").write_text(json.dumps({"summary": {"success_once": 0.0}}))
                (results / "val_final_h200.json").write_text(
                    json.dumps({"summary": {"success_once": success}, "max_episode_steps": 200})
                )
            self.assertEqual(study.read_val_success(run_root, TASK, 100, [1, 2]), [0.0, 0.0])
            self.assertEqual(study.read_val_success(run_root, TASK, 100, [1, 2], 200), [0.2, 0.4])
            self.assertIsNone(study.read_val_success(run_root, TASK, 100, [1, 2], 150))
            self.assertIsNone(study.read_val_success(run_root, TASK, 200, [1, 2], 200))
            (results / "val_final_h200.json").write_text(
                json.dumps({"summary": {"success_once": 0.4}, "max_episode_steps": 50})
            )
            with self.assertRaisesRegex(ValueError, "evaluated at 50 steps, not 200"):
                study.read_val_success(run_root, TASK, 100, [1, 2], 200)

    def test_pilot_prefers_fewer_steps_then_lower_lr_within_the_tie(self) -> None:
        candidates = [
            {"lr": 1e-4, "steps": 20000, "margin": 1.00},
            {"lr": 1e-4, "steps": 10000, "margin": 0.96},
            {"lr": 1e-5, "steps": 10000, "margin": 0.97},
            {"lr": 1e-5, "steps": 5000, "margin": 0.90},
        ]
        self.assertEqual(study.select_pilot(candidates, 0.05), candidates[2])
        self.assertEqual(study.select_pilot(candidates, 0.0), candidates[0])
        negative = [{"lr": 1e-5, "steps": 5000, "margin": -0.10}, {"lr": 1e-4, "steps": 5000, "margin": -0.104}]
        self.assertEqual(study.select_pilot(negative, 0.05), negative[0])

    def test_alpha_rule_prefers_the_smaller_value_within_the_tie(self) -> None:
        self.assertEqual(study.select_alpha({0.5: 60, 1.0: 62, 2.0: 59}, 2)["grid_value"], 0.5)
        choice = study.select_alpha({0.5: 60, 1.0: 63, 2.0: 59}, 2)
        self.assertEqual((choice["grid_value"], choice["at_grid_edge"]), (1.0, False))
        self.assertTrue(study.select_alpha({0.5: 50, 1.0: 55, 2.0: 70}, 2)["at_grid_edge"])
        self.assertEqual(study.select_guidance_arm({"successes": 70}, {"successes": 70}), "F")
        self.assertEqual(study.select_guidance_arm({"successes": 70}, {"successes": 71}), "A")


class ResolveArmTest(unittest.TestCase):
    def lock(self, **sections) -> LockFile:
        lock = LockFile(Path(tempfile.gettempdir()) / "unused-failure-lock.toml")
        lock.data = {
            "task": {"name": TASK, "num_demos": 100, "mode": "normal"},
            "checkpoints": {"s1": {"path": "/runs/peg_s1/checkpoints/final.pt", "sha256": "b" * 64}},
            "models": {"s1": {"failure": "/f.pt", "failure_sha256": "f" * 64, "success": "/s.pt", "success_sha256": "s" * 64}},
            **sections,
        }
        return lock

    def test_stage_order_is_enforced(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        lock = self.lock()
        with self.assertRaisesRegex(RuntimeError, "dry-run"):
            study.resolve_arm(lock, protocol, arm="A", split="tuning", seed=1, grid_value=1.0)
        with self.assertRaisesRegex(RuntimeError, "select-alpha"):
            study.resolve_arm(lock, protocol, arm="C1", split="tuning", seed=1, grid_value=1.0)
        with self.assertRaisesRegex(RuntimeError, "'gate'"):
            study.resolve_arm(lock, protocol, arm="B", split="test", seed=1)
        failed = self.lock(gate={"passed": False})
        with self.assertRaisesRegex(RuntimeError, "gate failed"):
            study.resolve_arm(failed, protocol, arm="B", split="test", seed=1)
        with self.assertRaisesRegex(ValueError, "grid value"):
            study.resolve_arm(lock, protocol, arm="F", split="tuning", seed=1, grid_value=0.7)
        with self.assertRaisesRegex(ValueError, "not the command line"):
            study.resolve_arm(self.lock(gate={"passed": True}), protocol, arm="F", split="test", seed=1, grid_value=1.0)

    def test_test_alphas_come_from_the_lock_file(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        lock = self.lock(
            gate={"passed": True},
            guidance={
                "dry_run": {"m": 0.02},
                "selection": {"guidance_arm": "A", "grid_value_F": 2.0, "grid_value_A": 1.0},
                "c1": {"grid_value": 0.5},
            },
        )
        adaptive = study.resolve_arm(lock, protocol, arm="A", split="test", seed=1)
        self.assertEqual((adaptive.mode, adaptive.grid_value, adaptive.alpha), ("adaptive", 1.0, 50.0))
        self.assertEqual(adaptive.negative_sha256, "f" * 64)
        self.assertEqual(adaptive.name, f"{TASK}_unet_n100_s1_A_a1")
        control = study.resolve_arm(lock, protocol, arm="C1", split="test", seed=1)
        self.assertEqual((control.mode, control.alpha, control.negative_sha256), ("adaptive", 25.0, "s" * 64))
        c2 = study.resolve_arm(lock, protocol, arm="C2", split="test", seed=1)
        self.assertEqual((c2.mode, c2.policy_sha256), (None, "s" * 64))
        base = study.resolve_arm(lock, protocol, arm="B", split="test", seed=1)
        self.assertEqual((base.name, base.base_sha256), (f"{TASK}_unet_n100_s1_B", "b" * 64))

    def test_low_success_mode_drops_the_controls(self) -> None:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        lock = self.lock(gate={"passed": True}, collection={"s1": {"low_success": True}})
        with self.assertRaisesRegex(RuntimeError, "C1/C2 are not run"):
            study.resolve_arm(lock, protocol, arm="C2", split="test", seed=1)


# ------------------------------------------------------------ end to end

PROTOCOL = """
[seeds]
collection_train = [20000, 21000]
collection_holdout = [21000, 22000]
guidance_tuning = [22000, 23000]
[collection]
dataset_size = 2
rollout_cap = 8
holdout_episodes = 8
pilot_holdout_episodes = 4
failure_truncation_factor = 1.5
low_success_min = 1
[baseline_cell]
min_val_success = 0.15
max_val_success = 0.85
val_seeds = [1, 2]
checkpoint_seeds = [1, 2]
[pilot]
learning_rates = [1e-3, 1e-2]
steps = [2, 3]
relative_tie = 0.05
[guidance]
alpha_grid = [0.5, 1.0]
dry_run_episodes = 4
tuning_episodes = 4
tie_episodes = 0
[evaluation]
test_episodes = 8
"""


class FakeEnv:
    """Even reset seeds succeed from the third step on; odd seeds never do.

    Episodes are truncated at ``max_steps``, the config's ``max_episode_steps``.
    """

    def __init__(self, num_envs: int, max_steps: int = MAX_STEPS):
        self.num_envs = num_envs
        self.max_steps = max_steps

    def _observation(self):
        rng = np.random.default_rng([*self.seeds, self.t])
        return {
            "rgb": rng.integers(0, 256, (self.num_envs, 64, 64, 3), dtype=np.uint8),
            "state": rng.normal(size=(self.num_envs, PROPRIO_DIM)).astype(np.float32),
        }

    def reset(self, seed):
        self.seeds, self.t = list(seed), 0
        return self._observation(), {}

    def step(self, action):
        assert np.asarray(action).shape == (self.num_envs, ACTION_DIM)
        self.t += 1
        success = np.array([seed % 2 == 0 and self.t >= 3 for seed in self.seeds])
        truncated = np.full(self.num_envs, self.t >= self.max_steps)
        reward = np.asarray(action).sum(axis=1)
        return self._observation(), reward, np.zeros(self.num_envs, bool), truncated, {"success": success}

    def close(self) -> None:
        pass


def fake_envs(cfg, num_envs, render_backend):
    return FakeEnv(num_envs, cfg.task.max_episode_steps)


def write_expert(path: Path, seeds: list[int], env_id: str, control_mode: str) -> None:
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as file:
        for index in range(len(seeds)):
            group = file.create_group(f"traj_{index}")
            group.create_dataset("obs_rgb/rgb", data=rng.integers(0, 256, (7, 64, 64, 3), dtype=np.uint8))
            group.create_dataset("obs_rgb/state", data=rng.normal(size=(7, PROPRIO_DIM)).astype(np.float32))
            group.create_dataset("actions", data=rng.uniform(-1, 1, (6, ACTION_DIM)).astype(np.float32))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {"env_id": env_id, "env_kwargs": {"control_mode": control_mode}},
                "episodes": [{"episode_id": index, "episode_seed": seed} for index, seed in enumerate(seeds)],
            }
        )
    )


def train_baselines(root: Path, run_root: Path, seeds=(1, 2), task: str = TASK) -> dict[int, Path]:
    """Tiny main-track baselines named like the N=100 cell, recording ``RECORDED_STEPS``,
    with validation results evaluated at ``MAX_STEPS`` (``eval_dp.py --max-episode-steps``)."""
    data_root = root / "data"
    checkpoints = {}
    for seed in seeds:
        cfg = config_lib.load(
            str(ROOT / "configs" / "tasks" / f"{task}.toml"),
            [
                f"data.root={json.dumps(str(data_root))}",
                "data.num_demos=4", "data.val_num_demos=1",
                "vision.feature_dim=8", "policy.unet_dims=[16, 32]", "policy.kernel_size=3",
                "policy.n_groups=4", "diffusion.num_diffusion_iters=4", "diffusion.num_inference_iters=4",
                "train.total_iters=2", "train.batch_size=2", "train.num_workers=0", "train.log_freq=100",
                "train.validation_steps=[2]", "train.checkpoint_steps=[]", "train.amp=false",
                "ema.decay=0.9", "eval.num_envs=2", "eval.test_episodes=8", f"train.seed={seed}",
                f"task.max_episode_steps={MAX_STEPS}",
            ],
        )
        if not (data_root / cfg.data.train_path).is_file():
            env = (cfg.task.env_id, cfg.task.control_mode)
            write_expert(data_root / cfg.data.train_path, [0, 1, 2, 3], *env)
            write_expert(data_root / cfg.data.val_path, [4000], *env)
        name = f"{task}_rgb_unet_n100_s{seed}"
        assert run_training(cfg, output_root=run_root, run_name=name, device="cpu") == 0
        (run_root / name / "eval").mkdir()
        (run_root / name / "eval" / f"val_final_h{MAX_STEPS}.json").write_text(
            json.dumps({"summary": {"success_once": 0.4}, "max_episode_steps": MAX_STEPS})
        )
        checkpoints[seed] = run_root / name / "checkpoints" / "final.pt"
        # Training refuses a horizon shorter than the demonstrations, so the
        # short horizon recorded by PlaceSphere's pre-2026-10-03 checkpoints is
        # written into the checkpoint afterwards.
        payload = torch.load(checkpoints[seed], map_location="cpu", weights_only=False)
        payload["config"]["task"]["max_episode_steps"] = RECORDED_STEPS
        torch.save(payload, checkpoints[seed])
    return checkpoints


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class StudyEndToEndTest(unittest.TestCase):
    def test_full_study_on_synthetic_data(self) -> None:
        for task in STUDY_TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                self.run_study(Path(directory), task)

    def run_study(self, root: Path, task: str) -> None:
        collect = load_script("collect_rollouts")
        finetune = load_script("finetune_dp")
        cli = load_script("failure_study")
        protocol_path = root / "protocol.toml"
        protocol_path.write_text(PROTOCOL)
        lock_path = root / "lock.toml"
        run_root = root / "runs"
        common = ["--task", task, "--lock", str(lock_path), "--protocol", str(protocol_path)]

        def stage(*arguments: str) -> None:
            self.assertEqual(cli.main([*arguments, *common], envs_factory=fake_envs), 0)

        def locked(section: str):
            return LockFile(lock_path).get(section)

        checkpoints = train_baselines(root, run_root, task=task)

        stage("select-cell", "--run-root", str(run_root), "--max-episode-steps", str(MAX_STEPS))
        self.assertEqual(locked("task.num_demos"), 100)
        self.assertEqual(locked("task.max_episode_steps"), MAX_STEPS)

        horizon = ["--max-episode-steps", str(MAX_STEPS)]
        for seed, checkpoint in checkpoints.items():
            for split in ("train", "holdout"):
                collect.main(
                    ["collect", str(checkpoint), "--split", split, "--protocol", str(protocol_path),
                     "--device", "cpu", *horizon],
                    envs_factory=fake_envs,
                )
            rollout = run_root / "failure_aware" / task / f"s{seed}"
            collect.main(["build", str(rollout), "--protocol", str(protocol_path)])
            provenance = json.loads((rollout / "raw_train.json").read_text())["rollout_provenance"]
            self.assertEqual(
                (provenance["max_episode_steps"], provenance["checkpoint_max_episode_steps"]),
                (MAX_STEPS, RECORDED_STEPS),
            )
            if seed == 1:
                # A collection at another horizon is refused.
                summary_path = rollout / "datasets" / "summary.json"
                original = summary_path.read_text()
                summary_path.write_text(json.dumps({**json.loads(original), "max_episode_steps": RECORDED_STEPS}))
                with self.assertRaisesRegex(ValueError, "the locked horizon is 12"):
                    stage("record-collection", "--seed", "1")
                summary_path.write_text(original)
            stage("record-collection", "--seed", str(seed))
        self.assertEqual(locked("collection.s1.train_rollouts"), 4)

        def tune(seed: int, label: str, lr: float, steps: int, *extra: str) -> None:
            argv = [str(checkpoints[seed]), "--label", label, "--lr", repr(lr), "--steps", str(steps),
                    "--protocol", str(protocol_path), "--device", "cpu", "--set", "train.warmup_steps=1", *extra]
            self.assertEqual(finetune.main(argv), 0)

        for lr in (1e-3, 1e-2):
            tune(1, "failure", lr, 3, "--checkpoint-steps", "2")
        stage("pilot", "--device", "cpu")
        lr, steps = locked("finetune.lr"), locked("finetune.steps")
        self.assertEqual(len(locked("finetune.candidate_margin")), 4)
        tune(1, "success", lr, steps)
        for label in ("failure", "success"):
            tune(2, label, lr, steps)
        for seed in (1, 2):
            stage("record-models", "--seed", str(seed))
        self.assertEqual(locked("models.s1.failure"), locked("finetune.pilot_checkpoint"))

        stage("gate", "--device", "cpu")
        self.assertIn("gap_fail", locked("gate.s1"))
        if not locked("gate.passed"):
            # Synthetic failures and successes are not separable, so the gate
            # may fail; the study must then refuse to evaluate, and the rest
            # of the flow is exercised as if it had passed.
            with self.assertRaisesRegex(RuntimeError, "gate failed"):
                stage("dry-run", "--device", "cpu")
            data = tomllib.loads(lock_path.read_text())
            data["gate"]["passed"] = True
            lock_path.write_text(dumps(data))

        with self.assertRaisesRegex(RuntimeError, "select-alpha"):
            stage("eval", "--split", "test", "--arm", "F", "--device", "cpu")
        stage("dry-run", "--device", "cpu")
        self.assertTrue(locked("guidance.dry_run.alpha0_reproduces_baseline"))
        self.assertGreater(locked("guidance.dry_run.m"), 0.0)

        for arm in ("F", "A"):
            stage("eval", "--split", "tuning", "--arm", arm, "--device", "cpu")
        stage("select-alpha")
        # Success depends only on the reset seed here, so every setting ties:
        # the smaller grid value and the F arm win (plan §5.5 tie rules).
        self.assertEqual(locked("guidance.selection.guidance_arm"), "F")
        self.assertEqual(locked("guidance.selection.grid_value_A"), 0.5)
        self.assertTrue(locked("guidance.selection.at_grid_edge_A"))
        stage("eval", "--split", "tuning", "--arm", "C1", "--device", "cpu")
        stage("select-c1")
        with self.assertRaises(FileExistsError):
            stage("select-alpha")

        for arm in study.ARMS:
            stage("eval", "--split", "test", "--arm", arm, "--device", "cpu")
        lock = LockFile(lock_path)
        protocol = load_protocol(protocol_path)
        test_seeds = list(range(10_000, 10_008))
        for seed in (1, 2):
            for arm in study.ARMS:
                spec = study.resolve_arm(lock, protocol, arm=arm, split="test", seed=seed)
                result = json.loads(study.eval_output_path(lock, spec).read_text())
                self.assertEqual([episode["seed"] for episode in result["episodes"]], test_seeds)
                self.assertEqual(result["arm"]["base_sha256"], locked(f"checkpoints.s{seed}.sha256"))
                self.assertEqual(result["lock"]["sha256"], lock.sha256)
                self.assertEqual(
                    (result["max_episode_steps"], result["checkpoint_max_episode_steps"]), (MAX_STEPS, RECORDED_STEPS)
                )
                if arm in ("F", "A", "C1"):
                    self.assertIn(result["arm"]["mode"], ("fixed", "adaptive"))
                    self.assertTrue(result["arm"]["negative_sha256"])
                    self.assertEqual(len(result["diagnostics"]["timesteps"]), 4)
            tuning = json.loads(
                study.eval_output_path(
                    lock, study.resolve_arm(lock, protocol, arm="F", split="tuning", seed=seed, grid_value=0.5)
                ).read_text()
            )
            self.assertEqual([episode["seed"] for episode in tuning["episodes"]], [22_000, 22_001, 22_002, 22_003])

        # Resubmitting skips finished results; a changed model file is refused.
        stage("eval", "--split", "test", "--arm", "C2", "--device", "cpu")
        spec = study.resolve_arm(lock, protocol, arm="C2", split="test", seed=2)
        study.eval_output_path(lock, spec).unlink()
        with open(locked("models.s2.success"), "ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "lock file records"):
            stage("eval", "--split", "test", "--arm", "C2", "--seeds", "2", "--device", "cpu")
        stage("status")


if __name__ == "__main__":
    unittest.main()
