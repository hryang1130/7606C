"""Complete state must stay distinct from RGB proprioception end to end."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, from_recorded, load, same_run
from dp_manip.runlist import runs

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip.data import ObservationWindowDataset, compute_normalization, read_dataset_info
    from dp_manip.envs import environment_kwargs
    from dp_manip.evaluate import _adapt_environment_observation, evaluate
    from dp_manip.policy import DiffusionPolicy
    from dp_manip.trainer import run_training
except ModuleNotFoundError:
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs/tasks/peginsertionside.toml"
EXPERIMENT = ROOT / "configs/experiments/state_n100.toml"


def state_config(*overrides):
    return load(TASK, overrides, experiment=EXPERIMENT, experiment_value="state")


def write_state_dataset(path: Path, seeds: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    entries = []
    with h5py.File(path, "w") as file:
        for episode_id, seed in enumerate(seeds):
            group = file.create_group(f"traj_{episode_id}")
            state = rng.normal(size=(4, 43)).astype(np.float32)
            state[:, -1] = seed + np.arange(4)
            group.create_dataset("obs", data=state)
            group.create_dataset("actions", data=rng.uniform(-2, 2, (3, 8)).astype(np.float32))
            group.create_dataset("success", data=np.array([False, False, True]))
            # A different width/value makes accidental proprio reads observable.
            # There is deliberately no RGB dataset to permit a visual fallback.
            group.create_dataset("obs_rgb/state", data=np.full((4, 25), -999, dtype=np.float32))
            entries.append({"episode_id": episode_id, "episode_seed": seed})
    path.with_suffix(".json").write_text(json.dumps({
        "env_info": {"env_id": "PegInsertionSide-v1", "env_kwargs": {
            "control_mode": "pd_joint_pos", "obs_mode": "state",
        }},
        "episodes": entries,
    }), encoding="utf-8")


class StateConfigTest(unittest.TestCase):
    def test_single_run_and_backward_compatible_rgb_identity(self):
        cells = runs(EXPERIMENT, "peginsertionside")
        self.assertEqual(len(cells), 1)
        cfg = cells[0].resolve()
        self.assertEqual((cfg.data.num_demos, cfg.train.seed, cfg.train.total_iters), (100, 1, 100000))
        self.assertEqual(cells[0].name, "peginsertionside_state_unet_n100_s1")
        rgb = load(TASK)
        recorded = rgb.to_dict()
        del recorded["task"]["obs_mode"]
        self.assertEqual(from_recorded(recorded), rgb)
        self.assertTrue(same_run(from_recorded(recorded), rgb))
        self.assertFalse(same_run(rgb, cfg))
        self.assertEqual(default_run_name(rgb), "peginsertionside_rgb_unet_n100_s1")
        with self.assertRaisesRegex(ValueError, "task.obs_mode"):
            load(TASK, ["task.obs_mode=depth"])

    def test_single_gpu_script_runs_and_evaluates_one_cell(self):
        script = ROOT / "slurm/state_single_gpu.sbatch"
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("--workers 1", text)
        self.assertIn("for split in val test", text)

    def test_submission_script_forwards_valid_arguments_and_stops_on_failure(self):
        from scripts.eval_queue import parse_args as parse_eval
        from scripts.train_queue import parse_args as parse_train

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = root / "python"
            recorder.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['STATE_ARGS'], 'a') as f:\n"
                "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "sys.exit(int(os.environ.get('STATE_EXIT', '0')))\n",
                encoding="utf-8",
            )
            recorder.chmod(0o755)
            record = root / "args.jsonl"
            env = dict(os.environ, SLURM_SUBMIT_DIR=str(ROOT), DATA_ROOT=str(root / "data"),
                       RUN_ROOT=str(root / "runs"), PYTHON=str(recorder), STATE_ARGS=str(record))
            script = ROOT / "slurm/state_single_gpu.sbatch"
            subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
            calls = [json.loads(line) for line in record.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            train = parse_train(calls[0][1:])
            self.assertEqual((train.workers, train.task), (1, "peginsertionside"))
            for call, split in zip(calls[1:], ("val", "test")):
                evaluation = parse_eval(call[1:])
                self.assertEqual((evaluation.workers, evaluation.split), (1, split))
            record.unlink()
            failed = subprocess.run(["bash", str(script)], env=dict(env, STATE_EXIT="1"), capture_output=True)
            self.assertEqual(failed.returncode, 1)
            self.assertEqual(len(record.read_text().splitlines()), 1)

            # Exit 75 must requeue the same job and skip evaluation.
            record.unlink()
            controller = root / "scontrol"
            controller.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$STATE_REQUEUE"\n', encoding="utf-8")
            controller.chmod(0o755)
            requeue = root / "requeue"
            env.update(PATH=f"{root}:{env['PATH']}", SLURM_JOB_ID="123", STATE_REQUEUE=str(requeue))
            subprocess.run(["bash", str(script)], env=dict(env, STATE_EXIT="75"), check=True, capture_output=True)
            self.assertEqual(requeue.read_text().strip(), "requeue 123")
            self.assertEqual(len(record.read_text().splitlines()), 1)


@unittest.skipUnless(HAVE_TORCH, "requires torch, diffusers and h5py")
class StateDataTest(unittest.TestCase):
    def test_selected_state_statistics_and_boundary_windows_without_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.h5"
            write_state_dataset(path, list(reversed(range(105))))
            info = read_dataset_info(path, 100, obs_mode="state")
            self.assertEqual(info.seeds, list(range(100)))
            self.assertEqual(info.proprio_dim, 43)
            self.assertIsNone(info.image_shape)
            self.assertEqual(info.num_cameras, 0)
            stats = compute_normalization(info)
            self.assertAlmostEqual(float(stats.proprio_mean[-1]), 50.5)
            lazy = ObservationWindowDataset(info, 2, 16, preload=False)
            preload = ObservationWindowDataset(info, 2, 16)
            try:
                for item in (0, 2, len(lazy) - 1):
                    self.assertEqual(set(preload[item]), {"proprio", "actions"})
                    for key in preload[item]:
                        np.testing.assert_array_equal(lazy[item][key], preload[item][key])
                np.testing.assert_array_equal(preload[0]["proprio"][0], preload[0]["proprio"][1])
                np.testing.assert_array_equal(preload[2]["actions"][-1], preload[2]["actions"][-2])
                self.assertLess(preload.preloaded_bytes, 100000)
            finally:
                lazy.close()
                preload.close()
            with self.assertRaisesRegex(ValueError, "missing obs_rgb/rgb"):
                read_dataset_info(path, 100)

    def test_misaligned_state_and_nonfinite_privileged_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.h5"
            write_state_dataset(path, [0])
            with h5py.File(path, "r+") as file:
                file["traj_0/obs"][0, -1] = np.nan
            info = read_dataset_info(path, obs_mode="state")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                compute_normalization(info)
            with h5py.File(path, "r+") as file:
                del file["traj_0/obs"]
                file.create_dataset("traj_0/obs", data=np.zeros((3, 43), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "aligned"):
                read_dataset_info(path, obs_mode="state")


class FakeStateEnvs:
    num_envs = 2

    def reset(self, *, seed):
        self.step_count = 0
        self.state = np.repeat(np.asarray(seed, dtype=np.float32)[:, None], 43, axis=1)
        return self.state[:, None], {}

    def step(self, actions):
        assert actions.shape == (2, 8) and np.isfinite(actions).all()
        self.step_count += 1
        return self.state[:, None] + self.step_count, np.ones(2), np.zeros(2, bool), np.full(2, self.step_count == 3), {"success": np.full(2, self.step_count == 2)}


@unittest.skipUnless(HAVE_TORCH, "requires torch, diffusers and h5py")
class StateLifecycleTest(unittest.TestCase):
    def test_train_load_resume_and_flat_state_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = state_config(
                f"data.root={root / 'data'}", "policy.unet_dims=[16,32]",
                "policy.kernel_size=3", "policy.n_groups=4",
                "diffusion.num_diffusion_iters=4", "diffusion.num_inference_iters=2",
                "train.total_iters=2", "train.batch_size=32", "train.num_workers=0",
                "train.log_freq=1", "train.resume_freq=1", "train.validation_steps=[2]",
                "train.checkpoint_steps=[1]", "train.amp=false",
            )
            for split, seeds in (("train", list(reversed(range(105)))), ("val", list(range(4000, 4050)))):
                write_state_dataset(Path(cfg.data.root) / getattr(cfg.data, f"{split}_path"), seeds)
            output = root / "runs"
            self.assertEqual(run_training(cfg, output_root=output, device="cpu"), 0)
            run_dir = output / default_run_name(cfg)
            final = run_dir / "checkpoints/final.pt"
            checkpoint = torch.load(final, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["train_data"]["seeds"], list(range(100)))
            self.assertEqual(checkpoint["train_data"]["observation_key"], "obs")
            self.assertEqual(checkpoint["train_data"]["proprio_dim"], 43)
            self.assertIsNone(checkpoint["train_data"]["image_shape"])
            policy = DiffusionPolicy.from_checkpoint(checkpoint).eval()
            self.assertFalse(any("image_encoder" in name for name, _ in policy.named_parameters()))
            state = torch.randn(2, 2, 43)
            self.assertEqual(policy.observation_features(None, state).shape, (2, 2, 43))
            result = evaluate(policy, FakeStateEnvs(), [5000, 5001, 5002, 5003], torch.device("cpu"), inference_seed=0)
            self.assertEqual(result["summary"]["success_once"], 1.0)
            self.assertEqual(result["summary"]["success_at_end"], 0.0)
            self.assertEqual(result["summary"]["num_episodes"], 4)
            final.unlink()
            self.assertEqual(run_training(cfg, output_root=output, device="cpu"), 0)
            resumed = torch.load(final, map_location="cpu", weights_only=False)
            for name, value in checkpoint["model"].items():
                torch.testing.assert_close(value, resumed["model"][name], rtol=0, atol=0)
            checkpoint["config"]["task"]["obs_mode"] = "rgb"
            with self.assertRaisesRegex(ValueError, "observation mode"):
                DiffusionPolicy.from_checkpoint(checkpoint)

    def test_state_environment_contract_and_wrong_observation_rejection(self):
        cfg = state_config()
        kwargs = environment_kwargs(cfg)
        self.assertEqual(kwargs["obs_mode"], "state")
        self.assertEqual(kwargs["sim_backend"], "physx_cpu")
        self.assertNotIn("sensor_configs", kwargs)
        rgb, state = _adapt_environment_observation(np.zeros((2, 1, 43)), 2, "state")
        self.assertIsNone(rgb)
        self.assertEqual(state.shape, (2, 43))
        with self.assertRaisesRegex(ValueError, "complete state"):
            _adapt_environment_observation({"state": np.zeros((2, 25))}, 2, "state")


if __name__ == "__main__":
    unittest.main()
