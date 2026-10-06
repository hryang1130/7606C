"""Control-mode experiment: the action space as a declared variable.

Task data paths name their control mode as ``{control_mode}``, so changing
``task.control_mode`` also selects that mode's demonstrations. The task's own
mode resolves to the literal paths runs recorded before, Gate B treats the two
derived paths as part of the variable, and new modes get a run-name tag.
"""

from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from dp_manip import config as config_lib
from dp_manip.config import default_run_name, from_recorded, load, load_experiment
from dp_manip.invariants import allowed_keys, config_differences, control_hash

try:
    import h5py
    import numpy as np
    import torch

    from dp_manip.policy import DiffusionPolicy
    from dp_manip.trainer import run_training
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
EXPERIMENTS = ROOT / "configs" / "experiments"
EXPERIMENT = EXPERIMENTS / "control_mode.toml"
PEG = TASKS / "peginsertionside.toml"


def demo_path(split: str, env_id: str, mode: str) -> str:
    return f"{split}/{env_id}/motionplanning/trajectory.state.{mode}.physx_cpu.h5"


class ControlModePathTest(unittest.TestCase):
    def test_task_paths_resolve_to_their_recorded_literal_paths(self) -> None:
        for task_path in sorted(TASKS.glob("*.toml")):
            with self.subTest(task=task_path.stem):
                raw = tomllib.loads(task_path.read_text(encoding="utf-8"))
                self.assertIn("{control_mode}", raw["data"]["train_path"])
                self.assertIn("{control_mode}", raw["data"]["val_path"])
                cfg = load(task_path)
                self.assertEqual(
                    cfg.data.train_path, demo_path("train", cfg.task.env_id, cfg.task.control_mode)
                )
                self.assertEqual(cfg.data.val_path, demo_path("val", cfg.task.env_id, cfg.task.control_mode))

    def test_control_mode_override_selects_that_modes_data(self) -> None:
        cfg = load(PEG, ['task.control_mode="pd_ee_delta_pose"'])
        self.assertEqual(cfg.data.train_path, demo_path("train", "PegInsertionSide-v1", "pd_ee_delta_pose"))
        self.assertEqual(cfg.data.val_path, demo_path("val", "PegInsertionSide-v1", "pd_ee_delta_pose"))

    def test_explicit_path_override_still_wins(self) -> None:
        cfg = load(PEG, ['data.train_path="custom/train.h5"'])
        self.assertEqual(cfg.data.train_path, "custom/train.h5")
        self.assertEqual(cfg.data.val_path, demo_path("val", "PegInsertionSide-v1", "pd_joint_pos"))

    def test_unknown_placeholder_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown placeholder"):
            load(PEG, ['data.train_path="train/{env_id}/x.h5"'])

    def test_recorded_configs_keep_their_literal_paths(self) -> None:
        recorded = load(PEG).to_dict()
        self.assertNotIn("{", recorded["data"]["train_path"])
        self.assertEqual(from_recorded(recorded).to_dict(), recorded)

    def test_run_names(self) -> None:
        joint = load(PEG, ["data.num_demos=200", "train.seed=1"])
        self.assertEqual(default_run_name(joint), "peginsertionside_rgb_unet_n200_s1")
        ee = load(PEG, ["data.num_demos=200", "train.seed=1", 'task.control_mode="pd_ee_delta_pose"'])
        self.assertEqual(default_run_name(ee), "peginsertionside_rgb_unet_eepose_n200_s1")
        both = load(
            PEG,
            [
                "data.num_demos=200",
                "train.seed=1",
                'task.control_mode="pd_ee_delta_pose"',
                'vision.pool="spatial_softmax"',
            ],
        )
        self.assertEqual(default_run_name(both), "peginsertionside_rgb_unet_ss32_eepose_n200_s1")
        # Modes the task files already declare keep their historical names.
        pick = load(TASKS / "pickcube.toml", ["data.num_demos=50", "train.seed=2"])
        self.assertEqual(default_run_name(pick), "pickcube_rgb_unet_n50_s2")


class ControlModeExperimentTest(unittest.TestCase):
    def test_spec(self) -> None:
        spec = load_experiment(EXPERIMENT)
        self.assertEqual(spec.variable, "task.control_mode")
        self.assertEqual(spec.values, ("pd_joint_pos", "pd_ee_delta_pose"))
        self.assertEqual(spec.seeds_for("pd_joint_pos"), (1, 2, 3))
        self.assertEqual(spec.seeds_for("pd_ee_delta_pose"), (1, 2, 3))
        self.assertEqual(dict(spec.fixed), {"data.num_demos": 200})
        allowed = allowed_keys(spec)
        self.assertTrue({"data.train_path", "data.val_path"} <= allowed)
        # Elsewhere the data paths stay ordinary controls.
        for name in ("data_size", "backbone", "vision_pool"):
            other = allowed_keys(load_experiment(EXPERIMENTS / f"{name}.toml"))
            self.assertFalse({"data.train_path", "data.val_path"} & other, name)

    def test_arms_differ_only_in_control_mode_and_its_paths(self) -> None:
        spec = load_experiment(EXPERIMENT)
        joint = load(PEG, ["train.seed=1"], experiment=EXPERIMENT, experiment_value="pd_joint_pos")
        ee = load(PEG, ["train.seed=1"], experiment=EXPERIMENT, experiment_value="pd_ee_delta_pose")
        self.assertEqual(
            set(config_differences(joint.to_dict(), ee.to_dict())),
            {"task.control_mode", "data.train_path", "data.val_path"},
        )
        keys = allowed_keys(spec)
        self.assertEqual(control_hash(joint, keys), control_hash(ee, keys))
        self.assertEqual(ee.data.num_demos, 200)
        self.assertEqual(ee.task.max_episode_steps, joint.task.max_episode_steps)

    def test_joint_arm_is_the_data_size_n200_cell(self) -> None:
        for seed in (1, 2, 3):
            joint = load(PEG, [f"train.seed={seed}"], experiment=EXPERIMENT, experiment_value="pd_joint_pos")
            data_size = load(
                PEG, [f"train.seed={seed}"], experiment=EXPERIMENTS / "data_size.toml", experiment_value=200
            )
            self.assertEqual(joint.to_dict(), data_size.to_dict())
            self.assertEqual(default_run_name(joint), default_run_name(data_size))

    def test_fixed_data_size_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "fixed to 200"):
            config_lib.load_run(PEG, experiment=EXPERIMENT, experiment_value="pd_ee_delta_pose", num_demos=100)

    def test_the_variable_cannot_be_overridden(self) -> None:
        with self.assertRaisesRegex(ValueError, "variable of experiment"):
            load(
                PEG,
                ['task.control_mode="pd_joint_pos"'],
                experiment=EXPERIMENT,
                experiment_value="pd_ee_delta_pose",
            )


def write_dataset(root: Path, relative: str, seeds: list[int], *, control_mode: str, action_dim: int) -> None:
    rng = np.random.default_rng(0)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = []
    with h5py.File(path, "w") as file:
        for episode_id, seed in enumerate(seeds):
            group = file.create_group(f"traj_{episode_id}")
            group.create_dataset("obs_rgb/rgb", data=rng.integers(0, 256, (6, 64, 64, 3), dtype=np.uint8))
            group.create_dataset("obs_rgb/state", data=rng.normal(size=(6, 5)).astype(np.float32))
            group.create_dataset(
                "actions", data=rng.uniform(-1.0, 1.0, size=(5, action_dim)).astype(np.float32)
            )
            group.create_dataset("success", data=np.ones(6, dtype=bool))
            episodes.append({"episode_id": episode_id, "episode_seed": int(seed)})
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "env_info": {"env_id": "PegInsertionSide-v1", "env_kwargs": {"control_mode": control_mode}},
                "episodes": episodes,
            }
        ),
        encoding="utf-8",
    )


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class EndEffectorPoseTrainingTest(unittest.TestCase):
    def test_ee_delta_pose_trains_loads_and_samples_seven_dim_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            cfg = load(
                PEG,
                [
                    f"data.root={json.dumps(str(data_root))}",
                    'task.control_mode="pd_ee_delta_pose"',
                    "data.num_demos=4",
                    "data.val_num_demos=1",
                    "vision.feature_dim=8",
                    "policy.unet_dims=[16, 32]",
                    "policy.kernel_size=3",
                    "policy.n_groups=4",
                    "diffusion.num_diffusion_iters=4",
                    "diffusion.num_inference_iters=2",
                    "train.total_iters=2",
                    "train.batch_size=2",
                    "train.num_workers=0",
                    "train.log_freq=1",
                    "train.resume_freq=100",
                    "train.validation_steps=[2]",
                    "train.checkpoint_steps=[]",
                    "train.amp=false",
                    "ema.decay=0.9",
                ],
            )
            # Only the ee export exists, so training proves the derived path is read.
            write_dataset(data_root, cfg.data.train_path, [0, 1, 2, 3], control_mode="pd_ee_delta_pose", action_dim=7)
            write_dataset(data_root, cfg.data.val_path, [4000], control_mode="pd_ee_delta_pose", action_dim=7)
            output_root = root / "runs"
            self.assertEqual(run_training(cfg, output_root=output_root, device="cpu"), 0)

            run_dir = output_root / default_run_name(cfg)
            self.assertEqual(run_dir.name, "peginsertionside_rgb_unet_eepose_n4_s1")
            checkpoint = torch.load(run_dir / "checkpoints" / "final.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["task"]["control_mode"], "pd_ee_delta_pose")
            policy = DiffusionPolicy.from_checkpoint(checkpoint, "cpu").eval()
            rgb = torch.randint(0, 256, (2, policy.obs_horizon, 3, 64, 64), dtype=torch.uint8)
            actions = policy.get_action(
                rgb, torch.randn(2, policy.obs_horizon, 5), generator=torch.Generator().manual_seed(0)
            )
            self.assertEqual(actions.shape, (2, policy.act_horizon, 7))
            self.assertTrue(torch.isfinite(actions).all())

    def test_dataset_exported_in_another_mode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory) / "data"
            cfg = load(
                PEG,
                [
                    f"data.root={json.dumps(str(data_root))}",
                    'task.control_mode="pd_ee_delta_pose"',
                    "data.num_demos=1",
                    "data.val_num_demos=1",
                    "train.num_workers=0",
                ],
            )
            # A joint export placed at the ee path must not be trained on as ee data.
            write_dataset(data_root, cfg.data.train_path, [0], control_mode="pd_joint_pos", action_dim=8)
            write_dataset(data_root, cfg.data.val_path, [4000], control_mode="pd_joint_pos", action_dim=8)
            with self.assertRaisesRegex(ValueError, "control_mode"):
                run_training(cfg, output_root=Path(directory) / "runs", device="cpu")


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class InspectDatasetControlModeTest(unittest.TestCase):
    def load_script(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("inspect_dataset", ROOT / "scripts" / "inspect_dataset.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_set_selects_the_ee_export(self) -> None:
        import contextlib
        import io

        script = self.load_script()
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            ee = load(PEG, ['task.control_mode="pd_ee_delta_pose"'])
            write_dataset(data_root, ee.data.train_path, [0, 1, 2, 3], control_mode="pd_ee_delta_pose", action_dim=7)
            write_dataset(
                data_root, ee.data.val_path, list(range(4000, 4050)), control_mode="pd_ee_delta_pose", action_dim=7
            )
            argv = ["--config", str(PEG), "--data-root", str(data_root), "--num-demos", "4"]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                script.main([*argv, "--set", "task.control_mode=pd_ee_delta_pose"])
            self.assertIn("PASS PegInsertionSide-v1", output.getvalue())
            self.assertIn("action 7 (pd_ee_delta_pose)", output.getvalue())
            # Without the override the task's own (joint) export is looked up and is missing.
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                script.main(argv)


if __name__ == "__main__":
    unittest.main()
