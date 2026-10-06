from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, from_dict, from_recorded, load, load_experiment, load_run, same_run


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs" / "baseline.toml"
TASKS = ROOT / "configs" / "tasks"


class LayeredConfigTest(unittest.TestCase):
    def test_merge_precedence(self) -> None:
        task_text = """
[task]
name = "precedence"
env_id = "PickCube-v1"
control_mode = "pd_ee_delta_pos"
max_episode_steps = 100

[data]
train_path = "train.h5"
val_path = "val.h5"
"""
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "task.toml"
            task.write_text(task_text, encoding="utf-8")
            experiment = Path(directory) / "experiment.toml"
            experiment.write_text(
                "[task]\nmax_episode_steps = 150\n\n[data]\nnum_demos = 300\n",
                encoding="utf-8",
            )
            with BASELINE.open("rb") as stream:
                baseline_demos = tomllib.load(stream)["data"]["num_demos"]
            resolved = load(task, baseline=BASELINE)
            self.assertEqual(resolved.data.num_demos, baseline_demos)
            self.assertEqual(resolved.task.max_episode_steps, 100)
            resolved = load(task, baseline=BASELINE, experiment=experiment)
            self.assertEqual(resolved.data.num_demos, 300)
            self.assertEqual(resolved.task.max_episode_steps, 150)
            resolved = load(
                task,
                ["data.num_demos=400"],
                baseline=BASELINE,
                experiment=experiment,
            )
            self.assertEqual(resolved.data.num_demos, 400)

    def test_task_layer_cannot_shadow_baseline(self) -> None:
        header = """
[task]
name = "drift"
env_id = "PickCube-v1"
control_mode = "pd_ee_delta_pos"
max_episode_steps = 100

[data]
train_path = "train.h5"
val_path = "val.h5"
"""
        cases = {
            "baseline section": header + "\n[train]\nbatch_size = 128\n",
            "baseline key in [data]": header + "num_demos = 200\n",
            "baseline key in [task]": header.replace(
                "max_episode_steps = 100", 'max_episode_steps = 100\nsim_backend = "physx_cuda"'
            ),
        }
        for label, text in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as directory:
                task = Path(directory) / "task.toml"
                task.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "baseline.toml"):
                    load(task, baseline=BASELINE)

    def test_task_configs_only_contain_task_specific_values(self) -> None:
        expected_task_keys = {"name", "env_id", "control_mode", "max_episode_steps"}
        expected_data_keys = {"train_path", "val_path"}
        for task_path in TASKS.glob("*.toml"):
            with self.subTest(task=task_path.name):
                with task_path.open("rb") as stream:
                    raw = tomllib.load(stream)
                self.assertEqual(set(raw), {"task", "data"})
                self.assertEqual(set(raw["task"]), expected_task_keys)
                self.assertEqual(set(raw["data"]), expected_data_keys)

        pick = load(TASKS / "pickcube.toml")
        stack = load(TASKS / "stackcube.toml")
        for section in ("vision", "policy", "train", "ema", "diffusion", "eval"):
            self.assertEqual(getattr(pick, section), getattr(stack, section))

    def test_legacy_entry_redirects_to_task_config(self) -> None:
        path = ROOT / "configs" / "pickcube_rgb.toml"
        with path.open("rb") as stream:
            self.assertEqual(set(tomllib.load(stream)), {"legacy"})
        self.assertEqual(load(path), load(TASKS / "pickcube.toml"))

    def test_new_tasks_reuse_baseline(self) -> None:
        baseline = load(TASKS / "pickcube.toml").to_dict()
        for name, env_id, horizon, mode in (
            ("placesphere", "PlaceSphere-v1", 200, "pd_ee_delta_pos"),
            ("liftpegupright", "LiftPegUpright-v1", 50, "pd_joint_pos"),
        ):
            with self.subTest(task=name):
                cfg = load(TASKS / f"{name}.toml")
                self.assertEqual(cfg.task.env_id, env_id)
                self.assertEqual(cfg.task.max_episode_steps, horizon)
                self.assertEqual(cfg.task.control_mode, mode)
                self.assertEqual(load(ROOT / "configs" / f"{name}_rgb.toml"), cfg)
                actual = cfg.to_dict()
                for split in ("train", "val"):
                    key = f"{split}_path"
                    self.assertEqual(
                        actual["data"].pop(key),
                        f"{split}/{env_id}/motionplanning/"
                        f"trajectory.state.{mode}.physx_cpu.h5",
                    )
                expected = {key: value for key, value in baseline.items() if key != "task"}
                expected["data"] = {
                    key: value for key, value in baseline["data"].items()
                    if key not in ("train_path", "val_path")
                }
                actual.pop("task")
                self.assertEqual(actual, expected)

    def test_num_demos_accepts_any_positive_integer(self) -> None:
        task = TASKS / "pickcube.toml"
        self.assertEqual(load(task, ["data.num_demos=37"]).data.num_demos, 37)
        for invalid in (0, -1, 2.5, True):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "must be positive"):
                value = str(invalid).lower() if isinstance(invalid, bool) else str(invalid)
                load(task, [f"data.num_demos={value}"])

    def test_data_size_experiment_definition_and_resolution(self) -> None:
        path = ROOT / "configs" / "experiments" / "data_size.toml"
        spec = load_experiment(path)
        self.assertEqual(spec.variable, "data.num_demos")
        self.assertEqual(spec.values, (25, 50, 100, 200))
        # The train-seed diagnostic must use seeds shared by every nested subset.
        self.assertEqual(spec.train_eval_episodes, min(spec.values))
        optional = load_experiment(ROOT / "configs" / "experiments" / "data_size_optional400.toml")
        self.assertEqual(optional.train_eval_episodes, spec.train_eval_episodes)
        resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value=50)
        self.assertEqual(resolved.data.num_demos, 50)

    def test_backbone_experiment_definition_and_resolution(self) -> None:
        path = ROOT / "configs" / "experiments" / "backbone.toml"
        spec = load_experiment(path)
        self.assertEqual(spec.name, "backbone")
        self.assertEqual(spec.variable, "policy.backbone")
        self.assertEqual(spec.values, ("unet", "transformer", "mlp"))
        for value in spec.values:
            # Track B trains every arm with seeds 1-5 (docs/final-plan.md §6).
            self.assertEqual(spec.seeds_for(value), (1, 2, 3, 4, 5))
        # The overfitting diagnostic reuses the first 25 seeds shared by the
        # data-size subsets and by both N_B choices.
        self.assertEqual(spec.train_eval_episodes, 25)

        with BASELINE.open("rb") as stream:
            baseline_demos = tomllib.load(stream)["data"]["num_demos"]
        normalized_arms = []
        for value in spec.values:
            resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value=value)
            self.assertEqual(resolved.policy.backbone, value)
            # N_B defaults to the canonical baseline size; hard tasks escalate
            # to 200 through the pre-registered rule, not through this file.
            self.assertEqual(resolved.data.num_demos, baseline_demos)
            # Gate B: normalizing the declared variable must make every arm
            # resolve to exactly the same config.
            arm = resolved.to_dict()
            arm["policy"]["backbone"] = spec.values[0]
            normalized_arms.append(arm)
        for arm in normalized_arms[1:]:
            self.assertEqual(arm, normalized_arms[0])

        with self.assertRaisesRegex(ValueError, "is not in experiment"):
            load(TASKS / "pickcube.toml", experiment=path, experiment_value="banana")

    def test_smoke_experiment_definition_and_resolution(self) -> None:
        path = ROOT / "configs" / "experiments" / "smoke.toml"
        spec = load_experiment(path)
        self.assertEqual(spec.name, "smoke")
        self.assertEqual(spec.variable, "policy.backbone")
        self.assertEqual(spec.values, ("unet", "transformer", "mlp"))
        for value in spec.values:
            self.assertEqual(spec.seeds_for(value), (1,))
        resolved = load(TASKS / "pickcube.toml", experiment=path, experiment_value="transformer")
        self.assertEqual(resolved.policy.backbone, "transformer")

    def test_optimizer_betas_resolve_from_baseline(self) -> None:
        resolved = load(TASKS / "pickcube.toml")
        self.assertEqual(resolved.train.betas, [0.95, 0.999])
        overridden = load(TASKS / "pickcube.toml", ["train.betas=[0.9,0.95]"])
        self.assertEqual(overridden.train.betas, [0.9, 0.95])

    def test_dual_trainer_data_loader_default_and_override(self) -> None:
        task = TASKS / "pickcube.toml"
        # The single 8-CPU job runs two trainers, so the default is 3 workers
        # each; the value stays a config override, never a hard-coded trainer
        # setting (plan §6.8).
        self.assertEqual(load(task).train.num_workers, 3)
        for value in (0, 2, 4):
            with self.subTest(num_workers=value):
                self.assertEqual(
                    load(task, [f"train.num_workers={value}"]).train.num_workers, value
                )

    def test_invalid_training_and_evaluation_values_are_rejected(self) -> None:
        cases = (
            ("train.lr=0.0", "train.lr"),
            ("train.weight_decay=-1e-6", "train.weight_decay"),
            ("train.grad_clip=0.0", "train.grad_clip"),
            ("train.num_workers=-1", "train.num_workers"),
            ("train.betas=[0.9]", "train.betas"),
            ("train.betas=[0.9, 1.0]", "train.betas"),
            ("eval.inference_seed=-1", "evaluation seeds"),
        )
        for override, message in cases:
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                load(TASKS / "pickcube.toml", [override])

    def test_experiment_value_requires_an_experiment_spec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "override.toml"
            override.write_text("[train]\nbatch_size = 8\n", encoding="utf-8")
            resolved = load(TASKS / "pickcube.toml", experiment=override)
            self.assertEqual(resolved.train.batch_size, 8)
            with self.assertRaisesRegex(ValueError, "requires an experiment spec"):
                load(TASKS / "pickcube.toml", experiment=override, experiment_value="50")

    def test_invalid_diagnostics_are_rejected(self) -> None:
        spec = """
[experiment]
name = "bad"
variable = "data.num_demos"
values = [10]

[replicates]
"10" = [1]

[diagnostics]
"""
        for body in ("train_eval_episodes = 0", "train_eval_episodes = true", "other = 1"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "experiment.toml"
                path.write_text(spec + body + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "diagnostics"):
                    load_experiment(path)

    def test_run_name_separates_backbone_arms(self) -> None:
        task = TASKS / "pickcube.toml"
        unet = load(task, ["data.num_demos=50", "train.seed=2"])
        # Existing UNet run directories keep their historical names.
        self.assertEqual(default_run_name(unet), "pickcube_rgb_unet_n50_s2")
        other = load(task, ["data.num_demos=50", "train.seed=2", 'policy.backbone="transformer"'])
        self.assertEqual(default_run_name(other), "pickcube_rgb_transformer_n50_s2")

    def test_recorded_configs_from_before_a_field_existed_still_load(self) -> None:
        current = load(TASKS / "pickcube.toml").to_dict()
        recorded = load(TASKS / "pickcube.toml").to_dict()
        # A Phase 6 run: no backbone selector, no backbone structure, no betas.
        del recorded["train"]["betas"]
        for key in [key for key in recorded["policy"] if key.startswith(("transformer_", "mlp_"))]:
            del recorded["policy"][key]
        del recorded["policy"]["backbone"]
        # Every run before data.preload existed read RGB lazily.
        del recorded["data"]["preload"]

        restored = from_recorded(recorded)
        # These are the values those runs actually trained with. They match
        # today's baseline except data.preload (true since Phase 21), which
        # never changes the samples, so an old resume.pt is still the same run.
        self.assertEqual(restored.train.betas, [0.95, 0.999])
        self.assertEqual(restored.policy.backbone, "unet")
        self.assertFalse(restored.data.preload)
        self.assertTrue(same_run(restored, load(TASKS / "pickcube.toml")))
        current["data"]["preload"] = False
        self.assertEqual(restored.to_dict(), current)

        # Fresh configs get no such help: baseline.toml is the only source.
        with self.assertRaisesRegex(ValueError, "missing .* required positional argument"):
            from_dict(recorded)

    def test_preload_is_runtime_metadata_not_a_control(self) -> None:
        from dp_manip.invariants import RUNTIME_KEYS, SEED_KEY, control_hash

        preloaded = load(TASKS / "pickcube.toml")
        lazy = load(TASKS / "pickcube.toml", ["data.preload=false"])
        self.assertTrue(preloaded.data.preload)
        self.assertFalse(lazy.data.preload)
        keys = {SEED_KEY, *RUNTIME_KEYS}
        self.assertEqual(control_hash(lazy, keys), control_hash(preloaded, keys))
        # A lazily recorded run is the same run as a preloaded invocation, but
        # any other difference still separates runs.
        self.assertTrue(same_run(lazy, preloaded))
        self.assertFalse(same_run(lazy, load(TASKS / "pickcube.toml", ["train.seed=2"])))
        # Runs recorded before the field existed keep their control hash.
        recorded = lazy.to_dict()
        del recorded["data"]["preload"]
        self.assertEqual(control_hash(from_recorded(recorded), keys), control_hash(lazy, keys))
        with self.assertRaisesRegex(ValueError, "data.preload"):
            load(TASKS / "pickcube.toml", ['data.preload="yes"'])

    def test_fresh_config_missing_a_baseline_value_is_rejected(self) -> None:
        text = BASELINE.read_text(encoding="utf-8")
        self.assertIn("betas = ", text)
        stripped = "\n".join(line for line in text.splitlines() if not line.startswith("betas = "))
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.toml"
            baseline.write_text(stripped + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "betas"):
                load(TASKS / "pickcube.toml", baseline=baseline)

    def test_experiment_variable_cannot_be_overridden(self) -> None:
        data_size = ROOT / "configs" / "experiments" / "data_size.toml"
        backbone = ROOT / "configs" / "experiments" / "backbone.toml"
        cases = (
            (data_size, 25, {"overrides": ["data.num_demos=4"]}),
            (data_size, 25, {"num_demos": 4}),
            (backbone, "mlp", {"overrides": ['policy.backbone="unet"']}),
        )
        for experiment, value, runtime in cases:
            with self.subTest(experiment=experiment.stem, runtime=runtime):
                with self.assertRaisesRegex(ValueError, "is the variable of experiment"):
                    load_run(
                        TASKS / "pickcube.toml",
                        runtime.get("overrides", ()),
                        experiment=experiment,
                        experiment_value=value,
                        num_demos=runtime.get("num_demos"),
                    )
        # Other keys stay overridable, e.g. N_B=200 for a hard backbone task.
        resolved = load_run(TASKS / "pickcube.toml", experiment=backbone, experiment_value="mlp", num_demos=200)
        self.assertEqual((resolved.policy.backbone, resolved.data.num_demos), ("mlp", 200))

    def test_load_run_applies_runtime_flags_after_overrides(self) -> None:
        resolved = load_run(
            TASKS / "pickcube.toml",
            ["train.seed=7", "data.num_demos=10"],
            num_demos=12,
            seed=3,
            data_root="/scratch/some dir/dataset",
        )
        self.assertEqual(resolved.train.seed, 3)
        self.assertEqual(resolved.data.num_demos, 12)
        self.assertEqual(resolved.data.root, "/scratch/some dir/dataset")

    def test_version_one_checkpoint_config_is_adapted(self) -> None:
        expected = load(TASKS / "pickcube.toml")
        legacy = expected.to_dict()
        legacy["train"]["ema_decay"] = legacy.pop("ema")["decay"]
        legacy["policy"].update(legacy.pop("diffusion"))
        self.assertEqual(from_dict(legacy), expected)


if __name__ == "__main__":
    unittest.main()
