from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, load_experiment
from dp_manip.runlist import eval_command, train_command


ROOT = Path(__file__).resolve().parents[1]


def load_sweep_module():
    script = ROOT / "scripts" / "sweep.py"
    module_spec = importlib.util.spec_from_file_location("sweep", script)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


class SweepTest(unittest.TestCase):
    def test_sweep_cells_come_from_experiment_spec(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        declared = load_experiment(experiment)
        task_runs = [run for run in module.runs(experiment) if run.task == "pickcube"]
        actual = [(run.value, run.seed) for run in task_runs]
        expected = [
            (value, seed)
            for value in declared.values
            for seed in declared.seeds_for(value)
        ]
        self.assertEqual(actual, expected)
        self.assertTrue(all(run.config.parent.name == "tasks" for run in task_runs))

    def test_run_names_match_the_trainer_default(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        for run in module.runs(experiment):
            with self.subTest(run=run):
                resolved = run.resolve()
                self.assertEqual(run.name, default_run_name(resolved))
                self.assertEqual(
                    run.name,
                    f"{run.task}_rgb_{resolved.policy.backbone}_n{run.value}_s{run.seed}",
                )

    def test_backbone_grid_runs_through_the_sweep(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "backbone.toml"
        declared = load_experiment(experiment)
        grid = module.runs(experiment)
        self.assertEqual(
            len(grid),
            len(module.TASKS) * sum(len(declared.seeds_for(value)) for value in declared.values),
        )
        names = [run.name for run in grid]
        self.assertEqual(len(set(names)), len(names), "two backbone cells share a run directory")

        run = next(run for run in grid if run.value == "transformer")
        self.assertEqual(run.resolve().policy.backbone, "transformer")
        args = argparse.Namespace(action="train", output_root=ROOT / "runs", data_root=None)
        command = module.build_command(args, run)
        self.assertEqual(command[command.index("--experiment-value") + 1], "transformer")
        self.assertIn("_rgb_transformer_", run.name)

        # The unet arm is the data-size N=100 cell: same config, same directory,
        # so whichever array runs second finds final.pt and skips it.
        data_size = {run.name for run in module.runs(ROOT / "configs" / "experiments" / "data_size.toml")}
        unet_names = {run.name for run in grid if run.value == "unet"}
        self.assertTrue(unet_names <= data_size)

    def test_full_grid_sizes_are_stable(self) -> None:
        module = load_sweep_module()
        data_size = ROOT / "configs" / "experiments" / "data_size.toml"
        backbone = ROOT / "configs" / "experiments" / "backbone.toml"
        self.assertEqual(len(module.runs(data_size)), 128)
        self.assertEqual(len(module.runs(backbone)), 120)
        original_tasks = (
            "pickcube", "stackcube", "pushcube", "pullcube",
            "peginsertionside", "plugcharger",
        )
        self.assertEqual(module.TASKS[:6], original_tasks)
        for experiment, old_count in ((data_size, 96), (backbone, 90)):
            original_runs = [
                run for task in original_tasks for run in module.runs(experiment, task=task)
            ]
            self.assertEqual(module.runs(experiment)[:old_count], original_runs)

    def test_task_filter_keeps_the_declared_order(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        full = module.runs(experiment)
        filtered = module.runs(experiment, task="peginsertionside")
        self.assertEqual(len(filtered), 16)
        self.assertEqual(filtered, [run for run in full if run.task == "peginsertionside"])
        self.assertEqual(
            [(run.value, run.seed) for run in filtered],
            [
                (25, 1),
                (25, 2),
                (25, 3),
                (50, 1),
                (50, 2),
                (50, 3),
                (100, 1),
                (100, 2),
                (100, 3),
                (100, 4),
                (100, 5),
                (200, 1),
                (200, 2),
                (200, 3),
                (200, 4),
                (200, 5),
            ],
        )

    def test_task_filter_on_the_backbone_grid(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "backbone.toml"
        full = module.runs(experiment)
        filtered = module.runs(experiment, task="peginsertionside")
        self.assertEqual(len(filtered), 15)
        self.assertEqual(filtered, [run for run in full if run.task == "peginsertionside"])
        self.assertEqual(
            [(run.value, run.seed) for run in filtered],
            [
                (value, seed)
                for value in ("unet", "transformer", "mlp")
                for seed in (1, 2, 3, 4, 5)
            ],
        )

    def test_unknown_task_is_rejected_by_the_run_list(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        with self.assertRaisesRegex(ValueError, "unknown task"):
            module.runs(experiment, task="peginsertion")

    def test_cli_rejects_an_unknown_task_with_a_nonzero_exit(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "sweep.py"), "show", "--task", "peginsertion"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("peginsertion", result.stderr)

    def test_selected_run_indexes_inside_the_filtered_grid(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "backbone.toml"
        filtered = module.runs(experiment, task="peginsertionside")
        args = argparse.Namespace(experiment=experiment, task="peginsertionside", index=0)
        self.assertEqual(module.selected_run(args), filtered[0])
        # Without --task the index keeps counting through the full task-major grid.
        args = argparse.Namespace(experiment=experiment, task=None, index=64)
        self.assertEqual(module.selected_run(args), module.runs(experiment)[64])
        args = argparse.Namespace(experiment=experiment, task="peginsertionside", index=15)
        with self.assertRaisesRegex(ValueError, r"\[0, 14\]"):
            module.selected_run(args)

    def test_backbone_unet_cells_reuse_the_data_size_n100_runs(self) -> None:
        module = load_sweep_module()
        output_root = ROOT / "runs"
        n100 = {
            (run.task, run.seed): run
            for run in module.runs(ROOT / "configs" / "experiments" / "data_size.toml")
            if run.value == 100
        }
        unet = {
            (run.task, run.seed): run
            for run in module.runs(ROOT / "configs" / "experiments" / "backbone.toml")
            if run.value == "unet"
        }
        self.assertEqual(len(n100), len(module.TASKS) * 5)
        self.assertEqual(set(unet), set(n100))
        for key, run in unet.items():
            with self.subTest(task=key[0], seed=key[1]):
                reference = n100[key]
                self.assertEqual(run.name, reference.name)
                self.assertEqual(run.resolve().to_dict(), reference.resolve().to_dict())
                self.assertEqual(run.directory(output_root), reference.directory(output_root))

    def test_sweep_train_command_is_the_shared_train_command(self) -> None:
        module = load_sweep_module()
        run = module.runs(module.DEFAULT_EXPERIMENT, task="peginsertionside")[0]
        args = argparse.Namespace(
            action="train",
            experiment=run.experiment,
            output_root=ROOT / "runs",
            data_root=ROOT / "data",
        )
        self.assertEqual(
            module.build_command(args, run),
            train_command(run, output_root=ROOT / "runs", data_root=ROOT / "data"),
        )

    def test_sweep_eval_command_is_the_shared_eval_command(self) -> None:
        module = load_sweep_module()
        run = module.runs(module.DEFAULT_EXPERIMENT, task="peginsertionside")[0]
        args = argparse.Namespace(
            action="eval",
            experiment=run.experiment,
            output_root=ROOT / "runs",
            checkpoint="step_010000.pt",
            split="train",
            episodes=None,
            num_envs=4,
            render_backend="gpu",
        )
        self.assertEqual(
            module.build_command(args, run),
            eval_command(
                run,
                output_root=ROOT / "runs",
                checkpoint="step_010000.pt",
                split="train",
                episodes=None,
                num_envs=4,
                render_backend="gpu",
            ),
        )

    def test_train_split_eval_uses_experiment_diagnostic_budget(self) -> None:
        module = load_sweep_module()
        experiment = ROOT / "configs" / "experiments" / "data_size.toml"
        declared = load_experiment(experiment)
        run = module.runs(experiment)[-1]

        def eval_command(split: str, episodes: int | None) -> list[str]:
            args = argparse.Namespace(
                action="eval",
                experiment=experiment,
                output_root=ROOT / "runs",
                checkpoint="final.pt",
                split=split,
                episodes=episodes,
                num_envs=None,
                render_backend=None,
            )
            return module.build_command(args, run)

        def episodes_flag(command: list[str]) -> str | None:
            return command[command.index("--episodes") + 1] if "--episodes" in command else None

        self.assertGreater(run.value, declared.train_eval_episodes)
        self.assertEqual(episodes_flag(eval_command("train", None)), str(declared.train_eval_episodes))
        self.assertEqual(episodes_flag(eval_command("train", 3)), "3")
        self.assertIsNone(episodes_flag(eval_command("test", None)))


if __name__ == "__main__":
    unittest.main()
