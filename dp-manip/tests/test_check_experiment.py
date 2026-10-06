"""Phase 13-14 regression tests: Gate B is checked, not just documented.

The checker compares every resolved cell of an experiment matrix against one
reference and reports any difference outside the declared differences defined
in ``dp_manip.invariants``. It can also audit the configs that runs actually
recorded in ``run.json``. The declared data-size and backbone matrices must
pass their own check.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from dp_manip.config import default_run_name, load, load_experiment
from dp_manip.invariants import allowed_keys, config_differences, control_hash

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
EXPERIMENTS = ROOT / "configs" / "experiments"


def load_checker():
    path = ROOT / "scripts" / "check_experiment.py"
    spec = importlib.util.spec_from_file_location("check_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MatrixCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = load_checker()

    def cell(self, label: str, *overrides: str):
        return self.checker.Cell(label, load(TASKS / "pickcube.toml", list(overrides)))

    def test_replicate_seed_is_not_drift(self) -> None:
        cells = [self.cell("s1", "train.seed=1"), self.cell("s2", "train.seed=2")]
        self.assertEqual(self.checker.check_cells(cells, {"train.seed"}), [])

    def test_unexpected_key_is_reported_with_both_values(self) -> None:
        cells = [
            self.cell("unet s1", "train.seed=1"),
            self.cell("unet s2", "train.seed=2", "train.batch_size=128"),
        ]
        drifts = self.checker.check_cells(cells, {"train.seed"})
        self.assertEqual([drift.key for drift in drifts], ["train.batch_size"])
        self.assertEqual(drifts[0].reference_label, "unet s1")
        self.assertEqual(drifts[0].reference_value, 64)
        self.assertEqual(drifts[0].cell_label, "unet s2")
        self.assertEqual(drifts[0].cell_value, 128)

    def test_declared_matrices_have_no_drift(self) -> None:
        for name in ("data_size", "data_size_optional400", "backbone", "vision_pool", "control_mode"):
            experiment = EXPERIMENTS / f"{name}.toml"
            spec = load_experiment(experiment)
            allowed = allowed_keys(spec)
            for task_path in sorted(TASKS.glob("*.toml")):
                with self.subTest(experiment=name, task=task_path.stem):
                    cells = self.checker.matrix_cells(task_path, experiment, spec)
                    self.assertGreaterEqual(len(cells), 2)
                    self.assertEqual(self.checker.check_cells(cells, allowed), [])
                    hashes = {control_hash(cell.config, allowed) for cell in cells}
                    self.assertEqual(len(hashes), 1)

    def test_drift_report_matches_the_plan_format(self) -> None:
        cells = [
            self.cell("unet s1", "train.seed=1"),
            self.cell("transformer s1", "train.seed=1", "train.batch_size=128"),
        ]
        report = self.checker.format_drifts("pickcube", self.checker.check_cells(cells, set()))
        self.assertIn("ERROR: Unexpected experiment config drift in pickcube", report)
        self.assertIn("train.batch_size:", report)
        self.assertIn("reference unet s1 = 64", report)
        self.assertIn("transformer s1 = 128", report)

    def test_cli_checks_a_single_seed_and_rejects_empty_selections(self) -> None:
        checker = load_checker()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = checker.main(["--experiment", "backbone", "--task", "pickcube", "--seed", "1"])
        self.assertEqual(status, 0)
        self.assertIn("Gate B ok", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = checker.main(["--experiment", "data_size", "--task", "pickcube", "--seed", "5"])
        self.assertEqual(status, 0)
        self.assertIn("2 cells ok", output.getvalue())

        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            status = checker.main(["--experiment", "data_size", "--task", "pickcube", "--seed", "9"])
        self.assertEqual(status, 1)

    def test_cli_applies_set_overrides_to_declaration_and_recorded_runs(self) -> None:
        # The smoke workflow submits a reduced budget with --set; the checker
        # must accept the same overrides or every smoke run would look stale.
        checker = load_checker()
        experiment = EXPERIMENTS / "smoke.toml"
        spec = load_experiment(experiment)
        overrides = (
            "train.total_iters=200",
            "train.validation_steps=[200]",
            "train.checkpoint_steps=[200]",
        )
        set_args = [argument for item in overrides for argument in ("--set", item)]
        declared = checker.matrix_cells(
            TASKS / "pickcube.toml", experiment, spec, overrides=overrides
        )
        self.assertEqual([cell.config.train.total_iters for cell in declared], [200, 200, 200])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = checker.main(["--experiment", "smoke", "--task", "pickcube", *set_args])
        self.assertEqual(status, 0, output.getvalue())
        self.assertIn("Gate B ok", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cell in declared:
                run_dir = root / default_run_name(cell.config)
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(
                    json.dumps({"config": cell.config.to_dict()}), encoding="utf-8"
                )
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                status = checker.main(
                    ["--experiment", "smoke", "--task", "pickcube", "--run-root", str(root), *set_args]
                )
            self.assertEqual(status, 0, output.getvalue())
            self.assertIn("3 cells ok", output.getvalue())

    def test_run_root_checks_the_recorded_configs(self) -> None:
        checker = load_checker()
        spec = load_experiment(EXPERIMENTS / "backbone.toml")
        declared = checker.matrix_cells(TASKS / "pickcube.toml", EXPERIMENTS / "backbone.toml", spec, seed=1)

        def write_runs(root: Path, edits: dict[str, dict[str, dict]] = {}, skip: tuple[str, ...] = ()):
            for cell in declared:
                if cell.label in skip:
                    continue
                raw = cell.config.to_dict()
                raw["data"]["root"] = "/scratch/somewhere"  # runtime path, never drift
                for section, values in edits.get(cell.label, {}).items():
                    raw[section].update(values)
                run_dir = root / default_run_name(cell.config)
                run_dir.mkdir(parents=True)
                (run_dir / "run.json").write_text(json.dumps({"config": raw}), encoding="utf-8")

        def run(root: Path) -> tuple[int, str]:
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                status = checker.main(
                    ["--experiment", "backbone", "--task", "pickcube", "--seed", "1", "--run-root", str(root)]
                )
            return status, output.getvalue()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root)
            status, output = run(root)
            self.assertEqual(status, 0, output)
            self.assertIn("3 cells ok", output)

        # An OOM workaround on one arm is exactly the drift the declaration
        # check cannot see: it only exists in that run's run.json.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root, {"transformer s1": {"train": {"batch_size": 32}}})
            status, output = run(root)
            self.assertEqual(status, 1)
            self.assertIn("run.json differs from the declared config", output)
            self.assertIn("Unexpected experiment config drift", output)
            self.assertIn("transformer s1 run.json = 32", output)

        # A baseline changed after every run finished leaves the arms mutually
        # consistent but stale against the current declaration.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = {"train": {"lr": 3e-4}}
            write_runs(root, {label: stale for label in ("unet s1", "transformer s1", "mlp s1")})
            status, output = run(root)
            self.assertEqual(status, 1)
            self.assertIn("train.lr:", output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_runs(root, skip=("mlp s1",))
            status, output = run(root)
            self.assertEqual(status, 0, output)
            self.assertIn("2/3 cells have run.json; not run: mlp s1", output)

        with tempfile.TemporaryDirectory() as directory:
            status, _ = run(Path(directory))
            self.assertEqual(status, 1)

    def test_runtime_root_does_not_count_as_mismatch(self) -> None:
        # ``data.root`` is runtime metadata: a moved dataset root must not fail
        # the recorded-config check.
        left = load(TASKS / "pickcube.toml")
        right = load(TASKS / "pickcube.toml", ["data.root=/scratch/other"])
        differences = config_differences(left.to_dict(), right.to_dict())
        self.assertEqual(set(differences), {"data.root"})

    def test_cli_reports_unknown_task_and_experiment(self) -> None:
        checker = load_checker()
        with self.assertRaisesRegex(FileNotFoundError, "unknown task"):
            checker.main(["--experiment", "backbone", "--task", "banana"])
        with self.assertRaisesRegex(FileNotFoundError, "unknown experiment"):
            checker.main(["--experiment", "banana", "--task", "pickcube"])


if __name__ == "__main__":
    unittest.main()
