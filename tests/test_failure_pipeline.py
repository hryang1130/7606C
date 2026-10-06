"""``scripts/failure_pipeline.py``: the resumable driver of the whole study.

Runs the real pipeline on synthetic baselines with a fake environment and
checks its pause points (budget, gate, test), that a preempted run resumes by
resubmitting the same command, that finished stages are skipped, that the
rollout root keeps a smoke run out of the study's directories, and that a
smoke protocol can never write the study's lock file.
"""

from __future__ import annotations

import contextlib
import io
import os
import signal
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_failure_study import (  # noqa: E402
    HAVE_TORCH,
    MAX_STEPS,
    PROTOCOL,
    ROOT,
    TASK,
    FakeEnv,
    fake_envs,
    load_script,
    train_baselines,
)

from dp_manip import failure_study as study  # noqa: E402
from dp_manip.failure_lock import LockFile, dumps, lock_path  # noqa: E402


@unittest.skipUnless(HAVE_TORCH, "requires the cluster torch environment")
class FailurePipelineTest(unittest.TestCase):
    def test_pipeline_pauses_resumes_and_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.run_pipeline(Path(directory))

    def run_pipeline(self, root: Path) -> None:
        pipeline = load_script("failure_pipeline")
        run_root, rollout_root, lock = root / "runs", root / "rollouts", root / "lock.toml"
        protocol = root / "protocol.toml"
        protocol.write_text(PROTOCOL)
        train_baselines(root, run_root)
        base = [
            "--task", TASK, "--run-root", str(run_root), "--rollout-root", str(rollout_root),
            "--protocol", str(protocol), "--lock", str(lock), "--device", "cpu",
            "--finetune-set", "train.warmup_steps=1", "--max-episode-steps", str(MAX_STEPS),
        ]

        def run(*extra: str, factory=fake_envs) -> int:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pipeline.main([*base, *extra], envs_factory=factory)
            self.last_output = output.getvalue()
            return code

        def locked(section: str):
            return LockFile(lock).get(section)

        # 1. Budget pause after checkpoint 1's collection, with a projection.
        self.assertEqual(run(), 0)
        self.assertIn("PAUSED (budget", self.last_output)
        self.assertIn("projected total", self.last_output)
        self.assertTrue(locked("collection.s1") and not locked("collection.s2") and not locked("finetune"))
        self.assertTrue((rollout_root / "failure_aware" / TASK / "s1" / "raw_train.json").is_file())
        self.assertFalse((run_root / "failure_aware").exists())
        self.assertEqual(run(), 0)  # still paused, nothing re-collected
        self.assertNotIn("collect train", self.last_output)
        self.assertEqual(locked("task.max_episode_steps"), MAX_STEPS)
        with self.assertRaisesRegex(ValueError, "lock file's horizon is 12"):
            run("--max-episode-steps", str(MAX_STEPS + 1))

        # 2. Past the budget pause: pilot, remaining models and the offline gate.
        self.assertEqual(run("--budget-confirmed"), 0)
        self.assertTrue(locked("models.s1") and locked("models.s2") and locked("gate"))
        if not locked("gate.passed"):
            self.assertIn("STOPPED", self.last_output)
            self.assertEqual(run("--budget-confirmed"), 0)
            self.assertFalse(locked("guidance.dry_run"))
            # Synthetic data cannot pass the gate; continue as if it had.
            data = tomllib.loads(lock.read_text())
            data["gate"]["passed"] = True
            lock.write_text(dumps(data))

        # 3. A preemption during the dry run exits 75; resubmitting resumes.
        sent = []

        def preempting(cfg, num_envs, render_backend):
            if not sent:
                sent.append(True)
                os.kill(os.getpid(), signal.SIGUSR1)
            return FakeEnv(num_envs)

        self.assertEqual(run("--budget-confirmed", factory=preempting), 75)
        self.assertFalse(locked("guidance.dry_run"))
        self.assertEqual(run("--budget-confirmed"), 0)
        self.assertIn("PAUSED (test", self.last_output)
        self.assertTrue(locked("guidance.dry_run") and locked("guidance.selection") and locked("guidance.c1"))
        lock_file = LockFile(lock)
        for seed in (1, 2):
            self.assertEqual(list(study.rollout_dir(lock_file, seed).glob("eval/test/*.json")), [])

        # 4. The test split, then an idempotent resubmission.
        self.assertEqual(run("--budget-confirmed", "--include-test"), 0)
        self.assertIn("DONE", self.last_output)
        for seed in (1, 2):
            results = sorted(path.name for path in study.rollout_dir(lock_file, seed).glob("eval/test/*.json"))
            self.assertEqual(len(results), len(study.ARMS), results)
        self.assertEqual(run("--budget-confirmed", "--include-test"), 0)
        for stage in ("== s1:", "== s2:", "== pilot", "== gate", "== dry-run", "== tuning", "== select"):
            self.assertNotIn(stage, self.last_output)

    def test_a_smoke_protocol_never_uses_the_study_lock(self) -> None:
        pipeline = load_script("failure_pipeline")
        smoke = str(ROOT / "configs" / "failure_aware" / "smoke_protocol.toml")
        common = ["--task", TASK, "--run-root", "/nonexistent", "--protocol", smoke]
        with self.assertRaisesRegex(ValueError, "needs its own --lock"):
            pipeline.main(common)
        with self.assertRaisesRegex(ValueError, "study's lock file"):
            pipeline.main([*common, "--lock", str(lock_path(TASK))])
        # Nor the study's rollout directories under the run root.
        for rollout_root in ([], ["--rollout-root", "/nonexistent"]):
            with self.subTest(rollout_root=rollout_root), tempfile.TemporaryDirectory() as directory:
                lock = str(Path(directory) / "smoke.toml")
                with self.assertRaisesRegex(ValueError, "needs its own --rollout-root"):
                    pipeline.main([*common, "--lock", lock, *rollout_root])

    def test_smoke_protocol_fits_the_study_task(self) -> None:
        from dp_manip import config as config_lib
        from dp_manip.failure_protocol import load_protocol

        protocol = load_protocol(ROOT / "configs" / "failure_aware" / "smoke_protocol.toml")
        protocol.check_against(config_lib.load(ROOT / "configs" / "tasks" / f"{TASK}.toml"))
        self.assertEqual(protocol.baseline_cell.checkpoint_seeds, (1,))


if __name__ == "__main__":
    unittest.main()
