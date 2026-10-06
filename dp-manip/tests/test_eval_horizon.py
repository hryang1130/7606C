"""Evaluation overrides preserve recorded training settings and old results."""
import importlib.util
import unittest
from pathlib import Path

from dp_manip.config import load
from dp_manip.envs import environment_kwargs
from dp_manip.runlist import eval_command, runs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_dp", ROOT / "scripts/eval_dp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvalHorizonTest(unittest.TestCase):
    def test_override_reaches_environment_without_changing_checkpoint(self):
        recorded = load(ROOT / "configs/tasks/placesphere.toml").to_dict()
        original = load(ROOT / "configs/tasks/placesphere.toml").to_dict()
        cfg = module.evaluation_config(recorded, 200)
        self.assertEqual(environment_kwargs(cfg)["max_episode_steps"], 200)
        self.assertEqual(recorded, original)
        cfg.task.max_episode_steps = original["task"]["max_episode_steps"]
        self.assertEqual(cfg.to_dict(), original)

    def test_output_names_preserve_old_results(self):
        checkpoint = ROOT / "runs/example/checkpoints/final.pt"
        self.assertEqual(module.result_path(checkpoint, "test", None).name, "test_final.json")
        self.assertEqual(module.result_path(checkpoint, "test", 200).name, "test_final_h200.json")
        self.assertEqual(module.result_path(checkpoint, "train", 200).name, "train_final_h200.json")

    def test_queue_command_keeps_train_budget(self):
        command = eval_command(runs(task="placesphere")[0], output_root=ROOT / "runs",
                               split="train", max_episode_steps=200)
        self.assertEqual(command[command.index("--episodes") + 1], "25")
        self.assertEqual(command[command.index("--max-episode-steps") + 1], "200")
        self.assertEqual(module.parse_args(["final.pt", "--max-episode-steps", "200"]).max_episode_steps, 200)
        with self.assertRaises(ValueError):
            module.evaluation_config(load(ROOT / "configs/tasks/placesphere.toml").to_dict(), 0)
