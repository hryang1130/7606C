"""Rollout videos: labels, quotas, state replay and the render camera.

ManiSkill and ffmpeg are cluster-only, so a fake vector environment returns
states that encode (seed, timestep), a fake render environment draws them into
frames, and a fake writer keeps the frames in memory.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from dp_manip.evaluate import evaluate
except ModuleNotFoundError:  # torch is only installed in the cluster environment
    HAVE_TORCH = False
else:
    HAVE_TORCH = True

from dp_manip import config as config_lib
from dp_manip.envs import environment_kwargs
from dp_manip.rollout_video import (
    StateRecorder,
    reference_mismatches,
    render_episode,
    video_environment_kwargs,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "configs" / "tasks"
FRAME = (6, 8, 3)  # (height, width, channels) of the fake render camera
MAX_STEPS = 5


def load_script():
    path = ROOT / "scripts" / "record_rollout_videos.py"
    spec = importlib.util.spec_from_file_location("record_rollout_videos", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def succeeds(seed: int) -> bool:
    return seed % 3 == 0


class FakeVectorEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    def _observation(self) -> dict:
        rgb = np.zeros((self.num_envs, 4, 4, 3), dtype=np.uint8)
        state = np.zeros((self.num_envs, 2), dtype=np.float32)
        return {"rgb": rgb, "state": state}

    def reset(self, seed):
        self.seeds = list(seed)
        self.t = 0
        return self._observation(), {}

    def step(self, action):
        self.t += 1
        success = np.array([succeeds(seed) and self.t >= 2 for seed in self.seeds])
        truncated = np.full(self.num_envs, self.t >= MAX_STEPS)
        reward = np.zeros(self.num_envs)
        return self._observation(), reward, np.zeros(self.num_envs, bool), truncated, {"success": success}

    def call(self, name):
        assert name == "get_state_dict"
        return tuple(
            {"actors": {"cube": np.array([[seed, self.t]])}, "controller": {"arm": np.zeros(1)}}
            for seed in self.seeds
        )


class FakeRenderEnv:
    """Draws the (seed, timestep) of the state it was last given."""

    def __init__(self, shape=FRAME):
        self.shape = shape
        self.resets: list[int] = []
        self.unwrapped = self

    def reset(self, seed):
        self.resets.append(seed)
        self.state = None

    def set_state_dict(self, state):
        assert "controller" not in state
        self.state = state

    def render(self):
        seed, t = self.state["actors"]["cube"][0]
        frame = np.zeros((1, *self.shape), dtype=np.uint8)
        frame[..., 0] = seed % 256
        frame[..., 1] = t
        return frame


class MemoryWriter:
    def __init__(self):
        self.frames: list[tuple[int, int]] = []
        self.closed = False

    def write(self, frame):
        self.frames.append((int(frame[0, 0, 0]), int(frame[0, 0, 1])))

    def close(self):
        self.closed = True


def run_waves(recorder: StateRecorder, envs: FakeVectorEnv, seeds: list[int]) -> int:
    """Drive the observer the way ``evaluate`` does; return the number of waves run."""
    waves = 0
    empty = np.zeros(0)
    for offset in range(0, len(seeds), envs.num_envs):
        chunk = seeds[offset : offset + envs.num_envs]
        envs.reset(chunk)
        recorder.on_reset(chunk, empty, empty)
        success_once = np.zeros(len(chunk), dtype=bool)
        for _ in range(MAX_STEPS):
            _, _, _, _, info = envs.step(None)
            success_once |= info["success"]
            recorder.on_step(None, empty, empty, None, info["success"])
        waves += 1
        episodes = [{"seed": seed, "success_once": bool(ok)} for seed, ok in zip(chunk, success_once)]
        if recorder.on_wave_end(episodes):
            break
    return waves


class StateRecorderTest(unittest.TestCase):
    def test_keeps_first_episodes_of_each_outcome_and_labels_them_by_success_once(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = StateRecorder(envs, quota={"success": 2, "failure": 1})
        run_waves(recorder, envs, list(range(10000, 10010)))
        # 10002 and 10005 are the first successes; 10000 is the first failure.
        self.assertEqual(sorted(recorder.kept_states), [10000, 10002, 10005])
        self.assertEqual(recorder.kept, {"success": 2, "failure": 1})
        for episode in recorder.episodes:
            self.assertEqual(episode["outcome"] == "success", succeeds(episode["seed"]))
            self.assertEqual(episode["kept"], episode["seed"] in recorder.kept_states)

    def test_saves_every_state_of_its_own_episode_without_the_controller(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = StateRecorder(envs, quota={"success": 1, "failure": 1})
        run_waves(recorder, envs, [10002, 10003])
        for seed in (10002, 10003):
            states = recorder.kept_states[seed]
            self.assertEqual([tuple(state["actors"]["cube"][0]) for state in states], [(seed, t) for t in range(MAX_STEPS + 1)])
            self.assertTrue(all("controller" not in state for state in states))

    def test_stops_after_the_wave_that_fills_both_quotas(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = StateRecorder(envs, quota={"success": 1, "failure": 1})
        waves = run_waves(recorder, envs, list(range(10000, 10010)))
        self.assertEqual(waves, 2)  # 10000/10001 fail, 10002 succeeds in the second wave
        self.assertEqual([episode["seed"] for episode in recorder.episodes], [10000, 10001, 10002, 10003])

    def test_reports_shortfall_when_a_split_has_no_successes(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = StateRecorder(envs, quota={"success": 2, "failure": 1})
        run_waves(recorder, envs, [10001, 10004])
        self.assertEqual(recorder.kept, {"success": 0, "failure": 1})
        self.assertFalse(recorder.done())


class RenderEpisodeTest(unittest.TestCase):
    def states(self, seed: int) -> list[dict]:
        return [{"actors": {"cube": np.array([[seed, t]])}} for t in range(MAX_STEPS + 1)]

    def test_rebuilds_the_scene_then_renders_every_state_and_holds_the_last(self) -> None:
        env, writer = FakeRenderEnv(), MemoryWriter()
        count = render_episode(env, 10003, self.states(10003), writer, frame_shape=FRAME, hold_frames=3)
        self.assertEqual(env.resets, [10003])
        expected = [(10003 % 256, t) for t in range(MAX_STEPS + 1)] + [(10003 % 256, MAX_STEPS)] * 3
        self.assertEqual(writer.frames, expected)
        self.assertEqual(count, len(expected))
        self.assertTrue(writer.closed)

    def test_rejects_a_render_camera_of_the_wrong_size_and_closes_the_writer(self) -> None:
        env, writer = FakeRenderEnv(), MemoryWriter()
        with self.assertRaisesRegex(RuntimeError, "render camera returned"):
            render_episode(env, 10000, self.states(10000), writer, frame_shape=(1080, 1920, 3), hold_frames=0)
        self.assertTrue(writer.closed)


class ReferenceTest(unittest.TestCase):
    def test_flags_episodes_that_differ_from_the_saved_evaluation(self) -> None:
        recorded = [{"seed": 1, "success_once": True}, {"seed": 2, "success_once": False}, {"seed": 3, "success_once": True}]
        reference = [{"seed": 1, "success_once": True}, {"seed": 2, "success_once": True}]
        mismatches = reference_mismatches(recorded, reference)
        self.assertEqual([item["seed"] for item in mismatches], [2, 3])

    def test_finds_the_only_saved_evaluation_and_refuses_to_guess_between_horizons(self) -> None:
        script = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "checkpoints").mkdir(parents=True)
            (run / "eval").mkdir()
            checkpoint = run / "checkpoints" / "final.pt"
            with self.assertRaises(FileNotFoundError):
                script.find_reference(checkpoint, "test", None)
            (run / "eval" / "test_final.json").write_text("{}", encoding="utf-8")
            self.assertEqual(script.find_reference(checkpoint, "test", None).name, "test_final.json")
            (run / "eval" / "test_final_h200.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "several saved test evaluations"):
                script.find_reference(checkpoint, "test", None)
            self.assertEqual(script.find_reference(checkpoint, "test", 200).name, "test_final_h200.json")

    def test_horizon_comes_from_the_reference_and_must_agree_with_the_flag(self) -> None:
        script = load_script()
        self.assertEqual(script.resolve_horizon(None, {"max_episode_steps": 200}), 200)
        self.assertEqual(script.resolve_horizon(300, None), 300)
        self.assertIsNone(script.resolve_horizon(None, {}))
        with self.assertRaises(ValueError):
            script.resolve_horizon(100, {"max_episode_steps": 200})

    def test_defaults_are_1080p(self) -> None:
        args = load_script().parse_args(["final.pt"])
        self.assertEqual((args.width, args.height), (1920, 1080))


class RenderCameraTest(unittest.TestCase):
    def test_only_adds_the_render_camera_to_the_evaluation_environment(self) -> None:
        for task in ("pickcube", "stackcube", "pushcube", "pullcube", "peginsertionside", "plugcharger"):
            with self.subTest(task=task):
                cfg = config_lib.load(TASKS / f"{task}.toml")
                evaluation = environment_kwargs(cfg)
                video = video_environment_kwargs(cfg, 1920, 1080, "default")
                self.assertEqual({key: video[key] for key in evaluation}, evaluation)
                self.assertEqual(set(video) - set(evaluation), {"render_mode", "human_render_camera_configs"})
                self.assertEqual(
                    video["human_render_camera_configs"], {"width": 1920, "height": 1080, "shader_pack": "default"}
                )


@unittest.skipUnless(HAVE_TORCH, "torch is only installed in the cluster environment")
class EvaluateIntegrationTest(unittest.TestCase):
    class Policy:
        obs_horizon = 2
        training = False

        def eval(self):
            return self

        def train(self, mode: bool = True):
            return self

        def get_action(self, rgb, proprio, *, generator):
            return torch.randn((rgb.shape[0], 2, 1), generator=generator)

    def test_recording_does_not_change_evaluation_results(self) -> None:
        seeds = list(range(10000, 10008))
        plain = evaluate(self.Policy(), FakeVectorEnv(2), seeds, torch.device("cpu"), inference_seed=7)
        envs = FakeVectorEnv(2)
        recorder = StateRecorder(envs, quota={"success": 10, "failure": 10})
        recorded = evaluate(self.Policy(), envs, seeds, torch.device("cpu"), inference_seed=7, observer=recorder)
        self.assertEqual(recorded["episodes"], plain["episodes"])
        self.assertEqual(sorted(recorder.kept_states), seeds)
        self.assertEqual(reference_mismatches(recorder.episodes, plain["episodes"]), [])


if __name__ == "__main__":
    unittest.main()
