"""Rollout videos: labels, quotas, reproducibility and the render camera.

ManiSkill and ffmpeg are cluster-only, so a fake vector environment renders
frames that encode (seed, timestep) and a fake writer keeps them in memory.
"""

from __future__ import annotations

import importlib.util
import json
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
from dp_manip.rollout_video import VideoRecorder, reference_mismatches, video_environment_kwargs, video_name

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
        self.render_calls = 0

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
        assert name == "render"
        self.render_calls += 1
        frames = []
        for seed in self.seeds:
            frame = np.zeros(FRAME, dtype=np.uint8)
            frame[..., 0] = seed % 256
            frame[..., 1] = self.t
            frames.append(frame)
        return tuple(frames)


class MemoryWriter:
    """Writes (seed, timestep) of every frame to the file on close."""

    def __init__(self, path: Path):
        self.path = path
        self.frames: list[tuple[int, int]] = []
        path.write_bytes(b"")

    def write(self, frame):
        self.frames.append((int(frame[0, 0, 0]), int(frame[0, 0, 1])))

    def close(self):
        self.path.write_text(json.dumps(self.frames), encoding="utf-8")


def frames_of(path: Path) -> list[tuple[int, int]]:
    return [tuple(item) for item in json.loads(path.read_text(encoding="utf-8"))]


def run_waves(recorder: VideoRecorder, envs: FakeVectorEnv, seeds: list[int]) -> int:
    """Drive the observer the way ``evaluate`` does; return the number of waves run."""
    waves = 0
    for offset in range(0, len(seeds), envs.num_envs):
        chunk = seeds[offset : offset + envs.num_envs]
        envs.reset(chunk)
        recorder.on_reset(chunk, None, None)
        success_once = np.zeros(len(chunk), dtype=bool)
        for _ in range(MAX_STEPS):
            _, _, _, _, info = envs.step(None)
            success_once |= info["success"]
            recorder.on_step(None, None, None, None, info["success"])
        waves += 1
        episodes = [{"seed": seed, "success_once": bool(ok)} for seed, ok in zip(chunk, success_once)]
        if recorder.on_wave_end(episodes):
            break
    return waves


def make_recorder(envs, output_dir: Path, success: int, failure: int, hold_frames: int = 0) -> VideoRecorder:
    return VideoRecorder(
        envs,
        output_dir,
        task="placesphere",
        split="test",
        quota={"success": success, "failure": failure},
        frame_shape=FRAME,
        hold_frames=hold_frames,
        writer_factory=MemoryWriter,
    )


class VideoRecorderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "videos"
        self.output.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_keeps_first_episodes_of_each_outcome_and_labels_them_by_success_once(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = make_recorder(envs, self.output, success=2, failure=1)
        run_waves(recorder, envs, list(range(10000, 10010)))
        kept = sorted(path.name for path in self.output.glob("*.mp4"))
        # 10002 and 10005 are the first successes; 10000 is the first failure.
        self.assertEqual(
            kept,
            sorted(
                [
                    video_name("placesphere", "test", 10002, "success"),
                    video_name("placesphere", "test", 10005, "success"),
                    video_name("placesphere", "test", 10000, "failure"),
                ]
            ),
        )
        for episode in recorder.episodes:
            self.assertEqual(episode["outcome"] == "success", succeeds(episode["seed"]))
        self.assertFalse((self.output / ".partial").exists() and any((self.output / ".partial").iterdir()))

    def test_stops_after_the_wave_that_fills_both_quotas(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = make_recorder(envs, self.output, success=1, failure=1)
        waves = run_waves(recorder, envs, list(range(10000, 10010)))
        self.assertEqual(waves, 2)  # 10000/10001 fail, 10002 succeeds in the second wave
        self.assertEqual([episode["seed"] for episode in recorder.episodes], [10000, 10001, 10002, 10003])

    def test_video_holds_every_step_of_its_own_episode_then_the_last_frame(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = make_recorder(envs, self.output, success=1, failure=1, hold_frames=3)
        run_waves(recorder, envs, list(range(10002, 10004)))
        frames = frames_of(self.output / video_name("placesphere", "test", 10003, "failure"))
        expected = [(10003 % 256, t) for t in range(MAX_STEPS + 1)] + [(10003 % 256, MAX_STEPS)] * 3
        self.assertEqual(frames, expected)

    def test_reports_shortfall_without_failing_when_a_split_has_no_successes(self) -> None:
        envs = FakeVectorEnv(num_envs=2)
        recorder = make_recorder(envs, self.output, success=2, failure=1)
        run_waves(recorder, envs, [10001, 10004])
        self.assertEqual(recorder.kept, {"success": 0, "failure": 1})
        self.assertFalse(recorder.done())

    def test_rejects_a_render_camera_of_the_wrong_size(self) -> None:
        envs = FakeVectorEnv(num_envs=1)
        recorder = VideoRecorder(
            envs,
            self.output,
            task="placesphere",
            split="test",
            quota={"success": 1, "failure": 1},
            frame_shape=(1080, 1920, 3),
            hold_frames=0,
            writer_factory=MemoryWriter,
        )
        envs.reset([10000])
        with self.assertRaisesRegex(RuntimeError, "render camera returned"):
            recorder.on_reset([10000], None, None)
        recorder.cleanup()
        self.assertFalse((self.output / ".partial").exists())


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
        with tempfile.TemporaryDirectory() as tmp:
            envs = FakeVectorEnv(2)
            recorder = make_recorder(envs, Path(tmp), success=10, failure=10)
            recorded = evaluate(
                self.Policy(), envs, seeds, torch.device("cpu"), inference_seed=7, observer=recorder
            )
            recorder.cleanup()
            self.assertEqual(recorded["episodes"], plain["episodes"])
            self.assertEqual(len(list(Path(tmp).glob("*.mp4"))), len(seeds))
            self.assertEqual(reference_mismatches(recorder.episodes, plain["episodes"]), [])


if __name__ == "__main__":
    unittest.main()
