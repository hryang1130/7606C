"""High-resolution success and failure videos of closed-loop policy rollouts.

Videos come from ManiSkill's human render camera (``render_camera``, the only
one every task here defines), resized to the requested resolution. The policy
still sees its own 128x128 sensor camera: the render camera only adds
``render_mode`` and ``human_render_camera_configs`` to the evaluation
environment, so a recorded rollout is the same episode an unrecorded
evaluation with the same seeds, ``num_envs`` and inference seed produces.

Each episode is labelled by its own ``success_once``, the study's main metric.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Config
from .envs import ensure_render_icd, environment_kwargs

OUTCOMES = ("success", "failure")


class FrameWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class FfmpegWriter:
    """H.264 MP4 that PowerPoint, Keynote and browsers play without plugins."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        import imageio_ffmpeg  # ships with mani-skill (imageio[ffmpeg])

        self._stream = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            fps=fps,
            codec="libx264",
            quality=None,  # rate control comes from -crf below
            macro_block_size=1,  # keep 1080 rows; the default pads to 1088
            pix_fmt_out="yuv420p",
            output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"],
        )
        self._stream.send(None)

    def write(self, frame: np.ndarray) -> None:
        self._stream.send(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._stream.close()


WriterFactory = Callable[[Path], FrameWriter]


def video_environment_kwargs(
    cfg: Config, width: int, height: int, shader: str, render_backend: str | None = None
) -> dict[str, Any]:
    """The evaluation kwargs plus a render camera; physics and sensors are unchanged."""
    kwargs = environment_kwargs(cfg, render_backend)
    kwargs["render_mode"] = "rgb_array"
    kwargs["human_render_camera_configs"] = {"width": width, "height": height, "shader_pack": shader}
    return kwargs


def make_video_envs(
    cfg: Config,
    num_envs: int,
    width: int,
    height: int,
    shader: str,
    render_backend: str | None = None,
):
    """:func:`dp_manip.envs.make_eval_envs` with a high-resolution render camera."""
    if cfg.task.sim_backend != "physx_cpu":
        raise ValueError("fair evaluation requires physx_cpu, matching the generated data")
    ensure_render_icd()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers environment IDs
    from mani_skill.utils.wrappers import CPUGymWrapper
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    kwargs = video_environment_kwargs(cfg, width, height, shader, render_backend)

    def make():
        def thunk():
            env = gym.make(cfg.task.env_id, **kwargs)
            env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
            return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

        return thunk

    constructors = [make() for _ in range(num_envs)]
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(constructors)
    return gym.vector.AsyncVectorEnv(constructors, context="forkserver")


def video_name(task: str, split: str, seed: int, outcome: str) -> str:
    return f"{task}_{split}_seed{seed}_{outcome}.mp4"


class VideoRecorder:
    """Rollout observer that keeps the first ``quota[outcome]`` episodes of each outcome.

    Every episode of a wave is encoded to a partial file while it runs, because
    its outcome is only known when the wave ends; episodes beyond the quota are
    deleted. Evaluation stops after the first wave that fills both quotas.
    """

    def __init__(
        self,
        envs,
        output_dir: Path,
        *,
        task: str,
        split: str,
        quota: dict[str, int],
        frame_shape: tuple[int, int, int],
        hold_frames: int,
        writer_factory: WriterFactory,
    ) -> None:
        if set(quota) != set(OUTCOMES) or any(count < 0 for count in quota.values()):
            raise ValueError(f"quota needs non-negative counts for {OUTCOMES}")
        self.envs = envs
        self.output_dir = output_dir
        self.partial_dir = output_dir / ".partial"
        self.task = task
        self.split = split
        self.quota = dict(quota)
        self.frame_shape = frame_shape
        self.hold_frames = hold_frames
        self.writer_factory = writer_factory
        self.kept = {outcome: 0 for outcome in OUTCOMES}
        self.episodes: list[dict] = []
        self._seeds: list[int] = []
        self._writers: list[FrameWriter] = []
        self._last: list[np.ndarray] = []

    def _partial_path(self, seed: int) -> Path:
        return self.partial_dir / f"seed{seed}.mp4"

    def _render(self) -> list[np.ndarray]:
        frames = [np.asarray(frame) for frame in self.envs.call("render")]
        if len(frames) != len(self._writers):
            raise RuntimeError(f"expected {len(self._writers)} rendered frames, got {len(frames)}")
        for frame in frames:
            if frame.shape != self.frame_shape or frame.dtype != np.uint8:
                raise RuntimeError(
                    f"render camera returned {frame.dtype} {frame.shape}, expected uint8 {self.frame_shape}"
                )
        return frames

    def _write_frames(self) -> None:
        self._last = self._render()
        for writer, frame in zip(self._writers, self._last):
            writer.write(frame)

    def on_reset(self, seeds: list[int], rgb: np.ndarray, proprio: np.ndarray) -> None:
        self.partial_dir.mkdir(parents=True, exist_ok=True)
        self._seeds = [int(seed) for seed in seeds]
        self._writers = [self.writer_factory(self._partial_path(seed)) for seed in self._seeds]
        self._write_frames()

    def on_step(self, actions, rgb, proprio, reward, success) -> None:
        self._write_frames()

    def on_wave_end(self, episodes: list[dict]) -> bool:
        if [episode["seed"] for episode in episodes] != self._seeds:
            raise RuntimeError("wave episodes do not match the recorded seeds")
        for writer, frame in zip(self._writers, self._last):
            for _ in range(self.hold_frames):
                writer.write(frame)
            writer.close()
        for episode in episodes:
            outcome = "success" if episode["success_once"] else "failure"
            partial = self._partial_path(episode["seed"])
            video = None
            if self.kept[outcome] < self.quota[outcome]:
                video = video_name(self.task, self.split, episode["seed"], outcome)
                os.replace(partial, self.output_dir / video)
                self.kept[outcome] += 1
            else:
                partial.unlink()
            self.episodes.append({**episode, "outcome": outcome, "video": video})
        self._writers, self._last = [], []
        return self.done()

    def done(self) -> bool:
        return all(self.kept[outcome] >= self.quota[outcome] for outcome in OUTCOMES)

    def cleanup(self) -> None:
        """Close writers of an interrupted wave and drop the partial directory."""
        for writer in self._writers:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - best effort after a failure
                pass
        self._writers = []
        if self.partial_dir.is_dir():
            for path in self.partial_dir.iterdir():
                path.unlink()
            self.partial_dir.rmdir()


def reference_mismatches(recorded: Sequence[dict], reference: Sequence[dict]) -> list[dict]:
    """Episodes whose ``success_once`` differs from a saved evaluation of the same seeds."""
    by_seed = {int(episode["seed"]): episode for episode in reference}
    mismatches = []
    for episode in recorded:
        expected = by_seed.get(int(episode["seed"]))
        if expected is None:
            mismatches.append({"seed": episode["seed"], "reason": "seed missing from reference"})
        elif bool(expected["success_once"]) != bool(episode["success_once"]):
            mismatches.append(
                {
                    "seed": episode["seed"],
                    "reason": "success_once differs",
                    "recorded": bool(episode["success_once"]),
                    "reference": bool(expected["success_once"]),
                }
            )
    return mismatches
