"""High-resolution success and failure videos of closed-loop policy rollouts.

Recording happens in two passes so that it cannot change what the policy sees:

1. The evaluation runs in the unmodified evaluation environments
   (:func:`dp_manip.envs.make_eval_envs`) while :class:`StateRecorder` saves
   each step's simulation state (``get_state_dict``); nothing is rendered.
2. :func:`render_videos` replays one episode's saved states into a new
   environment whose human render camera (``render_camera``, the only one
   every task here defines) is resized to the requested resolution, and
   encodes the MP4. Callers run it in a freshly spawned process per episode.

These separations come from what we observed on macOS (MoltenVK): rendering
the human camera in the evaluation environment corrupted the next few sensor
captures, so the policy saw different images than in a plain evaluation; and
from the third scene a process built (``gym.make`` builds one, and evaluation
rebuilds the scene on every ``reset``), the render camera drew every textured
surface green. A worker that makes one environment and resets it once stays
within two scenes. Whether NVIDIA drivers do the same is untested, so none of
it can happen here by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    def __init__(self, path, width: int, height: int, fps: float, crf: int) -> None:
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


def video_environment_kwargs(
    cfg: Config, width: int, height: int, shader: str, render_backend: str | None = None
) -> dict[str, Any]:
    """The evaluation kwargs plus a render camera; physics and sensors are unchanged."""
    kwargs = environment_kwargs(cfg, render_backend)
    kwargs["render_mode"] = "rgb_array"
    kwargs["human_render_camera_configs"] = {"width": width, "height": height, "shader_pack": shader}
    return kwargs


def make_render_env(cfg: Config, width: int, height: int, shader: str, render_backend: str | None = None):
    """One environment for replaying saved states through the render camera."""
    if cfg.task.sim_backend != "physx_cpu":
        raise ValueError("replay requires physx_cpu, matching the evaluation")
    ensure_render_icd()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  registers environment IDs

    return gym.make(cfg.task.env_id, **video_environment_kwargs(cfg, width, height, shader, render_backend))


def video_name(task: str, split: str, seed: int, outcome: str) -> str:
    return f"{task}_{split}_seed{seed}_{outcome}.mp4"


def _copy_state(state: Any) -> Any:
    """Detach a (nested) state dict from the simulator; drop the controller's targets."""
    if isinstance(state, dict):
        return {key: _copy_state(value) for key, value in state.items() if key != "controller"}
    if hasattr(state, "detach"):
        return state.detach().cpu().clone()
    return np.array(state, copy=True)


class StateRecorder:
    """Rollout observer that keeps the states of the first ``quota[outcome]`` episodes of each outcome.

    It only reads simulation states, never renders, so the evaluation it
    observes is the same as an unobserved one. Evaluation stops after the first
    wave that fills both quotas.
    """

    def __init__(self, envs, *, quota: dict[str, int]) -> None:
        if set(quota) != set(OUTCOMES) or any(count < 0 for count in quota.values()):
            raise ValueError(f"quota needs non-negative counts for {OUTCOMES}")
        self.envs = envs
        self.quota = dict(quota)
        self.kept = {outcome: 0 for outcome in OUTCOMES}
        self.episodes: list[dict] = []
        self.kept_states: dict[int, list[dict]] = {}
        self._seeds: list[int] = []
        self._states: list[list[dict]] = []

    def _capture(self) -> list[dict]:
        states = list(self.envs.call("get_state_dict"))
        if len(states) != len(self._seeds):
            raise RuntimeError(f"expected {len(self._seeds)} states, got {len(states)}")
        return [_copy_state(state) for state in states]

    def on_reset(self, seeds: list[int], rgb: np.ndarray, proprio: np.ndarray) -> None:
        self._seeds = [int(seed) for seed in seeds]
        self._states = [[state] for state in self._capture()]

    def on_step(self, actions, rgb, proprio, reward, success) -> None:
        for history, state in zip(self._states, self._capture()):
            history.append(state)

    def on_wave_end(self, episodes: list[dict]) -> bool:
        if [episode["seed"] for episode in episodes] != self._seeds:
            raise RuntimeError("wave episodes do not match the recorded seeds")
        for episode, states in zip(episodes, self._states):
            outcome = "success" if episode["success_once"] else "failure"
            keep = self.kept[outcome] < self.quota[outcome]
            if keep:
                self.kept[outcome] += 1
                self.kept_states[episode["seed"]] = states
            self.episodes.append({**episode, "outcome": outcome, "kept": keep})
        self._seeds, self._states = [], []
        return self.done()

    def done(self) -> bool:
        return all(self.kept[outcome] >= self.quota[outcome] for outcome in OUTCOMES)


def _frame(rendered, shape: tuple[int, int, int]) -> np.ndarray:
    frame = rendered.cpu().numpy() if hasattr(rendered, "cpu") else np.asarray(rendered)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.shape != shape or frame.dtype != np.uint8:
        raise RuntimeError(f"render camera returned {frame.dtype} {frame.shape}, expected uint8 {shape}")
    return frame


def render_episode(
    env,
    seed: int,
    states: Sequence[dict],
    writer: FrameWriter,
    *,
    frame_shape: tuple[int, int, int],
    hold_frames: int,
) -> int:
    """Replay one episode's saved states through the render camera; return the frame count.

    ``reset(seed=...)`` rebuilds the episode's scene (objects and their
    randomized appearance) before the saved states are applied, because
    evaluation reconfigures the scene on every reset.
    """
    if not states:
        raise ValueError(f"seed {seed} has no recorded states")
    env.reset(seed=seed)
    frame = None
    try:
        for state in states:
            env.unwrapped.set_state_dict(state)
            frame = _frame(env.render(), frame_shape)
            writer.write(frame)
        assert frame is not None  # states is non-empty
        for _ in range(hold_frames):
            writer.write(frame)
    finally:
        writer.close()
    return len(states) + hold_frames


def render_videos(
    cfg: Config,
    jobs: Sequence[tuple[int, Sequence[dict], str]],
    output_dir: str,
    *,
    width: int,
    height: int,
    shader: str,
    fps: float | None,
    crf: int,
    hold_seconds: float,
    render_backend: str | None = None,
) -> dict:
    """Render ``(seed, states, file name)`` jobs into ``output_dir``.

    Run it in a fresh process with a single job, so the process builds only two
    scenes (see the module docstring).

    Each MP4 is written under a temporary name and renamed when complete, so an
    interrupted render never leaves a truncated video under its final name.
    """
    from pathlib import Path

    directory = Path(output_dir)
    env = make_render_env(cfg, width, height, shader, render_backend)
    try:
        fps = fps or float(env.unwrapped.control_freq)
        hold_frames = round(hold_seconds * fps)
        frames = {}
        for seed, states, name in jobs:
            partial = directory / f".partial-{name}"  # ffmpeg picks the container from the suffix
            try:
                frames[seed] = render_episode(
                    env,
                    seed,
                    states,
                    FfmpegWriter(partial, width, height, fps, crf),
                    frame_shape=(height, width, 3),
                    hold_frames=hold_frames,
                )
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
            partial.replace(directory / name)
            print(f"wrote {name}", flush=True)
    finally:
        env.close()
    return {"fps": fps, "frames": frames}


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

