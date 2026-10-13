"""Failure-case strip for the report (figures/failure_cases.pdf).

One panel per task: the last simulated frame of a failed held-out test
episode, cropped around the object. The clips are not in the repository; they
are the rollout recordings the task owners made for the presentation
(scripts/record_rollout_videos.py in dp-manip), kept under
dp-manip/results/. PickCube, PushCube and PullCube come with a videos.json
that names the checkpoint (unet_n100_s1) and the episode outcome; PlaceSphere
is the baseline panel of the presentation clip (seed 10000, a failure of the
N=100 seed-1 baseline in failure-round2-review/diagnostics/s1_B.json). The
StackCube and LiftPegUpright clips were recorded with the same script, but
their videos.json (checkpoint, and the StackCube seed) was not kept.

Usage, from report/: python3 scripts/make_failure_figure.py <dp-manip/results>
Needs ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

T = "teammates"
# (label, clip, frame index, pre-scale width or None, crop x, y, size)
PANELS = [
    ("PickCube\nseed 10001",
     f"{T}/pickcube_video/test_final/pickcube_test_seed10001_failure.mp4", 100, None, 795, 400, 360),
    ("StackCube\nseed not logged",
     f"{T}/Stackcube_videos_JinQizheng/Task2_StackCube_Failure.mp4", 290, 1920, 620, 600, 360),
    ("PushCube\nseed 10043",
     f"{T}/pushcube_videos_wangbowen/pushcube_test_seed10043_failure.mp4", 100, None, 640, 660, 360),
    ("PullCube\nseed 10047",
     f"{T}/pullcube_videos_yangqihang/pullcube_videos_yangqihang/pullcube_test_seed10047_failure.mp4",
     100, None, 760, 600, 360),
    ("PlaceSphere\nseed 10000",
     "presentation-videos/05c_placesphere_success_vs_failure.mp4", 200, None, 1070, 154, 300),
    ("LiftPegUpright\nseed 10007",
     f"{T}/liftpegupright/liftpegupright/liftpegupright_test_seed10007_failure.mp4", 200, None, 500, 480, 520),
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "pdf.fonttype": 42,
})


def grab(clip: Path, frame: int, scale: int | None, x: int, y: int, size: int, out: Path) -> None:
    chain = [f"select='eq(n\\,{frame})'"]
    if scale:
        chain.append(f"scale={scale}:-1")
    chain += [f"crop={size}:{size}:{x}:{y}", "scale=360:360"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", ",".join(chain),
                    "-frames:v", "1", str(out)], check=True)


def main() -> None:
    results = Path(sys.argv[1])
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.5, 1.22))
    with tempfile.TemporaryDirectory() as tmp:
        for ax, (label, clip, frame, scale, x, y, size) in zip(axes, PANELS):
            png = Path(tmp) / "frame.png"
            grab(results / clip, frame, scale, x, y, size, png)
            ax.imshow(mpimg.imread(png))
            ax.set_title(label, fontsize=7, pad=3, linespacing=1.05)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#898781")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.78, wspace=0.05)
    out = Path(__file__).resolve().parent.parent / "figures" / "failure_cases.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.01)
    print(out)


if __name__ == "__main__":
    main()
