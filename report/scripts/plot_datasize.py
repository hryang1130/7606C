"""Data-size curves for the report (figures/datasize_curves.pdf).

The numbers duplicate tables/datasize.tex; change both together. They are
transcribed from dp-manip/docs/results-summary.zh-CN.md and
dp-manip/scripts/plot_cross_task.py (PushCube per-seed values), not read from
run files.
Usage, from report/: python3 scripts/plot_datasize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def mean_sd(values):
    a = np.asarray(values, dtype=float) * 100
    return float(a.mean()), float(a.std(ddof=1))


# success_once (%) on held-out test seeds: N -> (mean, seed SD).
# Order fixes the colour and marker of each task.
CURVES = {
    "PullCube": {25: (76.7, 1.2), 50: (91.3, 0.6), 100: (95.6, 0.9), 200: (97.2, 0.4)},
    "PushCube": {
        25: mean_sd([0.49, 0.43, 0.44]),
        50: mean_sd([0.82, 0.81, 0.84]),
        100: mean_sd([0.96, 0.96, 0.94, 0.95, 0.93]),
        200: mean_sd([0.94, 0.97, 0.97, 0.98, 0.94]),
    },
    "PickCube": {25: (0.7, 1.2), 50: (0.7, 0.6), 100: (19.2, 1.8), 200: (45.2, 3.1), 400: (53.4, 2.1)},
    "StackCube": {25: (1.7, 1.2), 50: (12.3, 2.5), 100: (41.0, 2.9), 200: (66.0, 3.1), 400: (83.2, 1.9)},
    "PlaceSphere": {25: (4.7, 2.9), 50: (25.3, 4.1), 100: (36.0, 3.6), 200: (76.6, 5.6), 400: (96.2, 0.8)},
    "LiftPegUpright": {25: (7.0, 1.0), 50: (36.3, 3.1), 100: (67.4, 2.1), 200: (74.8, 2.8), 400: (79.2, 0.8)},
}

# Okabe-Ito colour-blind-safe palette (blue, vermillion, bluish green, orange,
# reddish purple, sky blue). Marker and dash pattern repeat the identity so the
# figure also reads in grayscale.
COLOR = dict(zip(CURVES, ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]))
MARKER = dict(zip(CURVES, ["o", "s", "^", "D", "v", "X"]))
DASH = dict(zip(CURVES, ["-", "-", "--", "-", "--", "-."]))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 2.7))
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    for task, curve in CURVES.items():
        ns = sorted(curve)
        mean = np.array([curve[n][0] for n in ns])
        sd = np.array([curve[n][1] for n in ns])
        ax.errorbar(ns, mean, yerr=sd, color=COLOR[task], lw=1.2, ls=DASH[task],
                    marker=MARKER[task], ms=4.5, mew=0, elinewidth=0.8, capsize=2,
                    capthick=0.8, label=task, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.minorticks_off()
    ax.set_xlim(21, 470)
    ax.set_ylim(-3, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Training demonstrations $N$ (log scale)")
    ax.set_ylabel("success_once (%)")

    # Legend ordered like the curves at their right ends, top to bottom.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = sorted(CURVES, key=lambda t: -CURVES[t][max(CURVES[t])][0])
    ax.legend([by_label[name] for name in order], order, loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8, handlelength=3.0, labelcolor=INK)

    out = Path(__file__).resolve().parent.parent / "figures" / "datasize_curves.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(out)


if __name__ == "__main__":
    main()
