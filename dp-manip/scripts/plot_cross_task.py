"""Cross-task figures for the group summary (docs/results-summary.zh-CN.md).

Numbers are transcribed from the per-task reports in results/ and from
docs/experiment-progress.zh-CN.md; there are no run files to read here.
Usage: python3 scripts/plot_cross_task.py [--out docs/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TASK_COLOR = {  # categorical slots 1-5, fixed order
    "PushCube": "#2a78d6",
    "PickCube": "#eb6834",
    "StackCube": "#1baf7a",
    "PlaceSphere": "#eda100",
    "LiftPegUpright": "#e87ba4",
}
# Backbones get their own hues so they never read as a task colour.
# green/red sit in the CVD warn band, so marker shape is the secondary encoding.
BB_COLOR = {"UNet": "#4a3aa7", "Transformer": "#e34948", "MLP": "#008300"}
BB_MARKER = {"UNet": "o", "Transformer": "s", "MLP": "^"}
BACKBONES = ["UNet", "Transformer", "MLP"]

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#a32a2a", "#e34948", "#f0efec", "#3987e5", "#184f95"]  # worse ← 0 → better

plt.rcParams.update({
    "font.family": ["Hiragino Sans GB", "sans-serif"],
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


# ---------------------------------------------------------------- data
# Track A: UNet success_once on test seeds 10000-10099, mean / seed SD.
DATA_SIZE = {
    "PushCube": {  # per-seed values from task3.md
        25: mean_sd([0.49, 0.43, 0.44]),
        50: mean_sd([0.82, 0.81, 0.84]),
        100: mean_sd([0.96, 0.96, 0.94, 0.95, 0.93]),
        200: mean_sd([0.94, 0.97, 0.97, 0.98, 0.94]),
    },
    "LiftPegUpright": {25: (0.070, 0.010), 50: (0.363, 0.031), 100: (0.674, 0.021),
                       200: (0.748, 0.028), 400: (0.792, 0.008)},
    "StackCube": {25: (0.017, 0.012), 50: (0.123, 0.025), 100: (0.410, 0.029), 200: (0.660, 0.031)},
    "PlaceSphere": {25: (0.047, 0.029), 50: (0.253, 0.041), 100: (0.360, 0.036), 200: (0.766, 0.056),
                    400: (0.962, 0.008)},
    "PickCube": {25: (0.007, 0.012), 50: (0.007, 0.006), 100: (0.192, 0.018),
                 200: (0.452, 0.031), 400: (0.534, 0.021)},
}
PEG_VAL = {25: 0.0, 50: 0.0, 100: 0.0, 200: 0.008}  # val split, 50 episodes
HORIZON = {"PushCube": 100, "PickCube": 100, "StackCube": 200, "PlaceSphere": 200, "LiftPegUpright": 200}

# Track B: N=100. Tasks ordered easy → hard by the best backbone's success.
BB_TASKS = ["PushCube", "PickCube", "LiftPegUpright", "StackCube", "PegInsertionSide"]
BB_TICK = {"PushCube": "PushCube\n(易)", "PickCube": "PickCube", "LiftPegUpright": "LiftPeg-\nUpright",
           "StackCube": "StackCube", "PegInsertionSide": "Peg-\nInsertion(难)"}
BB_ONCE = {
    "PushCube": {"UNet": mean_sd([0.96, 0.96, 0.94, 0.95, 0.93]),
                 "Transformer": mean_sd([0.89, 0.90, 0.88, 0.89, 0.89]),
                 "MLP": mean_sd([0.95, 0.92, 0.92, 0.89, 0.93])},
    "PickCube": {"UNet": mean_sd([0.17, 0.18, 0.19, 0.21, 0.21]),
                 "Transformer": mean_sd([0.95, 0.92, 0.92, 0.84, 0.90]),
                 "MLP": mean_sd([0.81, 0.79, 0.82, 0.79, 0.77])},
    # task6_backbone: evaluated with a 400-step budget (demos run 153-702 steps).
    "LiftPegUpright": {"UNet": mean_sd([0.72, 0.71, 0.70, 0.74, 0.70]),
                       "Transformer": mean_sd([0.66, 0.59, 0.61, 0.74, 0.63]),
                       "MLP": mean_sd([0.51, 0.26, 0.45, 0.45, 0.38])},
    "StackCube": {"UNet": (0.410, 0.029), "Transformer": (0.418, 0.048), "MLP": (0.250, 0.025)},
    "PegInsertionSide": {"UNet": mean_sd([0, 0, 0, 0.01, 0]),
                         "Transformer": mean_sd([0, 0, 0, 0, 0.01]),
                         "MLP": (0.0, 0.0)},
}
BB_END = {  # PushCube reports means only
    "PushCube": {"UNet": (0.54, None), "Transformer": (0.63, None), "MLP": (0.67, None)},
    "PickCube": {"UNet": mean_sd([0.13, 0.13, 0.12, 0.14, 0.12]),
                 "Transformer": mean_sd([0.85, 0.76, 0.84, 0.74, 0.79]),
                 "MLP": mean_sd([0.74, 0.68, 0.69, 0.71, 0.65])},
    "LiftPegUpright": {"UNet": mean_sd([0.54, 0.41, 0.52, 0.66, 0.33]),
                       "Transformer": mean_sd([0.58, 0.52, 0.53, 0.64, 0.51]),
                       "MLP": mean_sd([0.05, 0.01, 0.08, 0.07, 0.05])},
    "StackCube": {"UNet": (0.356, 0.025), "Transformer": (0.340, 0.043), "MLP": (0.186, 0.017)},
    "PegInsertionSide": {"UNet": (0.0, None), "Transformer": (0.0, None), "MLP": (0.0, None)},
}
# Difference vs UNet and whether the report's test says the interval excludes 0.
BB_DELTA_SIG = {
    "PushCube": {"Transformer": True, "MLP": False},
    "PickCube": {"Transformer": True, "MLP": True},
    "LiftPegUpright": {"Transformer": False, "MLP": True},  # unpaired bootstrap, computed here
    "StackCube": {"Transformer": False, "MLP": True},
}


def n50(curve):
    """Demos needed to reach 0.5, by log2-linear interpolation; None if never."""
    ns = sorted(curve)
    for lo, hi in zip(ns, ns[1:]):
        p_lo, p_hi = curve[lo][0], curve[hi][0]
        if p_lo < 0.5 <= p_hi:
            return lo * 2 ** ((0.5 - p_lo) / (p_hi - p_lo))
    return None


# ---------------------------------------------------------------- figures
def fig_datasize(out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.grid(axis="y")
    ax.axhline(0.5, color=AXIS, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.text(25 * 0.93, 0.5, "50%", color=MUTED, fontsize=8, ha="right", va="center")

    peg_n = sorted(PEG_VAL)
    ax.plot(peg_n, [PEG_VAL[n] for n in peg_n], color=MUTED, lw=1.5, ls=(0, (2, 2)),
            marker="o", ms=4, label="PegInsertionSide（val，300 步）")
    ax.annotate("PegInsertionSide ≈ 0", (200, 0.008), xytext=(6, 4), textcoords="offset points",
                color=INK2, fontsize=8.5)

    order = sorted(DATA_SIZE, key=lambda t: n50(DATA_SIZE[t]) or 1e9)
    for task in order:
        curve = DATA_SIZE[task]
        ns = sorted(curve)
        m = np.array([curve[n][0] for n in ns])
        s = np.array([curve[n][1] for n in ns])
        c = TASK_COLOR[task]
        ax.fill_between(ns, m - s, m + s, color=c, alpha=0.12, lw=0)
        ax.plot(ns, m, color=c, lw=2, marker="o", ms=5.5, mec=SURFACE, mew=1.2,
                label=f"{task}（{HORIZON[task]} 步）", zorder=3)

    # Direct labels at line ends, nudged apart where they would collide.
    ends = {t: (max(DATA_SIZE[t]), DATA_SIZE[t][max(DATA_SIZE[t])][0]) for t in DATA_SIZE}
    nudge = {"PushCube": 0.025, "StackCube": -0.03, "PlaceSphere": 0.0, "LiftPegUpright": -0.01, "PickCube": -0.01}
    for t, (n, p) in ends.items():
        ax.annotate(t, (n, p + nudge[t]), xytext=(7, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks([25, 50, 100, 200, 400], ["25", "50", "100", "200", "400"])
    ax.set_xlim(21, 700)
    ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("训练示范数 N（log 刻度）")
    ax.set_ylabel("success_once（test，100 回合）")
    ax.set_title("UNet 数据量曲线：形状相近，难度决定曲线位置", loc="left")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=8, ncol=1, handlelength=2.2)
    fig.text(0.01, -0.02, "阴影 = 训练 seed 间 SD（N=25/50 为 3 seed，N≥100 为 5 seed）。"
             "各任务控制模式与回合长度不同，只比较形状与位置。", color=MUTED, fontsize=7.5)
    fig.savefig(out / "cross_task_datasize.png")
    plt.close(fig)


def _cell_text_color(rgba):
    r, g, b = rgba[:3]
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "#ffffff"


def fig_backbone_heatmap(out: Path):
    rows = ["PushCube", "PickCube", "LiftPegUpright", "StackCube", "PullCube"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"wspace": 0.6})
    seq = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    div = LinearSegmentedColormap.from_list("div", DIVERGING)
    seq_norm, div_norm = Normalize(0, 1), Normalize(-0.75, 0.75)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 3)
        ax.set_ylim(len(rows), 0)
        ax.set_xticks(np.arange(3) + 0.5, BACKBONES)
        ax.set_yticks(np.arange(len(rows)) + 0.5, rows)
        ax.tick_params(length=0)
        ax.xaxis.tick_top()
        for spine in ax.spines.values():
            spine.set_visible(False)

    for i, task in enumerate(rows):
        for j, bb in enumerate(BACKBONES):
            for ax in (ax1, ax2):
                if task == "PullCube":
                    ax.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=SURFACE,
                                               ec=AXIS, hatch="////", lw=0.8))
                    continue
            if task == "PullCube":
                continue
            m, s = BB_ONCE[task][bb]
            best = max(BB_ONCE[task][b][0] for b in BACKBONES)
            fc = seq(seq_norm(m))
            ax1.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=fc, lw=0))
            ax1.text(j + 0.5, i + 0.44, f"{m:.3f}", ha="center", va="center", color=_cell_text_color(fc),
                     fontsize=10.5, fontweight="bold" if m == best else "normal")
            ax1.text(j + 0.5, i + 0.72, f"±{s:.3f}", ha="center", va="center", color=_cell_text_color(fc),
                     fontsize=7.5)

            d = m - BB_ONCE[task]["UNet"][0]
            fc = div(div_norm(d)) if bb != "UNet" else div(div_norm(0))
            ax2.add_patch(Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, fc=fc, lw=0))
            if bb == "UNet":
                label = "基准"
            else:
                label = f"{d:+.3f}" + (" *" if BB_DELTA_SIG[task][bb] else "")
            ax2.text(j + 0.5, i + 0.5, label, ha="center", va="center", color=_cell_text_color(fc),
                     fontsize=10)
    for ax in (ax1, ax2):
        ax.text(1.5, rows.index("PullCube") + 0.5, "无报告", ha="center", va="center", color=INK2, fontsize=9,
                bbox=dict(fc=SURFACE, ec="none", pad=1.5))

    ax1.set_title("success_once（N=100，粗体 = 本任务最佳）", loc="left", pad=24)
    ax2.set_title("相对 UNet 的差值（* = 区间不含 0）", loc="left", pad=24)
    for ax, cmap, norm, ticks in ((ax1, seq, seq_norm, [0, 0.5, 1]), (ax2, div, div_norm, [-0.6, 0, 0.6])):
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.03, ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=MUTED, labelcolor=INK2, length=0)
    fig.text(0.01, -0.04, "行按难度排序（本任务最佳主干的成功率由高到低）。PegInsertionSide 三种主干均 ≈0，未列入。"
             "\n显著性：StackCube 为 z 检验，LiftPegUpright 为不配对的两层 bootstrap（评估 400 步），其余为配对两层 bootstrap。", color=MUTED, fontsize=7.5)
    fig.savefig(out / "cross_task_backbone_heatmap.png")
    plt.close(fig)


def fig_difficulty(out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"wspace": 0.22})
    x = np.arange(len(BB_TASKS))
    offset = {"UNet": -0.18, "Transformer": 0.0, "MLP": 0.18}

    panels = [
        (axes[0], BB_ONCE, "success_once", (-0.03, 1.02)),
        (axes[1], BB_END, "success_at_end", (-0.03, 1.02)),
    ]
    for ax, table, title, ylim in panels:
        ax.grid(axis="y")
        for bb in BACKBONES:
            m = np.array([table[t][bb][0] for t in BB_TASKS])
            s = np.array([table[t][bb][1] or 0 for t in BB_TASKS])
            xs = x + offset[bb]
            ax.plot(xs, m, color=BB_COLOR[bb], lw=1, alpha=0.45, zorder=1)
            ax.errorbar(xs, m, yerr=s, fmt=BB_MARKER[bb], color=BB_COLOR[bb], ms=7, mec=SURFACE,
                        mew=1.2, elinewidth=1.2, capsize=0, label=bb, zorder=3)
        ax.set_xticks(x, [BB_TICK[t] for t in BB_TASKS], fontsize=8.5)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("成功率（test，N=100）")
    axes[0].legend(loc="lower left", fontsize=8.5)

    # Relative to the best backbone on the same task; Peg is all ~0 so it is excluded.
    ax = axes[2]
    ax.grid(axis="y")
    rel_tasks = BB_TASKS[:4]
    xr = np.arange(len(rel_tasks))
    for bb in BACKBONES:
        rel = [BB_ONCE[t][bb][0] / max(BB_ONCE[t][b][0] for b in BACKBONES) for t in rel_tasks]
        ax.plot(xr, rel, color=BB_COLOR[bb], lw=2, marker=BB_MARKER[bb], ms=7, mec=SURFACE, mew=1.2, label=bb)
        ax.annotate(f"{bb} {rel[-1]:.2f}", (xr[-1], rel[-1]), xytext=(8, {"UNet": -9, "Transformer": 7, "MLP": 0}[bb]),
                    textcoords="offset points", color=INK, fontsize=8.5, va="center")
    ax.set_xticks(xr, [BB_TICK[t] for t in rel_tasks[:-1]] + ["StackCube\n(难)"], fontsize=8.5)
    ax.set_xlim(-0.3, 3.9)
    ax.set_ylim(0, 1.08)
    ax.set_title("相对本任务最佳主干（success_once）", loc="left")
    fig.text(0.01, -0.06, "误差棒 = 5 个训练 seed 的 SD（PushCube 与 PegInsertion 的 success_at_end 无 SD）。"
             "任务按最佳主干成功率排序作为难度代理；只有 4 个可区分的任务，趋势仅供参考。LiftPegUpright 评估 400 步。",
             color=MUTED, fontsize=7.5)
    fig.savefig(out / "cross_task_difficulty.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_datasize(args.out)
    fig_backbone_heatmap(args.out)
    fig_difficulty(args.out)
    for task, curve in DATA_SIZE.items():
        v = n50(curve)
        print(f"N50 {task:15s} {'never' if v is None else f'{v:.0f}'}")


if __name__ == "__main__":
    main()
