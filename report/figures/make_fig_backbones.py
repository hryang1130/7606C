"""Draw the three noise-prediction backbones of dp-manip side by side.

Deterministic schematic (not an AI-generated image). Read off
    dp_manip/conditional_unet1d.py, backbones/transformer.py, backbones/mlp.py
and configs/baseline.toml, instantiated with the PickCube N=100 shapes
    Tp=16, action_dim=4, To=2, Dobs=128+29=157
Backbone parameter counts are measured, not copied.

Run:  MPLCONFIGDIR=/tmp/mpl-cfg dp-manip/.venv/bin/python report/figures/make_fig_backbones.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().with_name("fig_networks.png")

C_ACT = ("#fde68a", "#b45309")
C_TIME = ("#e9d5ff", "#7e22ce")
C_OBS = ("#dbeafe", "#1d4ed8")
C_COND = ("#bbf7d0", "#15803d")
C_BLK = ("#f1f5f9", "#334155")
C_KEY = ("#fce7f3", "#be185d")
C_NOTE = ("#fee2e2", "#b91c1c")


def box(ax, x, y, w, h, text, kind=C_BLK, fs=7.2, weight="normal"):
    fc, ec = kind
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight=weight, zorder=3, linespacing=1.55, color="#0f172a")


def arrow(ax, p, q, color="#334155", lw=1.2, rad=0.0, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=10, linewidth=lw, color=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1,
    ))


def stack(ax, rows, top=0.892, bottom=0.018, gap=0.013, max_h=0.086, fs=6.9):
    """One vertical flow inside a panel. Rows are (text, kind, fontsize)."""
    n = len(rows)
    inner = top - bottom - gap * (n - 1)
    h = min(inner / n, max_h)
    x0, x1 = 0.035, 0.965
    y = top
    spans = []
    for text, kind, size in rows:
        box(ax, x0, y - h, x1 - x0, h, text, kind, size or fs)
        spans.append((y - h, y))
        y -= h + gap
    for i in range(n - 1):
        arrow(ax, ((x0 + x1) / 2, spans[i][0]), ((x0 + x1) / 2, spans[i + 1][1]))
    return spans


fig = plt.figure(figsize=(17.0, 16.0), dpi=150)
gs = fig.add_gridspec(3, 3, height_ratios=[0.185, 0.685, 0.130], hspace=0.10, wspace=0.055)

# ---------------------------------------------------------------------------
# (a) shared front end
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 1.00, "(a) shared front end — one visual encoder for all three arms   [config section: vision.*]",
        ha="center", va="top", fontsize=12, weight="bold")

box(ax, 0.010, 0.16, 0.100, 0.46, "rgb\n(B, 2, 3,\n128, 128)", C_OBS, 7.4)
box(ax, 0.120, 0.16, 0.130, 0.46, "ResNet-18\nBatchNorm → GroupNorm\nstride 32 · from scratch", C_OBS, 7.2)
box(ax, 0.260, 0.16, 0.085, 0.46, "layer4 map\n512 × 4 × 4", C_OBS, 7.4)
box(ax, 0.355, 0.07, 0.290, 0.64,
    "pool head   ← the variable of this experiment\n"
    "avg:   AdaptiveAvgPool2d(1) → 512 → Linear(512→128)\n"
    "ss32:  Conv(512→32, k1) → softmax over the 16 positions\n"
    "          → 32 · (x, y) ∈ [-1,1]² → Linear(64→128)", C_KEY, 7.4)
box(ax, 0.655, 0.16, 0.075, 0.46, "per-cam\nfeature\n128", C_OBS, 7.2)
box(ax, 0.740, 0.16, 0.100, 0.46, "proprio 29\n→ z-score\nclamp ±10", C_OBS, 7.2)
box(ax, 0.850, 0.16, 0.140, 0.46, "obs_features\n(B, To=2, Dobs=157)", C_COND, 7.8, "bold")
for p, q in (((0.110, 0.39), (0.120, 0.39)), ((0.250, 0.39), (0.260, 0.39)),
             ((0.345, 0.39), (0.355, 0.39)), ((0.645, 0.39), (0.655, 0.39)),
             ((0.730, 0.39), (0.740, 0.39)), ((0.840, 0.39), (0.850, 0.39))):
    arrow(ax, p, q)

ax_u, ax_t, ax_m = (fig.add_subplot(gs[1, i]) for i in range(3))
for a, title, sub in (
    (ax_u, "(b) 1D conditional U-Net",
     "CNN backbone from the paper §3.1 · 69.71M params\n3 conv levels · FiLM conditioning · one global cond vector"),
    (ax_t, "(c) time-series diffusion transformer",
     "DP-T lowdim configuration · 9.00M params\nTo condition tokens · causal attention over action tokens"),
    (ax_m, "(d) MLP noise predictor",
     "VariDP local baseline · 0.41M params\none concatenated vector · no temporal structure"),
):
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
    a.add_patch(FancyBboxPatch((0.002, 0.002), 0.996, 0.996,
                               boxstyle="round,pad=0.001,rounding_size=0.02",
                               linewidth=1.0, edgecolor="#cbd5e1", facecolor="white", zorder=0))
    a.text(0.5, 0.995, title, ha="center", va="top", fontsize=11, weight="bold")
    a.text(0.5, 0.952, sub, ha="center", va="top", fontsize=7.7, color="#475569", linespacing=1.5)

# ---------------------------------------------------------------------------
# (b) U-Net
# ---------------------------------------------------------------------------
stack(ax_u, [
    ("IN     noisy_actions (B,16,4)  →  transpose  →  (B,4,16)\n"
     "         channel = action dim, length = predicted horizon Tp", C_ACT, 6.9),
    ("COND   obs (B,2,157) → flatten 314\n"
     "         t: k → SinusoidalPosEmb(256) → 256→1024→256\n"
     "         cond = [ t_emb 256  ‖  obs 314 ]  =  570", C_COND, 6.9),
    ("DOWN0   4→256,  256→256,   ↓ Conv1d(256, k3, s2)     16→8", C_BLK, 6.9),
    ("DOWN1   256→512,  512→512,   ↓ Conv1d(512, k3, s2)    8→4", C_BLK, 6.9),
    ("DOWN2   512→1024,  1024→1024     (no ↓, stays 4)", C_BLK, 6.9),
    ("MID     ResBlock(1024→1024)  × 2", C_BLK, 6.9),
    ("UP0     cat(skip) → 2048→512,  512→512,   ↑ ConvT1d     4→8", C_BLK, 6.9),
    ("UP1     cat(skip) → 1024→256,  256→256    (no ↑, stays 8)", C_BLK, 6.9),
    ("OUT     Conv1dBlock(256,k5) → Conv1d(256→4,k1) → (B,16,4)", C_ACT, 6.9),
    ("ResBlock = Conv1d(in→out,k5) → GroupNorm(8) → Mish\n"
     "         → FiLM:  (scale, bias) = Linear(570, 2·out);   out ← scale ⊙ out + bias\n"
     "         → Conv1d(out→out,k5) → GroupNorm(8) → Mish → + residual Conv1d(in→out,k1)",
     C_KEY, 6.6),
    ("FiLM is the ONLY route from pixels to actions — the obs vector is\n"
     "never concatenated to the feature map and has no skip around the bottleneck",
     C_NOTE, 6.8),
], fs=6.9)

# ---------------------------------------------------------------------------
# (c) Transformer
# ---------------------------------------------------------------------------
stack(ax_t, [
    ("IN     noisy_actions (B,16,4) → Linear(4→256)\n"
     "          + learned pos_emb (1, 16, 256)", C_ACT, 6.9),
    ("MEM    time token:  k → SinusoidalPosEmb(256)\n"
     "          obs tokens:  (B,2,157) → Linear(157→256)\n"
     "          memory = [ time ‖ obs₀ ‖ obs₁ ] = 3 tokens × 256", C_COND, 6.9),
    ("MEM ENCODER   n_cond_layers = 0 → plain MLP\n"
     "          Linear(256→1024) → Mish → Linear(1024→256)", C_COND, 6.9),
    ("DECODER   8 × TransformerDecoderLayer\n"
     "          d_model 256 · 4 heads · FFN 1024 · dropout 0.3 · pre-LN", C_BLK, 6.9),
    ("EACH LAYER   ① causal self-attn over the 16 action tokens (tgt_mask)\n"
     "                    ② cross-attn to the 3 memory tokens (memory_mask)\n"
     "                    ③ position-wise FFN  256→1024→256", C_KEY, 6.7),
    ("OUT    LayerNorm(256) → Linear(256→4) → (B,16,4)", C_ACT, 6.9),
    ("causal mask: token t attends only to tokens ≤ t, so a longer\n"
     "horizon does not change the earlier steps — the MLP cannot do this",
     C_NOTE, 6.8),
    ("obs enters as 3 discrete memory slots; each action token decides\n"
     "by itself what to read from them", C_NOTE, 6.8),
], fs=6.9)

# ---------------------------------------------------------------------------
# (d) MLP
# ---------------------------------------------------------------------------
stack(ax_m, [
    ("IN     noisy_actions (B,16,4) → flatten 64\n"
     "          timestep k → SinusoidalPosEmb(128)", C_ACT, 6.9),
    ("OBS    (B,2,157) → flatten 314 → obs_mlp:\n"
     "          Linear(314→256) → Mish → LayerNorm → Linear(256→256)", C_OBS, 6.9),
    ("CAT    [ actions 64  ‖  t_emb 128  ‖  obs 256 ]  =  448", C_COND, 6.9),
    ("HIDDEN   3 × ( Linear(256) → Mish → LayerNorm(256) )", C_BLK, 6.9),
    ("OUT     Linear(256→64) → reshape (B,16,4)", C_ACT, 6.9),
    ("the 16 predicted steps are independent outputs of one 448-d\n"
     "vector — they share no weights and see nothing of each other", C_NOTE, 6.8),
    ("no convolution, no attention → no notion of which step is\n"
     "adjacent to which", C_NOTE, 6.8),
], fs=6.9)

# ---------------------------------------------------------------------------
# comparison table
# ---------------------------------------------------------------------------
ax = fig.add_subplot(gs[2, :])
ax.axis("off")
cols = ["arm", "backbone\nparams", "how obs_features enters  εθ", "temporal structure inside  εθ",
        "train 100k steps", "1 inference\n(100 DDPM, B=4)"]
rows = [
    ["U-Net  (b)", "69.71 M", "flatten → FiLM (scale, bias) inside every ResBlock",
     "3 conv down/up levels + 2 mid blocks", "6,195 s", "540 ms"],
    ["Transformer  (c)", "9.00 M", "To tokens → cross-attention in all 8 decoder blocks",
     "causal self-attention over 16 action tokens", "6,316 s", "570 ms"],
    ["MLP  (d)", "0.41 M", "concatenated once into a 448-d vector",
     "none — every step is an independent output", "2,670 s", "97 ms"],
]
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)
widths = [0.085, 0.075, 0.275, 0.245, 0.100, 0.105]
for (r, c), cell in table.get_celld().items():
    cell.set_width(widths[c])
    cell.set_edgecolor("#cbd5e1")
    if r == 0:
        cell.set_facecolor("#e2e8f0")
        cell.set_text_props(weight="bold")
    elif c == 0:
        cell.set_facecolor("#f8fafc")
        cell.set_text_props(weight="bold")
ax.text(0.5, -0.03, "shared ResNet-18 encoder = 11.24 M params (avg pool) for every arm  →  "
        "policy totals 80.96 M / 20.24 M / 11.65 M.   Timing on RTX 4080 SUPER, report_pickcube_yhr.md §5.",
        ha="center", va="top", fontsize=8.0, color="#475569")

fig.suptitle("Diffusion Policy backbones in dp-manip — PickCube N=100 shapes (Tp=16, Da=4, To=2, Dobs=157)",
             fontsize=13.5, weight="bold", y=0.997)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
