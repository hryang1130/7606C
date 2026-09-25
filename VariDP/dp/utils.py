"""dp.utils — 训练通用小工具（EMA / loss 曲线）

原来 `train/train.py` 用 importlib 去加载 `train_local/train.py` 里的这两个函数，
现在抽到这里，两台机器的 train.py 都 import 同一份（避免再出现"改一处漏一处"）。
"""
from __future__ import annotations

import copy
import csv

import torch


class EMA:
    """指数滑动平均权重

    注意：这里是固定 decay；DP 官方用的是带 warmup 的动态 decay
    （`EMAModel(update_after_step=0, inv_gamma=1.0, power=0.75, min_value=0.0, max_value=0.9999)`）。
    eval.py 默认用 EMA 权重，`--use-raw` 可切回原始权重做对比。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for pe, pm in zip(self.ema.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(pm.detach(), alpha=1 - self.decay)
        for be, bm in zip(self.ema.buffers(), model.buffers()):
            be.copy_(bm)


def plot_curve(log_path: str, png_path: str, title: str):
    """把 log.csv 画成 loss + lr 双图。

    matplotlib 缺失/无显示环境时只警告不中断（两台机器都可能是无 Vulkan/无 X 的环境，
    所以强制 Agg 后端）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                        # 无 Vulkan / 无显示，必须 Agg
        import matplotlib.pyplot as plt
    except Exception as e:                           # 没装 matplotlib 就跳过
        print(f"[warn] 画图跳过: {e}")
        return
    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    ep = [int(r["epoch"]) for r in rows]
    tr = [float(r["train_loss"]) for r in rows]
    va = [(int(r["epoch"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, tr, label="train")
    if va:
        ax[0].plot([x for x, _ in va], [y for _, y in va], "o-", ms=3, label="val")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (log)")
    ax[0].set_title(f"{title}\ndenoising loss")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, [float(r["lr"]) for r in rows])
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("lr")
    ax[1].set_title("learning rate"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(png_path, dpi=130); plt.close()
    print(f"[ok] 曲线已保存 {png_path}")
