"""dp — 本项目可复用的 Diffusion Policy 库（本机 Windows 与学院 GPU Linux 共用）

  dp.backbones   三种噪声预测主干：MLP / 官方 ConditionalUnet1D / 官方 TransformerForDiffusion
  dp.dp_lib      DiffusionPolicy（DDPM 训练 + DDIM 采样）+ 数据集 + 归一化统计
  dp.utils       EMA、loss 曲线等训练小工具

入口脚本（不放在库里，避免"库依赖脚本目录"）：
  train_local/train.py, train_local/eval.py   本机 Windows（RTX 4060 Ti，physx_cpu 单环境评测）
  train/train.py,       train/eval.py         学院 GPU（Linux，RTX 4080，可并行仿真）
  tools/run_local.py                          一键：训练 → 评测 → 汇总
  tools/summarize_runs.py                     把某个 runs/ 汇总成 SUMMARY_all.md
"""
from . import backbones, dp_lib, utils  # noqa: F401

__all__ = ["backbones", "dp_lib", "utils"]
