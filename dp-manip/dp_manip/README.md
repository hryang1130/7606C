# dp_manip package

当前实现是 RGB Diffusion Policy：

- `data.py` 直接读取 `maniskill-demogen` 的 `obs_rgb/{rgb,state}`，图像懒加载；
- `vision.py` 用 GroupNorm ResNet-18 编码每个相机；
- `policy.py` 将视觉特征与非特权 proprioception 拼接，作为 1D Conditional UNet 的条件；
- `training.py` 提供 EMA、cosine warmup 与原子 checkpoint；
- `envs.py` / `evaluate.py` 负责固定 seed 的 RGB 闭环评估。

`conditional_unet1d.py` 源自 ManiSkill 官方 DP baseline（Apache-2.0，见
`LICENSE-ManiSkill`）。其余训练结构已针对 RGB 数据与 Slurm 集群重写。
