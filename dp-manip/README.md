# dp-manip：集群 RGB Diffusion Policy

本目录是六个 ManiSkill 任务的 **RGB-based Diffusion Policy** 训练与评估工程。数据由
`maniskill-demogen` 生成；训练面向 Linux GPU 集群，用 Slurm 数组作业运行。

保留的研究问题是数据量：对每个任务使用同一训练池的嵌套子集
`25 ⊂ 50 ⊂ 100 ⊂ 200`，只有在 `100 → 200` 仍未饱和时才增加 `400`。不同训练种子
看到完全相同的前 N 条示范，因此种子间方差只反映优化随机性。

## 六个任务

| 配置 | 环境 | 控制模式 | 动作维 | 评估步数 |
| --- | --- | --- | ---: | ---: |
| `pickcube_rgb.toml` | PickCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `stackcube_rgb.toml` | StackCube-v1 | `pd_ee_delta_pos` | 4 | 200 |
| `pushcube_rgb.toml` | PushCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `pullcube_rgb.toml` | PullCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `peginsertionside_rgb.toml` | PegInsertionSide-v1 | `pd_joint_pos` | 8 | 300 |
| `plugcharger_rgb.toml` | PlugCharger-v1 | `pd_joint_pos` | 8 | 200 |

PegInsertionSide 与 PlugCharger 使用 `pd_joint_pos`，与 `maniskill-demogen/tasks.py` 的最终数据一致。
绝对关节目标会先按训练子集做 min-max 归一化，执行时还原，不裁剪到 `[-1, 1]`。

## 数据契约

每个 split 使用 `maniskill-demogen/data/dataset/` 下的文件：

```text
{train,val}/<Env>/motionplanning/trajectory.state.<control>.physx_cpu.h5
```

训练只读取：

```text
traj_i/obs_rgb/rgb    uint8   (T+1, 128, 128, 3*C)
traj_i/obs_rgb/state  float32 (T+1, P)
traj_i/actions        float32 (T, A)
```

`traj_i/obs` 是特权 state，RGB 策略不会读取。文件名里的 `.state.` 是为了兼容 ManiSkill 官方
示范布局，不能据此判断训练观测类型。

训练文件固定 400 条（种子池 `<4000`），验证示范固定 50 条（种子 `4000–4999`）；闭环验证
使用 `5000–5049`，正式测试使用 `10000–10099`。

## 模型与公平性

- 每个相机的 3 通道图像从 HDF5 的通道拼接中拆出；默认共享一套 GroupNorm ResNet-18。
- 每帧视觉特征与 `obs_rgb/state` 的非特权 proprioception 拼接，再把两帧历史输入条件 1D UNet。
- 动作预测/执行 horizon 为 `16/8`，DDPM 训练和推理均为 100 步。
- 所有任务、N 和训练种子固定 100k optimizer steps；RGB batch 默认为 64。
- state z-score 与 action min/max **只用当前 N 条训练示范**计算。
- 验证去噪 loss 使用独立的 50 条验证示范；主结果只用 `final.pt`，不按 loss 挑 checkpoint。
- 闭环评估固定 `physx_cpu`，与数据生成后端一致；策略推理仍在 CUDA 上。

## 集群快速开始

```bash
# 1. 登录节点建环境
./setup.sh

# 2. 在提交作业前检查六套数据（DATA_ROOT 指向 demogen 的 data/dataset）
.venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT"

# 3. 查看数组索引映射
.venv/bin/python scripts/sweep.py show --tier core

# 4. 提交 96 个核心训练；%4 表示最多同时跑 4 个，可按配额调整
sbatch --export=ALL,DATA_ROOT="$DATA_ROOT",RUN_ROOT=/scratch/$USER/dp-runs \
  slurm/train_array.sbatch

# 5. 全部 final.pt 完成后，固定测试种子做闭环评估
sbatch --export=ALL,RUN_ROOT=/scratch/$USER/dp-runs slurm/eval_array.sbatch
```

单次训练也可以直接运行：

```bash
.venv/bin/python scripts/train_dp.py \
  --config configs/pickcube_rgb.toml \
  --data-root "$DATA_ROOT" --num-demos 25 --seed 1
```

作业收到 Slurm 的 `USR1`/`TERM` 后会写 `checkpoints/resume.pt` 并以状态 75 退出；
`train_array.sbatch` 随后 requeue，同一数组项自动续训。已存在 `final.pt` 的数组项会直接成功退出。

## 产物

```text
runs/<task>_rgb_unet_n<N>_s<seed>/
  run.json
  metrics.jsonl
  summary.json
  checkpoints/
    step_010000.pt
    step_030000.pt
    step_060000.pt
    final.pt
    resume.pt
  eval/
    test_final.json
```

中间 checkpoint 用于过拟合/训练进程分析；正式表格只使用 `final.pt`。

## 代码导航

- `dp_manip/data.py`：demogen schema 校验、流式统计、HDF5 懒加载 temporal windows。
- `dp_manip/vision.py`：不依赖 torchvision 的 GroupNorm ResNet-18 与随机平移增强。
- `dp_manip/policy.py`：RGB 编码、归一化、条件 UNet、DDPM。
- `scripts/train_dp.py`：可恢复的集群训练入口。
- `scripts/eval_dp.py`：固定种子 RGB 闭环评估。
- `scripts/sweep.py`、`slurm/`：核心 96 组与条件 N=400 的数组作业。
- `PLAN.md`：数据量实验矩阵和运行口径。

旧的本地 state-based 试验记录保留在 `docs/`，只作历史参考；本 README、`PLAN.md` 和
`configs/*_rgb.toml` 是当前权威定义。
