# 实验三（Track 3: Simulation Experiments）完整教程

> 目标：在最少 6 个不同的 ManiSkill 任务上**自己采集/生成专家演示**，训练并评测 Diffusion Policy（DP）基线，并围绕一个研究问题做受控实验。
> 本教程按“照着做就能出结果”的顺序组织，所有命令都适配**你本机已装好的 venv 环境**。

> ⚠️ **代码真源提示（文档更新：2026-09-25）**
>
> 本文已按"不重复仓库代码"精简（**1776 行 → 现在的体量**）：
> ① `dp_lib.py` / `train_dp.py` / `eval_dp.py` 三个**历史版本（MLP-only）代码块**已删除（仓库里有真源：`dp/` 与 `train_local/`）；
> ② 文内那些一次性脚本正文**已落成真文件**，放在 `scripts/`（`fetch_demos.py` / `convert_all.py` / `gen_demos_scripted.py` / `check_datasets.py` / `fetch_ms_baseline.py` / `eval_all.py`），本文只留结论与用法。
> **与仓库不一致时一律以仓库代码为准**；被删内容的原始全文备份在 `.ref/教程_v0_full.md`。
>
> | 教程里的代码块 | 现在的真源 |
> |---|---|
> | ~~6.3 的 `dp_lib.py`~~（已删） | `dp/dp_lib.py`（+ `dp/backbones.py` 三种主干 + `dp/utils.py`） |
> | ~~7.1 的 `train_dp.py`~~（已删） | `train_local/train.py`（本机）/ `train/train.py`（学院 GPU） |
> | ~~8.1 的 `eval_dp.py`~~（已删） | `train_local/eval.py` / `train/eval.py`（后者支持并行仿真） |
> | 7.2 / 8.2 的批量脚本 | `tools/run_local.py`（一键）/ `tools/summarize_runs.py`（汇总）/ `scripts/eval_all.py`（批量评测） |
> | 4 / 5 / 6.5 的演示生成与质检脚本 | `scripts/`（`fetch_demos.py` / `convert_all.py` / `gen_demos_scripted.py` / `check_datasets.py` / `fetch_ms_baseline.py`） |
>
> 主干结构在 2026-09-25 已升级为**官方实现的完整移植**：UNet = `ConditionalUnet1D`（66.418 M），
> Transformer = `TransformerForDiffusion`（DP-T，8.972 M），MLP 基线不变（0.353 M）。详见 `train_local/README.md` 第 6 节。

---

## 目录

- [0. 先读：本机环境的 10 个硬约束（决定整个技术方案）](#0-先读本机环境的-10-个硬约束决定整个技术方案)
- [1. 实验三到底要什么（对照评分表）](#1-实验三到底要什么对照评分表)
- [2. 总体路线图与里程碑](#2-总体路线图与里程碑)
- [3. 第 1 步：六个任务选型](#3-第-1-步六个任务选型)
- [4. 第 2 步：生成专家演示](#4-第-2-步生成专家演示)
- [5. 第 3 步：数据集格式与质量检查](#5-第-3-步数据集格式与质量检查)
- [6. 第 4 步：Diffusion Policy 实现](#6-第-4-步diffusion-policy-实现)
- [7. 第 5 步：逐任务训练](#7-第-5-步逐任务训练)
- [8. 第 6 步：评测（held-out seeds）](#8-第-6-步评测held-out-seeds)
- [9. 第 7 步：深入研究实验（研究问题）](#9-第-7-步深入研究实验研究问题)
- [10. 第 8 步：交付物与报告结构](#10-第-8-步交付物与报告结构)
- [11. 踩坑速查表](#11-踩坑速查表)
- [12. 建议排期](#12-建议排期)
- [附录 A：DP 网络结构对比实验（三种主干）](#附录-adp-网络结构对比实验三种主干)

---

## 0. 先读：本机环境的 10 个硬约束（决定整个技术方案）

这 10 条是实测结论，直接决定你能走哪条路，**先看这一节能省你两天时间**。

| # | 约束 | 实测结论 | 对你的影响 |
|---|---|---|---|
| 1 | `mplib` 不可用 | ManiSkill 官方运动规划解法 `mani_skill.examples.motionplanning.panda.run` 依赖 `mplib`，而 PyPI 上 `mplib` 只有 **源码包**（需编译 FCL/OMPL），Windows 基本装不上 | **官方“本地跑运动规划生成演示”这条路走不通** |
| 2 | `pinocchio` 已可用 | 已把 conda-forge 的 `pinocchio 4.1.0` 移植进 venv（36 个 DLL + `_pinocchio_dlls.pth`） | ✅ 控制模式转换（`pd_joint_pos → pd_ee_delta_pos`）可正常工作 |
| 3 | 无 Vulkan 驱动 | 机器上没有任何 Vulkan ICD（注册表无 `Khronos\Vulkan\Drivers`，无 `nvoglv64.dll`） | `--save-video` / `env.render()` **必崩**，不要用；要用视频就另外装显卡驱动自带的 Vulkan Runtime |
| 4 | `huggingface.co` 被代理拦 | urllib 直连报 `Tunnel connection failed: 502` | 用 **hf-mirror.com** 镜像下载，且**必须带 User-Agent**（否则 403） |
| 5 | `max_episode_steps` 默认只有 50 | 桌面任务的注册值是 50 步，但演示轨迹通常要 74~120 步 | **生成与评测都必须显式传 `max_episode_steps`**，否则轨迹会被硬截断 |
| 6 | 机器人底座不在世界原点 | PickCube 里 panda 根节点在 `x = -0.615` | 自己写控制器时**必须做 world → base 坐标变换**，直接用世界坐标当动作会完全跑偏 |
| 7 | pip 包里没有 DP 基线 | `pip install mani_skill` 只装 `mani_skill/` 库；`examples/` 下只有 `benchmarking` / `motionplanning` / `teleoperation`，**没有 `baselines/`** | 官方 DP 基线代码得从 GitHub 仓库取，见 [第 6.5 节](#65-推荐直接用-maniskill-官方-dp-基线) |
| 8 | `github.com` 不通 | `github.com` 与 `raw.githubusercontent.com` 均 SSL handshake timeout；项目根 `ManiSkill/.git`（26 MB，无分支、HEAD 无法解析、无工作树）就是 clone 中断的残骸 | 用 `cdn.jsdelivr.net` 或 `gh-proxy.com` 取单文件（已验证可用） |
| 9 | 官方 DP 依赖缺 2 个 | `diffusers` ✅ / `tensorboard` ❌ / `wandb` ❌ | `pip install tensorboard`；wandb 用 `--no-track` 绕过 |
| 10 | **GPU 并行仿真不可用** | RTX 4060 Ti（8 GB）CUDA 正常，但 `--sim-backend physx_cuda` 报 `Could not find module 'cuda.dll'` —— 本机没装 CUDA Toolkit（只有 `System32\nvcuda.dll`） | 训练走 GPU 没问题；**评测只能用 CPU 单环境仿真**（`--sim-backend physx_cpu`，`num_envs=1`），见 [7.3 节](#73-分两阶段跑本地-4060-先通一个任务再上集群) |

补充：`replay_trajectory` 在 ManiSkill 3.0.1 里已把 `--num-procs` **改名为 `-n/--num-envs`**，老教程里的 `--num-procs` 会直接报错。

> 上表是**本机 Windows** 的约束。学院 GPU（Linux, RTX 4080）的差异见 [`学院_uv安装指南.md`](学院_uv安装指南.md) 附录 C：
> pinocchio 轮子自带（不用移植）、`physx_cuda` 并行仿真**可能可用**（评测可 `--num-envs 32`）、无外网时用镜像源装依赖。

**关键环境路径（后面命令都用它）**

```text
项目根目录   <项目根目录>                              （本仓库所在文件夹，放哪都行）
Python       <项目根目录>\venv\Scripts\python.exe      （本机 venv；Linux 上用 .venv/bin/python）
演示数据根   %USERPROFILE%\.maniskill\demos            （即 mani_skill.DEMO_DIR，在项目目录之外）
```

为方便，在**项目根目录下**先设一个快捷（CMD；PowerShell 写 `$PY = ".\venv\Scripts\python.exe"`）：

```bat
set PY=%CD%\venv\Scripts\python.exe
```

> 🔁 **改过项目目录名 / 搬过家**：跑一次 `python tools\fix_venv_paths.py --apply`（修 venv 里写死的绝对路径，见
> `train_local/README.md` §11），再在 PyCharm 里重新指定一次解释器即可。

---

## 1. 实验三到底要什么（对照评分表）

| 项 | 分值 | 具体要求 | 踩分点 |
|---|---|---|---|
| Task coverage & demonstration generation | 20 | ≥6 个**不同**任务；有**可用的采集/生成流程**；记录数据量 + 质量检查 | ⚠️ **"downloading datasets alone does not satisfy this requirement"** —— 纯下载不算 |
| DP implementation & baseline training | 20 | 每个任务都训一个 DP 基线；记录架构、预处理、训练配置、算力 | 6 个任务的结果都要有 |
| Further investigation | 20 | 选 1 个研究问题，做受控实验 | 题目清楚 + 变量受控 |
| Evaluation & analysis | 25 | held-out seeds；每任务成功率 + 评测 episode 数 + 失败案例分析 | 权重最高，别只报一个数字 |
| Demo & reproducibility | 5 | 可复现的代码/数据/checkpoint | 给一键脚本 |
| Report / presentation / Q&A | 10 | 5–8 页报告 + 10 分钟讲 + 5 分钟问答 | 展示演示生成过程 + 所有任务的 rollout |

**交付物**：报告（5–8 页）、演示（10 min，含 demo，要有 6 个任务的视频覆盖）、代码+数据+checkpoint+复现说明、贡献说明、**LLM Usage Statement（必须！）**。

> 一句话策略：**"流程可复现" 比 "成功率高" 值钱**。成功率低但失败分析清楚，比堆一个漂亮数字得分高。

---

## 2. 总体路线图与里程碑

```
M1 任务选型         →  写死 6 个 env_id
M2 演示生成         →  6 个任务各 100~300 条成功轨迹（含质量检查报告）
M3 DP 基线          →  6 个任务各训一个 DP，出每任务成功率
M4 深入研究         →  1 个受控实验（数据效率 / DP 改进消融）
M5 交付             →  报告 + 视频 + 代码 + checkpoint
```

---

## 3. 第 1 步：六个任务选型

ManiSkill 3.0.1 中 **panda 机械臂 + 有官方演示** 的任务共 16 个。为了满足“不同操作技能”，推荐下面这套组合（**都能统一用 `pd_ee_delta_pos` 4 维动作空间**，便于公平对比和后续 co-training 实验）：

| # | 环境 ID | 操作技能 | 成功判据（要点） |
|---|---|---|---|
| 1 | `PickCube-v1` | 抓取 + 搬运 | 方块被举到 `goal_site` 附近 **且机械臂静止** |
| 2 | `StackCube-v1` | 精密堆叠 | `cubeA` 叠在 `cubeB` 上 |
| 3 | `PushCube-v1` | 非抓取推动 | 方块被推入 `goal_region` |
| 4 | `PullCube-v1` | 抓取 + 拖拽 | 方块被拖入 `goal_region` |
| 5 | `PegInsertionSide-v1` ⚠️ | 紧公差侧向插入 | peg 插入孔中（难度高，适合做失败分析） |
| 6 | `PlugCharger-v1` ⚠️ | 插接（紧配合） | charger 插入 receptacle |

> ⚠️ 带标记的两个任务**需要旋转末端**，`pd_ee_delta_pos`（4 维）做不到：请改用 **`pd_ee_delta_pose`（7 维：位置+旋转）**——官方 `baselines.sh` 里 `PegInsertionSide-v1` 用的就是它（数据与训练都按 7 维处理）。

**可替换/加练**：`PullCubeTool-v1`（工具使用）、`LiftPegUpright-v1`（姿态翻转）、`PlaceSphere-v1`（放入容器）、`StackPyramid-v1`（多层堆叠）。有官方运动规划演示的完整清单：

```
DrawTriangle-v1  PickCube-v1      StackCube-v1     PegInsertionSide-v1
PlugCharger-v1   PlaceSphere-v1  PushCube-v1      PullCubeTool-v1
LiftPegUpright-v1 PullCube-v1    DrawSVG-v1       StackPyramid-v1
（另有下载演示：PokeCube-v1 / RollBall-v1 / PushT-v1 / AnymalC-Reach-v1 /
  TwoRobotPickCube-v1 / TwoRobotStackCube-v1）
```

**统一设定（写进报告）**

```
观测 obs_mode     = "state"              → 扁平 42 维向量（见第 5 节）
动作 control_mode = "pd_ee_delta_pos"    → 4 维 [dx, dy, dz, gripper] ∈ [-1,1]
                    （仅 #5 #6 两个需要旋转的任务改用 "pd_ee_delta_pose" 7 维）
仿真后端           = "cpu"                （GPU 后端不支持控制模式转换，且本机无 Vulkan）
评测 episode 数    = 每个任务 50 次，seeds 2000~2049（与训练 seeds 完全不重叠）
max_episode_steps = 按官方 baselines.sh：100 / 100 / 200 / 300（见第 6.5 节）
```

> 💡 不要用 `-v1` 的随机种子差异冒充“不同任务”——评分表明确说 “Different random seeds of the same task do not count as separate tasks”。

---

## 4. 第 2 步：生成专家演示

课程允许 `teleoperation, motion planning, scripted controllers, or another justified method`。本仓库提供两条路线的现成脚本：

| 路线 | 脚本 | 说明 |
|---|---|---|
| A（覆盖 6 个任务） | `scripts/fetch_demos.py` → `scripts/convert_all.py` | 从 hf-mirror 下官方演示，再重放转换成 `state` + `pd_ee_delta_pos/pose` |
| B（"自己生成"证据） | `scripts/gen_demos_scripted.py` | 自写端点闭环控制器 + waypoint 程序，从零生成演示（不依赖 mplib） |

```bat
:: 路线 A（需要联网）
python scripts\fetch_demos.py PickCube-v1 StackCube-v1
python scripts\convert_all.py --tasks PickCube-v1 StackCube-v1

:: 路线 B（自己生成，本机就能跑）
python scripts\gen_demos_scripted.py --env-id PickCube-v1 -n 200
```

| 项 | 结论 |
|---|---|
| 数据位置 | `~/.maniskill/demos/<Task>-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` |
| 已有数据 | **PickCube-v1（1000 条成功轨迹）**，本项目的训练与评测都用它 |
| 其他任务 | 需要时再补；课程允许 `teleoperation / motion planning / scripted controllers / another justified method` 任一方式 |
| 交付口径 | 报告里写明演示来源与筛选标准（质检要点见 §5.4） |

## 5. 第 3 步：数据集格式与质量检查

### 5.1 文件结构（`obs_mode=state` + `pd_ee_delta_pos`）

```
trajectory.state.pd_ee_delta_pos.physx_cpu.h5
├── traj_0/
│   ├── obs                (T+1, 42) float32     ← 含最后一步观测
│   ├── actions            (T, 4)   float32
│   ├── terminated         (T,)     bool
│   ├── truncated          (T,)     bool
│   ├── success            (T,)     bool         ← 每步是否成功
│   └── env_states/                              ← 完整仿真状态（可精确重放）
│       ├── actors/  table-workspace, cube, goal_site  各 (T+1, 13)
│       └── articulations/  panda  (T+1, 31)
├── traj_1/ ...
└── traj_999/
trajectory.state.pd_ee_delta_pos.physx_cpu.json   ← 元数据
```

### 5.2 42 维 `state` 观测的精确布局（已实测，写报告直接引用）

| 切片 | 字段 | 维度 | 含义 |
|---|---|---|---|
| `[0:9]` | `agent.qpos` | 9 | 7 个关节角 + 2 个夹爪指关节 |
| `[9:18]` | `agent.qvel` | 9 | 对应关节速度 |
| `[18:19]` | `extra.is_grasped` | 1 | 是否夹住物体（0/1） |
| `[19:26]` | `extra.tcp_pose` | 7 | TCP 位姿（pos 3 + quat 4, wxyz） |
| `[26:29]` | `extra.goal_pos` | 3 | 目标点位置 |
| `[29:36]` | `extra.obj_pose` | 7 | 物体位姿 |
| `[36:39]` | `extra.tcp_to_obj_pos` | 3 | 物体 − TCP |
| `[39:42]` | `extra.obj_to_goal_pos` | 3 | 目标 − 物体 |

拼接顺序 = `dict(agent=..., extra=...)` 的插入顺序（`flatten_state_dict` 不排序）。

### 5.3 数据集质检（脚本已删除，留要点）

```bat
python scripts\check_datasets.py            :: 默认扫 4 维与 7 维数据集，报告写到 train_local\runs\_reports\
```

报告需要的质检结论与自查图见 §5.4。

### 5.4 质检要看什么（报告里必须有）

| 指标 | 期望 | 不达标怎么办 |
|---|---|---|
| `success_rate` | **100%**（脚本里已只保留成功轨迹） | 若不达标，检查 `flush_trajectory(save=success)` |
| `n_episodes` | 每任务 ≥100，推荐 200~300 | 加长生成时间 |
| `len_min/max` | 方差不要过大（如 40~200） | 过长的轨迹通常是控制器卡住，考虑丢弃 |
| `action_smoothness` | 越小越平滑；突然很大说明有跳变 | 检查是否出现动作饱和振荡 |
| `act_min/max` | 严格落在 `[-1, 1]` 内 | 超出说明动作空间用错 |
| 夹爪通道分布 | 应呈**双峰**（约 ±1） | 若为连续值，说明用了位置控制而非开合 |

**额外必查**：动作的夹爪维度（第 4 维）直方图。用 `matplotlib` 画一下，双峰图放进报告里很有说服力：

```python
import h5py, numpy as np, matplotlib
matplotlib.use("Agg")          # 本机无 Vulkan，必须用 Agg 后端
import matplotlib.pyplot as plt

with h5py.File(H5, "r") as f:
    g = f["traj_0"]
    a, o = g["actions"][:], g["obs"][:]
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(a[:, :3]); ax[0, 0].set_title("action dx,dy,dz")
ax[0, 1].hist(a[:, 3], bins=30); ax[0, 1].set_title("gripper channel (should be bimodal)")
ax[1, 0].plot(o[:, 29 + 2], label="obj z"); ax[1, 0].plot(o[:, 19 + 2], label="tcp z")
ax[1, 0].legend(); ax[1, 0].set_title("heights")
ax[1, 1].plot(o[:, 36], label="tcp->obj x"); ax[1, 1].legend(); ax[1, 1].set_title("relative pos")
plt.tight_layout(); plt.savefig("traj_overview.png", dpi=130)
```

---

## 6. 第 4 步：Diffusion Policy 实现

### 6.1 为什么用 state-based（MLP）版本

> 📌 **先看第 6.5 节**：ManiSkill 官方有一份可直接使用的 DP 基线（含 `ConditionalUnet1D`）；本项目当前**只维护自研实现**（更适合做「改进 DP」研究问题的可控实验平台），官方基线需要时再单独取。

- 官方 DP 仓库提供了 `ConditionalUnet1D`（图像）和低维 state 变体。我们的观测是 **42 维 state**（不是图像），用 **MLP + 条件注入** 就是 DP 论文里对应的 state 版本，**改动量小、训得快（4060 Ti 上单任务几分钟）**，而且更容易做消融。
- 图像版 DP 需要相机 + 渲染，而**本机没有 Vulkan**，走不通。

### 6.2 超参数（DP 论文默认值，直接用）

| 名称 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 观测视野 | `To` | 2 | 用最近 2 帧 state |
| 预测视野 | `Tp` | 16 | 一次预测 16 步动作 |
| 动作视野 | `Ta` | 8 | 只执行前 8 步，然后重规划 |
| 训练扩散步 | `T` | 100 | squaredcos_cap_v2 噪声调度 |
| 推理采样步 | — | 10 | DDIM，提速 10× |
| 优化器 | — | AdamW, lr 1e-4, wd 1e-6 | 带 cosine 退火 |
| EMA | — | decay 0.995 | 评测用 EMA 权重 |
| Batch | — | 256 | |
| Epoch | — | 300 | 早停看验证集损失 |

### 6.3 模型 + 数据集代码（真源 `dp/dp_lib.py`）

> 📌 **本节原来内嵌整套 MLP-only 代码（约 175 行），已删除** —— 仓库里有更新更准的真源：
> `dp/dp_lib.py`（模型 / 数据集 / 加噪 / 采样）+ `dp/backbones.py`（三种主干）+ `dp/utils.py`。
> 下面是接口速查，需要细节直接看真源：

| 接口 | 说明 |
|---|---|
| `cosine_beta_schedule(T)` | squaredcos_cap_v2 噪声调度（DP 默认） |
| `SinusoidalPosEmb(dim)` | 时间步 sin/cos 嵌入 |
| `DPDataset(trajs, stats, To, Tp)` | 按 episode 切窗、两端重复边界帧补齐、观测 z-score / 动作 min-max 归一化 |
| `DiffusionPolicy(obs_dim, act_dim, backbone=...)` | `compute_loss(obs_seq, act_seq)`（ε-prediction MSE）、`sample(obs_seq, K)`（DDIM） |
| `load_trajectories` / `compute_stats` / `split_episodes` / `find_dataset` | 读 h5、统计量、按 episode 划分、自动定位数据集 |

### 6.4 预处理要点

1. **观测 z-score 归一化**，`std < 1e-3` 的维度不做缩放（否则 `is_grasped`、静止时 `qvel` 这类维度会被放大成噪声）。
2. **动作归一化到 [-1,1]**（min/max）。对 `pd_ee_delta_pos` 这类本来就在 [-1,1] 的动作，这一步近似恒等映射，但保留它是为了将来换动作空间时不需要改代码。
3. **统计量只在训练集上算**，然后存进 checkpoint 给评测用 —— 这是防止信息泄漏的硬要求，报告里必须写。
4. **按 episode 划分 train/val**（比如 9:1），**绝不按单步随机划分**（同一 episode 的相邻帧高度相关，按步划分会严重高估验证效果）。

### 6.5 官方 DP 基线：与本项目实现的分工

> 📌 官方基线代码在 GitHub，`pip install mani_skill` **不含** `examples/baselines/`（见 §0 约束 7）。
> 需要时用 `python scripts\fetch_ms_baseline.py`（走 jsdelivr / gh-proxy 镜像）抓到 `third_party/ms_dp_baseline/`，
> 它自带 `train.py` 与 `baselines.sh`；本项目自己的实现则始终在 `dp/` + `train_local/`。

| 用途 | 用哪套 | 理由 |
|---|---|---|
| 6 个任务的 DP 基线（评分项 2，20 分） | 需要时另取官方基线 | 权威、超参已调好（`baselines.sh`），报告里写"复现官方 baseline"更稳 |
| **"Improving DP" 研究问题（评分项 3，20 分）** | **本项目实现 `dp/dp_lib.py` + `dp/backbones.py`** | 代码在自己手里才能改主干 / 视野 / 采样步数做消融 |

> ⚠️ 两者是不同实现，报告里别把自研版当官方基线来报。

**官方 `baselines.sh` 的超参**（本项目自研实现沿用同一套 `control_mode` / `max_episode_steps`，便于对比）：

| 任务 | `--control-mode` | `--max-episode-steps` | `--total-iters` |
|---|---|---|---|
| `PickCube-v1` | `pd_ee_delta_pos` | 100 | 30000 |
| `PushCube-v1` | `pd_ee_delta_pos` | 100 | 30000 |
| `StackCube-v1` | `pd_ee_delta_pos` | 200 | 30000 |
| `PegInsertionSide-v1` | **`pd_ee_delta_pose`**（7 维，插 peg 必须能转末端） | 300 | 100000 |
| `PushT-v1` | **`pd_ee_delta_pose`** | 150（`--num_eval_envs 100`，`--act_horizon 1`） | 50000 |

> `max_episode_steps` 必须按上表给（官方建议取平均演示长度的 2 倍），否则策略没时间在演示同长的时间里完成任务。

## 7. 第 5 步：逐任务训练

> 📦 **本节代码已写成现成可跑的一整套脚本**（2026-09-25 目录已重构为 库 / 工具 / 入口 三层）：
> ```
> pythonProject1/
> ├── dp/                      可复用库（两台机器共用同一份）
> │   ├── backbones.py         三种主干：MLP / 官方 UNet / 官方 Transformer(DP-T)
> │   ├── dp_lib.py            模型 + 数据集（真源，§6.3 只留接口速查）
> │   └── utils.py             EMA / loss 曲线
> ├── tools/
> │   ├── run_local.py         一键：数据检查 → 训练 → 评测 → 出 SUMMARY
> │   └── summarize_runs.py    汇总某个 runs/ → SUMMARY_all.md
> ├── train_local/             本机 Windows 入口（train.py / eval.py + README + runs/）
> └── train/                   学院 GPU 入口（train.py / eval.py + README + runs/）
> ```
> 在仓库根目录跑：`python tools/run_local.py --env-id PickCube-v1 --backbone unet`

### 7.1 训练脚本（真源 `train_local/train.py`）

> 📌 **本节原来内嵌整套训练脚本（约 120 行），已删除** —— 真源是 `train_local/train.py`（本机）
> 与 `train/train.py`（学院 GPU），两者训练循环与产物格式一致。参数表见 `train_local/README.md` 第 2 节，关键点：

| 要点 | 说明 |
|---|---|
| 数据划分 | **按 episode** 划分 train/val（绝不按单步随机切，否则验证集泄漏） |
| 迭代换算 | `--total-iters`（默认 30000，对齐官方 baseline）自动换成 epoch；也可 `--epochs` 直接指定 |
| 数据效率 | `--demo-frac` 从训练集里再裁一部分演示 |
| EMA | 固定 decay 0.995（`dp/utils.py`），评测默认用 EMA 权重 |
| 产物 | `best.pt` / `last.pt` / `log.csv` / `loss_curve.png` / `train_summary.json` |
| 冒烟 | `python train_local/train.py --env-id PickCube-v1 --epochs 2` |

### 7.2 一键训练 6 个任务

```bash
# PowerShell（每任务一轮；换主干时加 --backbone unet / transformer）
foreach ($T in "PickCube-v1","StackCube-v1","PushCube-v1","PullCube-v1","PegInsertionSide-v1","PlugCharger-v1") {
  python train_local/train.py --env-id $T --epochs 300
}
```

### 7.3 分两阶段跑：本地 4060 先通一个任务，再上集群

#### (1) 本地 RTX 4060 Ti（8 GB）到底能跑什么 —— 实测

| 项目 | 实测结果 |
|---|---|
| `torch.cuda.is_available()` | ✅ True（torch 2.14.0+cu126） |
| GPU | NVIDIA GeForce RTX 4060 Ti，**8.0 GB**，34 SM，compute capability 8.9 |
| **DP 模型训练（前向 / 反向）** | ✅ 纯 PyTorch，走 CUDA，**完全没问题** |
| `--sim-backend physx_cuda`（GPU 并行仿真） | ❌ `FileNotFoundError: Could not find module 'cuda.dll'` —— 本机**没装 CUDA Toolkit**（只有 `C:\Windows\System32\nvcuda.dll`），PhysX GPU 后端加载不了 |
| `--sim-backend physx_cpu` + `num_envs>1` | ❌ 不支持进程内向量化，只能 `num_envs=1`（或用多进程） |
| 渲染 / 录视频 | ❌ 无 Vulkan |

> ⚠️ 别把"训练"和"仿真"混为一谈。**DP 训练是纯张量运算 → 走 GPU；仿真评测是物理引擎 → 本机只能走 CPU。** 两者互不影响。

**=> 本地的正确姿势：训练放 GPU，评测放 CPU 单环境。** 评测开销完全可接受：CPU 仿真约 **60~140 步/秒**，50 个 episode × 100 步 ≈ 5000 步 ≈ **40~80 秒**。所以"在本地跑通第一个任务"是**完全可行的**，不用等集群。

#### (2) 阶段 1：本地跑通 PickCube-v1

```bash
python train_local/train.py --env-id PickCube-v1 --epochs 300 --device cuda
python train_local/eval.py  --ckpt train_local/runs/<刚训好的实验目录>/best.pt -n 50 --seed0 2000
```

**显存**：三个主干在 batch 256 下的峰值显存实测 ≤ 2.2 GB（最大是 66.4 M 的 UNet），8 GB 绰绰有余。

**怎么判断跑通了**：

| 观察项 | 期望 |
|---|---|
| 训练 loss | 从 ~1 降到 < 0.01（30000 iter 内） |
| 评测成功率 | PickCube 本机 MLP 基线实测 **96%（48/50）**；低于 50% 先查 `max_episode_steps` 与演示长度 |
| 单 epoch 时间 | 4060 Ti：MLP 几秒；UNet/Transformer 约 5~8× 慢（见 A.3.2） |

**第一个任务建议就用 `PickCube-v1`**：动作空间最简单（4 维）、演示已经现成（1000 条）、官方有调好的超参，跑通它只为了**打通整条链路**（数据 → 训练 → 评测 → 出数字）。

#### (3) 阶段 2：迁到学院集群

集群上会**变好**的地方：

| 项 | 本地 | 集群（Linux） |
|---|---|---|
| `physx_cuda` GPU 并行仿真 | ❌ 缺 cuda.dll | ✅ 有 CUDA Toolkit，可开 100 个并行环境，评测几十秒一轮 |
| `mplib`（官方运动规划） | ❌ Windows 装不上 | ✅ 有现成 wheel → **可以真正自己跑运动规划生成演示** |
| `pinocchio` | 手工移植 36 个 DLL | ✅ SAPIEN Linux 内置 |
| Vulkan 渲染 | ❌ 无驱动 | ✅ 通常可用 → **能出报告要求的 6 个任务 rollout 视频** |

**要带走的东西（清单）**：

1. 数据集：`~/.maniskill/demos/<Task>-v1/motionplanning/*.state.pd_ee_delta_pos.physx_cpu.h5`（**h5 跨平台，直接拷贝即可，不用重新生成**）
2. 代码：仓库整体（`dp/` + `tools/` + `train_local/` + `train/` + `pyproject.toml`），见根 `README.md`
3. 版本锁（本地实测组合）：

```text
python        3.13.12（集群若是 3.9~3.12 也可以，mani_skill 3.0.1 都支持）
torch         2.14.0+cu126
torchvision   0.29.0+cu126
torchaudio    2.11.0+cu126
mani_skill    3.0.1
sapien        3.0.3
gymnasium / h5py / numpy / matplotlib
# 可选：diffusers / tensorboard（用官方 DP 基线时才需要）；mplib（仅 Linux，用运动规划生成演示时才需要）
```

4. **种子清单**：训练 seeds、评测 seeds（2000~2049）、数据划分 seed 全部写进一个 `SEEDS.md`，本地与集群保持一致。

> ⚠️ **一致性红线**：CPU 仿真和 GPU 仿真在**相同 seed 下的初始状态随机化并不完全一致**。如果你本地用 `physx_cpu` 出的数字，不要和集群 `physx_cuda` 出的数字混在同一张对比表里。要么**所有任务统一在一个后端上评测**，要么在报告里明确分栏标注后端。评分表里"fair comparisons"是硬要求。

### 7.4 算力记录（报告里必须写）

| 项 | 值 |
|---|---|
| GPU（本地） | NVIDIA GeForce RTX 4060 Ti，**8.0 GB**，34 SM，cc 8.9，驱动 561.09 |
| PyTorch | 2.14.0+cu126（CUDA 12.6），`torch.cuda.is_available() = True` |
| 单任务训练时间 | 30k iters 实测：MLP ~5 min、Transformer ~27 min、UNet ~40 min（见 A.3.2） |
| 训练显存占用 | batch 256 峰值 ≤ 2.2 GB（UNet 66.4 M 最大；MLP/Transformer 更小） |
| 评测（本地 CPU 单环境） | CPU 仿真约 60~140 步/秒；50 episode × 100 步 ≈ 40~80 秒 |
| 评测（集群 GPU 并行） | `physx_cuda` 可开 ~100 并行环境，一轮评测几十秒 |
| 演示生成 | 脚本控制器约 15~40 条/分钟（受 IK/PD 收敛速度限制；本项目的 PickCube 数据已生成好） |

---

## 8. 第 6 步：评测（held-out seeds）

### 8.1 评测脚本（真源 `train_local/eval.py`）

> 📌 **本节原来内嵌整套评测脚本（约 80 行），已删除** —— 真源是 `train_local/eval.py`（本机，`physx_cpu` 单环境）
> 与 `train/eval.py`（学院 GPU，支持 `--num-envs` 并行仿真）。关键点：

| 要点 | 说明 |
|---|---|
| 观测历史 | 维护最近 `To` 帧（首帧重复补齐），与训练切窗一致 |
| 动作执行 | receding horizon：每 `Ta` 步重规划一次，块内其余步直接复用 |
| 反归一化 | 观测用训练集统计量 z-score，动作反归一化回真实尺度 |
| 失败分类 | 自动区分「超时」（跑满 `max_episode_steps`）与「提前终止」（`terminated`） |
| 输出 | `eval_seed<seed0>_n<N>.json`：逐 episode 的 seed / success / steps |

### 8.2 批量评测并汇总成表

```bat
python scripts\eval_all.py                          :: 扫 train_local\runs\*\best.pt，逐个调 eval.py
python tools\summarize_runs.py --runs-dir train_local\runs --csv   :: 汇总成 5 节报告
```

`summarize_runs.py` 会自动出 5 节（总览 / 按任务对比 / 评测明细 / 训练配置 / 完整性提示），可直接贴进报告。

### 8.3 评测的铁律（评分表里 25 分就在这）

1. **seeds 必须 held-out**：训练用 `0 ~ n-1`，评测用 `2000 ~ 2049`，报告里明确写出来。
2. **每个任务至少 50 个评测 episode**，并报告 `n_ok/n_total`，不要只写百分比。
3. **失败案例要分类**：抓不到 / 抓了掉了 / 送不到位 / 超时。给出“动作块执行到第几步开始跑飞”这类定量观察。
4. **对比要公平**：数据量、算力预算、`Tp/Ta`、评测 seeds 必须完全一致。论文里写清楚“差异只在 X 上”。
5. **多次随机种子**：关键结论至少跑 3 个训练 seed，报告 `mean ± std`。

---

## 9. 第 7 步：深入研究实验（研究问题）

从建议清单挑一个，**推荐下面两个之一（都很好做、容易出结论）**。

### 选项 A：数据效率（Data efficiency）—— 最推荐

**问题**：需要多少条演示？多样性/质量比数量更重要吗？

**设计**（用已有的 `--demo-frac` 就能做）：

| 数据量 | 100 条 | 50 条 | 25 条 | 10 条 |
|---|---|---|---|---|
| 训练 seed | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 | 0, 1, 2 |
| 评测 | 50 ep，seeds 2000–2049（固定） | 同 | 同 | 同 |

```bash
# 四档数据量（分档跑），或直接：python tools/run_local.py --env-id PickCube-v1 --backbone unet --data-efficiency
foreach ($F in 1.0, 0.5, 0.25, 0.1) {
  python train_local/train.py --env-id PickCube-v1 --backbone unet --demo-frac $F --seed 0 --epochs 300
}
```

在 **2~3 个任务**上做（如 PickCube + StackCube + PushCube），画 `成功率 vs 演示条数` 曲线并标出标准差带。

**附加的“质量 vs 数量”对照**（很出彩）：故意用**未筛选**的演示（把脚本里 `save=success` 改成 `save=True`，此时成功约 60~80%），同样 50 条演示再训一次，和 50 条全成功演示对比 → 直接回答“质量是否比数量更重要”。

### 选项 B：改进 DP（预测视野 / 动作视野消融）

**问题**：改预测视野 `Tp` 或动作视野 `Ta` 能否提升成功率或推理速度？

**设计**（4 组受控实验，其余超参完全固定）：

| 组 | Tp | Ta | 假设 |
|---|---|---|---|
| baseline | 16 | 8 | — |
| 短视野 | 8 | 4 | 反应更快，但可能不够“前瞻” |
| 长视野 | 32 | 16 | 更前瞻，但误差累积 |
| 慢重规划 | 16 | 16 | 推理频率减半 → 速度更快，成功率？ |

测 **成功率 + 单步推理延迟（ms）**，做成功率-延迟的帕累托图。

### 报告呈现（必备三张图）

1. 成功率 vs 数据量（含 std 带）
2. 每任务成功率柱状图（6 个任务）
3. 失败模式饼图 / 堆叠柱状图

---

## 10. 第 8 步：交付物与报告结构

### 10.1 建议的代码/数据目录

```
project/
├── dp/                       可复用库（backbones.py / dp_lib.py / utils.py）
├── tools/                    run_local.py（一键） / summarize_runs.py（汇总）
├── train_local/  train/      两套入口脚本（本机 Windows / 学院 GPU），各自带 runs/
├── runs/<task>_frac<f>_<backbone>_seed<s>/best.pt   checkpoints（含主干名）
├── data/                     生成的数据集（或下载链接）
├── videos/                   每任务成功/失败 rollout（若无 Vulkan，用帧序列拼 GIF）
└── REPORT.md
```

### 10.2 报告结构（5–8 页，对齐评分表）

| 章节 | 页数 | 对应评分项 |
|---|---|---|
| 1. 任务选择与观测/动作空间 | 0.5 | Task coverage (20) |
| 2. 演示生成流程（两条路线 + 数据量 + 质量检查表 + 光滑性/夹爪分布图） | 1.5 | Task coverage (20) |
| 3. DP 架构与训练配置（含 42 维观测布局表、归一化策略、算力） | 1 | DP baseline (20) |
| 4. 每任务基线结果（表 + 柱状图 + 失败案例分析） | 1.5 | Evaluation (25) |
| 5. 深入研究：问题 / 假设 / 受控变量 / 结果 / 结论 | 1.5 | Further investigation (20) |
| 6. 局限性与风险（无 Vulkan 无法出视频、mplib 不可用、成功率的方差） | 0.5 | Evaluation (25) |
| 7. LLM Usage Statement | 0.25 | 合规必写 |
| 8. 贡献声明 + 复现说明 | 0.25 | Demo & repro (5) |

### 10.3 LLM Usage Statement 示例（按课程给的模板改）

```
LLM Usage Statement
- 演示生成：使用 Claude 起草 waypoint 控制器骨架与 world→base 坐标变换；
  小组自行调试抓取高度、夹爪闭合步数，并逐条验证轨迹成功率。
- DP 实现（`dp/dp_lib.py`、`dp/backbones.py`）：使用 Claude 解释 Diffusion Policy 论文的噪声调度与
  DDIM 采样公式，代码由小组自行编写并逐项对照论文核对。
- 报告第 2、5 节：使用 Claude 改善语法与表达；技术结论均由小组验证。
- 所有引用的外部代码/数据（ManiSkill、Diffusion Policy 官方仓库、
  ManiSkill_Demonstrations 数据集）已在参考文献中注明。
```

---

## 11. 踩坑速查表

| 报错/现象 | 根因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'mplib'` | Windows 无 mplib 轮子 | 别用官方 mp 脚本；走下载转换 或 自写控制器 |
| 找不到官方 DP 代码 / `baselines/` 目录不存在 | pip 包里不含 `examples/baselines/` | 需要时从 GitHub 取；本机 github.com 不通，用 `cdn.jsdelivr.net/gh/...@main/<path>` 取单文件 |
| `github.com` 连不上、clone 半途死掉 | 本机 GitHub 被墙 | 用 `cdn.jsdelivr.net/gh/...@main/<path>` 取单文件 |
| 官方 `train.py` 报 `No module named 'tensorboard' / 'wandb'` | 依赖缺失 | `pip install tensorboard`；wandb 加 `--no-track` |
| `TypeError: 'NoneType' object is not callable`（`PinocchioModel`） | SAPIEN 的 pinocchio 未装 | 已移植到 venv（`_pinocchio_dlls` + `.pth`）；换机器需重做 |
| `--num-procs` 参数不识别 | 3.0.1 改名为 `-n/--num-envs` | 用 `-n 4` |
| `urllib.error.URLError: Tunnel connection failed: 502` | huggingface.co 被代理拦 | 用 `hf-mirror.com` + User-Agent |
| `HTTP Error 403: Forbidden`（hf-mirror） | 缺 User-Agent | 加 `{"User-Agent": "Mozilla/5.0"}` |
| `--save-video` 直接段错误 | 无 Vulkan ICD | 不要开视频；matplotlib 用 `Agg` 后端出图 |
| episode 在 50 步被截断 | 默认 `max_episode_steps=50` | `gym.make(..., max_episode_steps=300)` |
| 自己写的控制器完全不动 | 机器人基座在 `x=-0.615`，不是原点 | 必须做 world→base 变换 |
| PickCube 举到目标却不判成功 | 判据含 `is_robot_static`，且目标在空中 | **不要松手**，握着停住 ~60 步 |
| 训练爆炸 / loss = NaN | 低方差维度被 `std≈0` 放大 | `std < 1e-3` 时不做缩放；梯度裁剪 1.0 |
| 验证损失很低但成功率是 0 | 按单步随机划分导致泄漏 | 按 episode 划分 |
| 转换进程退出码非 0 | 主进程删分片被安全删除拦截 | 数据已合并；手动删 `*.physx_cpu.{0,1,2,3}.h5/.json` |
| 换了 `unet` / `transformer` 主干后旧 ckpt 加载报 `Missing key(s) in state_dict: "noise_..."` | 2026-09-25 起主干换成官方实现，结构变了（参数量 2.806 / 3.348 M → 66.418 / 8.972 M） | 旧 unet/transformer 结果作废，**必须重训**；只有 `mlp` 的旧 ckpt 仍可直接评测 |
| 想知道"DP 的 Transformer 到底多少参数" | 论文 Table 8 里 DP-T 的 `#D-params`（扩散网络）= **9 M**，另有 `#V-params`（视觉编码器，image 版才有）= 22 M，两者相加 31 M 常被误当成 state 版参数 | state 版（我们的场景）就是 **9 M 量级**，实测 8.972 M；22 M 那部分只属于图像版的两路 ResNet18 |

---

## 12. 建议排期

| 阶段 | 内容 | 产出 |
|---|---|---|
| D1–D2 | 6 个任务下载 + 转换（路线 A）；跑质检出 report.md | 6 个数据集 |
| D3–D4 | 自写控制器覆盖 2~3 个任务（路线 B）；处理失败任务 | 自建生成流程 |
| D5–D6 | 跑通 DP 训练 + 评测（先只做 PickCube 打通链路） | 端到端可复现 |
| D7–D9 | 6 个任务全量训练 + 评测（每任务 50 ep） | 基线结果表 |
| D10–D12 | 数据效率（或视野消融）实验，3 个 seed | 深入研究结论 |
| D13–D14 | 出图、写报告、录演示 | 交付 |

---

### 参考链接

- ManiSkill 官方仓库 / 文档：https://github.com/haosulab/ManiSkill
- Diffusion Policy 论文与官方仓库：https://diffusion-policy.cs.columbia.edu/
- 演示数据集（需镜像访问）：`haosulab/ManiSkill_Demonstrations`
- 本机镜像下载地址：`https://hf-mirror.com/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/<Task>-v1.zip`

---

## 附录 A：DP 网络结构对比实验（三种主干）

> 对应评分表 **Further investigation（20 分）** 的主实验。
> 目标：在同一套数据 / 评测协议下，对比 **MLP / 1D-UNet / Transformer** 三种 DP 主干的训练效果。
> **代码已落地**（`dp/backbones.py` 提供三种主干，`dp/dp_lib.py` / `train_local/train.py` 已接入 `--backbone`），
> 本附录给**方案、命令与报告写法**，训练由你自己执行。

### A.0 结论先行（推荐方案）

| 项 | 建议 |
|---|---|
| 对比对象 | MLP（已有，baseline） vs 1D-UNet vs Transformer，共 3 个主干 |
| 任务范围 | 先 **3 个任务**（难度梯度）：PickCube-v1 → StackCube-v1 → PegInsertionSide-v1 |
| 总工作量 | 代码已写好；本地 4060 训练约 **一晚（6~10 h GPU）** |
| 扩展 | 6 任务 × 3 主干 = 18 组 → 建议上集群 |
| 为什么值得做 | 文献结论是「简单任务三种主干打平、难任务 UNet 明显更强」——你的实验大概率能复现这个趋势，本身就是一条干净的叙事线 |

### A.1 主流 DP 网络结构盘点（2026 年视角）

条件去噪网络的统一契约：`ε_θ(x_t, t, obs) → ε̂`，x_t 是 `(Tp, A)` 的动作序列。差别全在**主干怎么处理这段序列**、以及**观测条件怎么注进去**。

**A.1.1 三大主干（Chi et al. 2023 起的共识分类）**

| 主干 | 代表 | 处理方式 | 条件注入 | 特点 |
|---|---|---|---|---|
| **1D 时序 UNet**（DP-CNN） | DP 论文默认；ManiSkill 官方 baseline 用 `conditional_unet1d.py` | 动作序列当一维信号，Conv1d 下采样-上采样 + skip connection | **FiLM**（每个残差块用条件调制 scale/shift） | 局部+全局兼顾，难任务上最强；官方 baseline 的默认选择 |
| **Transformer**（DP-T） | DP 论文变体（minGPT 式解码器）；后续 DiT 类 | 动作 token + 观测 token 做注意力 | cross-attention / AdaLN / FiLM | 全局依赖强、适合长序列；实测训练更挑超参、更不稳 |
| **MLP** | 低维 state 任务的朴素选择 | 展平拼接后全连接 | 直接拼接进输入向量 | 最快最稳；**简单任务够用，难任务天花板低** |

**A.1.2 文献里的实测锚点（引用时注明出处）**

emergentmind 汇总的 UNet vs MLP 对比（图像版 DP、官方设置）：

| 任务 | 难度 | UNet | MLP |
|---|---|---|---|
| StackCube (ManiSkill) | 易 | 99% | 99% |
| PegInsertionSide (ManiSkill) | 难 | 80% | **21%** |
| TurnFaucet | 难 | 59% | 22% |
| Door (Adroit) | 难 | 95% | 35% |

> 注意：这是**图像观测 + 官方完整训练**的数字，你的 state 版数字会不同，别直接对表 —— 要复现的是**趋势**（难任务上 UNet ≫ MLP，易任务打平），不是数值。

**A.1.3 2024–2026 的新变体（了解即可，不建议做）**

- **U-DiT Policy**（arXiv 2509.24579, 2025-09）：U 形 Diffusion Transformer，融合 UNet 多尺度 + Transformer 全局建模，自称比 DP-U 高 10%。
- **Modulated Attention**（MTDP, 2025）：把条件调制进 Q/K/V，不只靠 cross-attention。
- **3D Diffuser Actor / DP3**：3D 点云 / 物体表征版。
- **Consistency Policy**：改采样（少步数），不是改主干。

**为什么不建议**：这些都需要额外依赖、更大算力或非 state 观测，对「6 任务 + 公平对比」的课程框架性价比低。报告里放一段 related work 即可。

### A.2 实验设计（公平性是 25 分那部分的命根子）

**A.2.1 固定变量清单（除主干外全部锁死）**

| 变量 | 取值 | 说明 |
|---|---|---|
| 数据 | 同一个 h5，`success_only=True` | 三种主干吃完全相同的数据 |
| train/val 划分 | `--seed 0` | 同一个划分 → val loss 可横向比 |
| 观测/动作 | `state`(42) / `pd_ee_delta_pos`(4)，`To/Tp/Ta = 2/16/8` | 不许动 |
| 总迭代数 | 82,200（即 `--epochs 300`） | 对齐 MLP baseline 那轮 |
| 优化器 | AdamW, lr 1e-4, wd 1e-6, batch 256, EMA 0.995 | 三者相同；若 UNet/Transformer 发散再单独调 lr 并**在报告里声明** |
| 评测 | `eval.py -n 50 --seed0 2000`，`physx_cpu`，same `max_episode_steps` | 同一批 held-out seeds，绝不重叠 |
| 改动的唯一变量 | `--backbone` | 其余一切相同 |

**A.2.2 每个主干记什么**

- 成功率（`n_ok/50`）+ 失败分类（超时/早停，`eval.py` 自带）
- 参数量、训练时长、**iters/s**（三种主干差好几倍，报告里要有）
- `log.csv` 的 val loss 曲线（同一张图，x 轴用 iteration）
- （可选）推理耗时：`eval.py` 的总用时就是代理指标

### A.3 工作量评估

**A.3.1 实现成本 —— ✅ 已全部落地**

| 项 | 状态 | 说明 |
|---|---|---|
| `dp/backbones.py`（MLP / UNet / Transformer + 工厂） | ✅ 已写好（2026-09-25 升级） | UNet = 官方 `ConditionalUnet1D` 完整移植（每级 2 残差块 + 2 mid 块 + ConvTranspose1d 上采样 + 时间嵌入与条件拼接后 FiLM）；Transformer = 官方 `TransformerForDiffusion`（DP-T，cond memory + TransformerDecoder cross-attn + 因果 mask）。接口按主干分发：`forward(x, timestep, global_cond=...)` / `forward(x, timestep, cond=...)` / MLP 仍是 `forward(x, t_emb, c)` |
| `dp/dp_lib.py` 接入 | ✅ 已改 | `DiffusionPolicy(backbone=...)`，`compute_loss`/`sample` 全部复用；unet/transformer 走 A1 条件路径（无 obs encoder，条件=归一化 obs） |
| `train.py` 接入 | ✅ 已改 | `--backbone` 与主干超参（`--unet-*` / `--tf-*`）写进 ckpt，实验名带主干 |
| `eval.py` | ✅ 零改动 | `backbone` 随 `model_kwargs` 存进 ckpt，自动识别；**旧 unet / transformer ckpt 已不兼容**（结构变了），旧 MLP ckpt 仍兼容 |
| 冒烟验证 | ✅ 已过 | 三种主干都能训练；参数量 0.353 / **66.418** / **8.972** M（后两者与官方实现权重互灌、前向输出误差 0） |

> 结论：**直接用 A.6 的命令即可，不需要再改代码。** 代码细节见 `dp/backbones.py`（头注释含与官方源码的逐行对应关系）与 `train_local/README.md` §6。

**A.3.2 训练成本（本地 4060 Ti 8GB，batch 256 实测）**

| 主干 | 实测速度 | 相对 MLP | 单任务 82k iters | 3 任务 | 6 任务 |
|---|---|---|---|---|---|
| MLP（0.353 M） | 97 iters/s（82k iters ≈ 849 s） | 1× | ~15 min | ~45 min | ~1.5 h |
| 1D-UNet（66.418 M） | ~80 ms/iter ≈ 12 iters/s | ~8× 慢 | ~1.9 h | ~5.7 h | ~11 h |
| Transformer（8.972 M） | ~55 ms/iter ≈ 18 iters/s | ~5× 慢 | ~1.3 h | ~3.8 h | ~7.6 h |
| **合计（3 主干）** | | | **~3.4 h** | **~10 h（过夜）** | ~20 h（尽量上集群） |

> 口径：batch 256、`Tp=16`、obs 42 维；UNet/Transformer 的数字来自真机 `compute_loss` 反传的平均耗时（含调度开销，未含数据加载）。
> 按 30k iters（官方 baseline 常用的 `--total-iters 30000`）算则是 **UNet ≈ 40 min、Transformer ≈ 27 min、MLP ≈ 5 min**。
> ⚠️ 仍然是量级参考：**动手第一件事**还是 `--epochs 2` 冒烟时看打印的 iters/s，用实测值重算再决定做几个任务。

**A.3.3 推荐的裁剪**

| 档位 | 规模 | 何时选 |
|---|---|---|
| 最小可行 | 2 主干（MLP vs UNet）× 3 任务 = 4 组新训练 | 时间紧；文献里 Transformer 本来就不占优 |
| **推荐** | 3 主干 × 3 任务 = 9 组 | 本地一晚跑完，报告三线对比最好看 |
| 完整 | 3 主干 × 6 任务 = 18 组 | 上集群；PegInsertionSide 这类难任务才是分化点 |

### A.4 任务选择的关键：必须有一个「难任务」

文献趋势是「**易任务打平、难任务分化**」。PickCube 你已经 96%，三种主干大概率都 90+，拉不开差距。所以：

| 任务 | 难度 | 动作空间 | 预期作用 |
|---|---|---|---|
| PickCube-v1 | 易（已有 96% MLP 基线） | 4 维 `pd_ee_delta_pos` | 锚点：验证「易任务打平」 |
| StackCube-v1 | 中 | 4 维 | 文献锚点（99/99），数据需自行生成（§4） |
| PegInsertionSide-v1 | **难** | **7 维 `pd_ee_delta_pose`** | 分化点：预期 UNet ≫ MLP |

两个前置注意：

1. **StackCube / PegInsertionSide 的数据集还没生成**（§4 的生成脚本已删除，需要时从 `.ref/教程_v0_full.md` 恢复）。注意：
   - StackCube 用 `pd_ee_delta_pos`（4 维）+ `--max-episode-steps 200`
   - PegInsertionSide 要 `pd_ee_delta_pose`（7 维，有旋转）+ `--max-episode-steps 300`；数据集后缀是 `.state.pd_ee_delta_pose.physx_cpu.h5`，而 `dp.dp_lib.find_dataset` 只认 `pd_ee_delta_pos`，所以要**手动用 `--h5` 传路径**，或改 `H5_SUFFIX`。
2. 新生成的数据集先质检再训（要点见 §5.4）：条数、成功率、动作落在 `[-1,1]`、夹爪通道是否双峰。

### A.5 代码接入要点（已落地，供理解改动）

三种主干**不再共用同一个 forward 签名**——因为官方 UNet / Transformer 都要求"传入原始 timestep、主干内部自己做时间嵌入"：

- MLP：`forward(x, t_emb, c)`，`t_emb:(B,128)` 由外部 `SinusoidalPosEmb` 算好
- UNet（官方）：`forward(sample, timestep, global_cond=...)`，`global_cond:(B, obs_dim*To)`（展平）
- Transformer（官方 DP-T）：`forward(sample, timestep, cond=...)`，`cond:(B, To, obs_dim)`（逐步）

`DiffusionPolicy.cond()` 负责按主干准备条件，`eps()` 负责分发签名（`dp/dp_lib.py`）：

```python
def eps(self, x, t, c):
    if self.backbone == "unet":
        return self.noise_pred(x, t, global_cond=c)   # 官方签名
    if self.backbone == "transformer":
        return self.noise_pred(x, t, cond=c)          # 官方签名
    return self.noise_pred(x, self.t_emb(t), c)       # MLP
```

`compute_loss` / `sample` **一行都不用动**。完整的三种主干实现请看源码，不再在本教程内重复贴出：

- `dp/backbones.py` — `MLPNoisePred` / `ConditionalUnet1D`（官方移植）/ `TransformerForDiffusion`（官方移植）+ `build_noise_pred`
- `dp/dp_lib.py` — `DiffusionPolicy(backbone=...)`、`cond()` / `eps()` 分发、A1 条件路径
- `dp/utils.py` — `EMA` / `plot_curve`（两套 train.py 共用）
- `train_local/train.py`、`train/train.py` — `--backbone` 与主干超参、写入 ckpt、实验名含主干

### A.6 操作流程（按顺序执行）

```bash
# 在仓库根目录（pythonProject1/）执行

# ---- Step 0 冒烟：确认三种主干都能训、loss 都在降（每个几十秒）----
python train_local/train.py --env-id PickCube-v1 --backbone unet        --epochs 2
python train_local/train.py --env-id PickCube-v1 --backbone transformer --epochs 2
#   顺带记下打印里的 iters/s，用它修正 A.3.2 的时间估算

# ---- Step 1 主实验：PickCube（易任务锚点；MLP 那组你已经有了）----
python train_local/train.py --env-id PickCube-v1 --backbone unet
python train_local/train.py --env-id PickCube-v1 --backbone transformer
python train_local/eval.py --ckpt train_local/runs/PickCube-v1_frac1.0_unet_seed0/best.pt         -n 50 --seed0 2000
python train_local/eval.py --ckpt train_local/runs/PickCube-v1_frac1.0_transformer_seed0/best.pt -n 50 --seed0 2000

# ---- Step 2 中间难度：StackCube（先转数据集，见 A.4）----
python train_local/train.py --env-id StackCube-v1 --max-episode-steps 200                 # mlp
python train_local/train.py --env-id StackCube-v1 --max-episode-steps 200 --backbone unet
python train_local/train.py --env-id StackCube-v1 --max-episode-steps 200 --backbone transformer
#   三个都各自 eval -n 50 --seed0 2000

# ---- Step 3 难任务：PegInsertionSide（7 维动作，分化点）----
#   转数据集（pd_ee_delta_pose，max_episode_steps 300），训练时 --h5 手动传路径

# ---- Step 4 汇总出表出图 ----
python tools/summarize_runs.py --runs-dir train_local/runs --csv
```

> 也可以直接写个 bat/sh 把 9 组「train + eval」串起来过夜跑；
> `tools/run_local.py` 现已支持 `--backbone`（会写成 `<env>_frac<f>_<backbone>_seed<s>/`，与 `train.py` 命名一致），
> 想跑农场那套入口脚本时再加 `--train-dir train --device cuda`。

### A.7 报告呈现

**A.7.1 主表（每个格子写 n_ok/50）**

| 任务 | MLP | 1D-UNet | Transformer |
|---|---|---|---|
| PickCube-v1 | 96.0 (48/50) | ? | ? |
| StackCube-v1 | ? | ? | ? |
| PegInsertionSide-v1 | ? | ? | ? |

**A.7.2 图清单**

1. **分组柱状图**：任务 × 主干 → 成功率（主图）
2. **val loss 曲线**：三主干同图，x 轴 iteration（`log.csv` 直接画）—— 说明「谁收敛快/谁更稳」
3. **成本表**：参数量 / iters/s / 训练时长 / 评测耗时（回答 "inference speed"）

**A.7.3 叙事线（照着写就不会偏）**

1. 三种主干在同一协议下对比 → 易任务（PickCube）预计打平，难任务（PegInsertionSide）预计 UNet 领先 → 与文献趋势一致/不一致及原因分析
2. MLP 输在哪儿：观测条件只做一次拼接、没有时序归纳偏置；UNet 的 FiLM + skip 保留了动作序列的局部结构
3. Transformer 若不稳：讨论因果 mask 下的位置信息缺失、小数据上注意力难训（引用 DP 论文原话 "less stable"）
4. 结论落回「按任务难度选主干」的工程建议

### A.8 风险与坑

| 风险 | 预案 |
|---|---|
| UNet / Transformer 在 82k iters 发散或 loss 不降 | 先降 lr 到 5e-5 或加 warmup；**报告里声明每个主干用的 lr**（公平性允许调各自主干的 lr，只要数据/评测一致） |
| GroupNorm 报错通道不整除 | `down_dims` 全是 8 的倍数（64/128/256），别乱改；`act_dim=7`（PegInsertionSide）只进 conv_in，无归一化，安全 |
| Transformer 显存/速度超预期 | `d_model=192, n_layers=3` 砍一档 |
| PegInsertionSide 演示成功率低 / 条数少 | 先按 §5.4 做质检；必要时降低 demo_frac 或换 LiftPegUpright-v1 当难任务 |
| 三种主干 Eval 结果混入不同后端 | 全程 `physx_cpu` + seeds 2000..2049，跑之前 grep 确认 |
| `Tp` 改了导致 UNet 形状错 | UNet 有 `assert Tp % 4 == 0`；做 Tp 消融时选 8/16/32 |

### A.9 参考

- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023（三大主干 + FiLM 的出处）
- *Diffusion Models for Robotic Manipulation: A Survey*（arXiv 2504.8438）：三大架构的系统对比
- U-DiT Policy（arXiv 2509.24579, 2025-09）：UNet vs Transformer 的最新组合方案
- ManiSkill 官方 DP baseline（`examples/baselines/diffusion_policy/conditional_unet1d.py`）—— 本方案 UNet 的结构参照；想用官方原版按 §6.5 的说明从 GitHub 取（本机 github.com 不通时的取文件方法见 §11）
