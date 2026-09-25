# DASC7606C 小组项目 · Track 3：Simulation Experiments

在 **≥6 个 ManiSkill 任务**上自采/生成专家演示，训练并评测 **Diffusion Policy（DP）** 基线，
并围绕一个研究问题做受控实验（**数据效率** / **网络结构对比**）。

> **本机环境**：Windows + RTX 4060 Ti 8 GB；PyTorch 2.14.0+cu126；ManiSkill 3.0.1 / SAPIEN 3.0.3（已移植 pinocchio 4.1.0）。
> 训练走 **GPU**，评测只能走 **CPU 单环境仿真** —— 无 CUDA Toolkit → `physx_cuda` 不可用；无 Vulkan → 不出视频。

---

## 目录结构

```
pythonProject1/
├── README.md                          ← 本文件：项目总览 + 导航 + 快速上手
├── dp/                                ← ★ 可复用库（两台机器、所有脚本共用同一份）
│   ├── backbones.py                   ← 三种噪声预测主干：MLP / 官方 UNet / 官方 Transformer(DP-T)
│   ├── dp_lib.py                      ← DP 模型（DDPM 训练 / DDIM 采样）+ 数据集 + 归一化统计
│   └── utils.py                       ← EMA / loss 曲线（两套 train.py 共用，不再是复制粘贴）
├── tools/                             ← ★ 机器无关的工具脚本
│   ├── run_local.py                   ← 一键：训练 → 评测 → 出 SUMMARY（--train-dir 可指向任一套）
│   ├── summarize_runs.py              ← 汇总某个 runs/ → SUMMARY_all.md（--runs-dir 指哪扫哪）
│   └── fix_venv_paths.py              ← 改名 / 搬家后修 venv 里写死的绝对路径
├── scripts/                           ← ★ 数据侧一次性脚本（演示下载/转换/生成、质检、批量评测）
│   ├── fetch_demos.py                 ← 从 hf-mirror 下载官方演示
│   ├── convert_all.py                 ← 重放转换成 state + pd_ee_delta_pos / pose
│   ├── gen_demos_scripted.py          ← 自写脚本控制器生成演示（不依赖 mplib）
│   ├── check_datasets.py              ← 数据集质检 → report.md / report.csv
│   ├── fetch_ms_baseline.py           ← 需要时抓 ManiSkill 官方 DP 基线做对照
│   └── eval_all.py                    ← 批量评测多个 checkpoint
├── train_local/                       ← ★ 本机 Windows（RTX 4060 Ti）入口脚本
│   ├── README.md                      ← 参数、产物、排错、实测结果
│   ├── train.py                       ← 单任务训练（默认 --num-workers 0，Windows 安全）
│   ├── eval.py                        ← held-out seeds 评测（physx_cpu 单环境）
│   └── runs/                          ← 本机实验产物（checkpoint / log.csv / 评测 JSON）
├── train/                             ← ★ 学院 GPU（Linux, RTX 4080）入口脚本
│   ├── README.md
│   ├── train.py                       ← 默认 --device cuda / --num-workers 8，产物存 train/runs
│   └── eval.py                        ← 批量评测（--num-envs 并行环境 + --sim-backend gpu/cpu）
├── pyproject.toml                     ← 依赖声明（uv sync 或 pip install -e . 一把装好）
├── 实验三_Track3_完整教程.md           ← 主文档：全流程教程（附录 A = 三种主干结构对比实验）
├── 学院_uv安装指南.md                  ← 学院 GPU 环境搭建（uv / 原生 venv 两种）
├── DASC7606C_cleaned.txt              ← 课程项目说明（清理后的纯文本，便于检索）
├── DASC7606C_Group_Project_0921.pdf   ← 课程项目说明原文
├── intro.pdf                          ← 课程 / 项目介绍
├── .ref/                              ← 论文与官方源码对照资料 + 结构等价性验证脚本（scratch，可删）
├── wheels/  venv/  .venv/             ← 离线依赖包与虚拟环境
└── .workbuddy/                        ← 工作记录（勿删）
```

## 快速上手

```bash
# 0) 环境：本机直接用现成的 venv/ 即可（脚本会自动把仓库根挂到 sys.path，无需装包）；
#          学院 GPU 上用 uv sync（读仓库根 pyproject.toml 建 .venv），详见 学院_uv安装指南.md

# 1) 一键：检查数据 → 训练 → 评测 → 出汇总（主干默认 mlp）
python tools/run_local.py --env-id PickCube-v1

# 2) 换主干做结构对比（只换噪声预测网络，其余全不变）
python train_local/train.py --env-id PickCube-v1 --backbone unet
python train_local/eval.py  --ckpt train_local/runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --seed0 2000

# 3) 数据效率实验（1.0 / 0.5 / 0.25 / 0.1 四档）
python tools/run_local.py --env-id PickCube-v1 --data-efficiency --backbone unet

# 4) 换任务（需先按教程第 4 节生成该任务的数据集）
python train_local/train.py --env-id StackCube-v1 --backbone unet --max-episode-steps 200

# 5) 汇总 runs/ 下所有实验 → train_local/runs/SUMMARY_all.md（写报告收尾用）
python tools/summarize_runs.py --csv
```

**学院 GPU 机器（RTX 4080, Linux）**：用 `train/` 目录代替 `train_local/`（环境搭建只有两步：
装 uv → 在仓库根 `uv sync`，详见 `学院_uv安装指南.md`）。训练命令一致；评测额外支持并行仿真：

```bash
cd train
python train.py --env-id PickCube-v1 --backbone unet   # 默认 --device cuda
python eval.py  --ckpt runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --num-envs 32 --sim-backend gpu

# 汇总训练评测结果（回到仓库根目录）
python tools/summarize_runs.py --runs-dir train/runs --out train/runs/SUMMARY_all.md
```

> 注意：`train/runs` 与 `train_local/runs` 是两套独立目录，别混放；评测 JSON 文件名带
> `_gpuN`/`_cpuN` 后缀，不会覆盖本机 `cpu` 结果。对比实验仍须统一仿真后端（建议 GPU 机器
> 全部用 `--sim-backend gpu` 一组做完，或全部拷回本机用 `cpu` 评）。

前提：数据集已生成到 `~/.maniskill/demos/<Task>-v1/` 下（如
`PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`）。没有就按教程第 4 节转换/生成。

## 三条铁律（评分表的分主要在这）

| 铁律 | 说明 |
|---|---|
| **held-out 评测** | 训练 seeds 用 `0,1,2...`，评测 seeds 用 `2000~2049`，**绝不重叠** |
| **同一后端** | 所有对比必须在同一仿真后端上做（本机统一 `physx_cpu`），否则不可比 |
| **报 n_ok/n_total** | 每任务 ≥50 episode，写清 `48/50` 而不是只写百分比；失败要分类 |

## 文档导航（职责不重叠）

| 文档 | 什么时候看 |
|---|---|
| 根 `README.md`（本文件） | 想快速知道项目是什么、文件在哪、怎么跑 |
| `实验三_Track3_完整教程.md` | 要完整走一遍流程（任务选型 → 演示生成 → 训练 → 评测 → 报告）；**附录 A = 网络结构对比实验**。已精简两轮（1776 → 819 行）：历史代码块与一次性脚本正文删掉、只留结论，需要时从 `.ref/教程_v0_full.md` 恢复 |
| `train_local/README.md` | 本机代码怎么用（参数 / 产物 / 排错 / 实测数据 / 主干结构） |
| `学院_uv安装指南.md` | 要在学院 GPU 机器（RTX 4080, Linux）上用 uv 搭环境 |
| `train/README.md` | 在学院 GPU 机器上训练 / 并行评测的具体命令 |
| `scripts/`（各脚本头部 docstring） | 要下载/转换/质检演示数据，或批量评测多个 checkpoint 时 |
| `dp/__init__.py` 与各脚本头注释 | 只想看代码接口（库 ↔ 入口脚本的分工） |

## 三种主干（结构对比实验的对象）

只换**噪声预测网络**，其余（数据 / 划分 / 加噪 / 采样 / EMA / 评测协议）完全共用。
参数量为**本机实测**（PickCube-v1：obs 42 维 × To 2、act 4 维、Tp 16）：

| `--backbone` | 网络 | 参数量（实测） | 说明 |
|---|---|---|---|
| `mlp` | 展平拼接 + 3 层 MLP（`MLPNoisePred`） | **0.353 M** | 基线；论文无此档，是本项目自加的对照 |
| `unet` | 官方 `ConditionalUnet1D` 完整移植（每级 2 个残差块 + 2 个 mid 块 + ConvTranspose1d 上采样 + 时间嵌入/条件拼接后 FiLM） | **66.418 M** | DP 论文默认主干；lowdim 配置 `down_dims=[256,512,1024], k=5, n_groups=8, cond_predict_scale=True` |
| `transformer` | 官方 `TransformerForDiffusion` 完整移植（DP-T：cond memory = 时间 token + To 个 obs token，动作 token 走 TransformerDecoder 因果 self-attn + cross-attn） | **8.972 M** | 论文 Transformer 变体；`n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3, n_cond_layers=0` |

对比量级：**MLP : Transformer : UNet ≈ 1 : 25 : 188**。

> **参数口径与论文对齐**：上表对应论文 Table 8 的 `#D-params`（扩散网络本身）。论文同一行还有
> `#V-params`（视觉编码器，**image 版才有**）= 22 M —— 常被引用的"DP-T 31 M"就是 9 M + 22 M（其中 22 M 是两路 ResNet18）。
> 我们的观测是 42 维 state、没有相机，所以对应 **9 M** 这一档（实测 8.972 M）。
>
> 注意三个数字会随观测/动作维度与 `Tp` 变（UNet 的 cond 维度 = obs_dim × To；Transformer 的 cond 维度 = obs_dim）。

## 当前进展（PickCube-v1，state 观测 / `pd_ee_delta_pos`）

全部在 held-out seeds `2000..2049` 上评测（50 episode）：

| 主干 | 参数量 | 演示条数 | 总迭代数 | 成功率 | 成功/总数 | 训练用时 |
|---|---|---|---|---|---|---|
| MLP | 0.353 M | 900 | 82,200（`--epochs 300`） | 96.0% | 48/50 | ~849 s |
| 1D-UNet（**旧小模型，结构已废弃**） | 2.806 M | 900（+100 val） | 29,866 | **100%** | **50/50** | ~684 s |
| Transformer（**旧 GPT 式，结构已废弃**） | 3.348 M | 900（+100 val） | 29,866 | **100%** | **50/50** | ~971 s |
| 1D-UNet（官方移植） | 66.418 M | 900（+100 val） | **待重训** | — | — | — |
| Transformer（官方移植） | 8.972 M | 900（+100 val） | **待重训** | — | — | — |

> **值得注意**：旧版小 UNet 只用了约 **1/3 的迭代数**（29.9k vs 82.2k）就拿到 100%，明显超过 MLP 的 96% ——
> 这与文献趋势（难任务上 UNet ≫ MLP）一致，是结构对比实验的一个亮点。三种主干代码均已冒烟验证可训练。
>
> ⚠️ **2026-09-25 起 UNet 与 Transformer 主干都换成了官方实现的完整移植**（见上一节参数量表）：
> 旧版 2.806 M 的小 UNet 与 3.348 M 的 GPT 式 Transformer 已被替换，对应 checkpoint 结构不兼容、不可再加载，
> 所以表中两个「已废弃」行的 100% 只能作为历史记录（数据源：`train_local/runs/SUMMARY_all.md`），
> **引用前必须用新的 66.418 M UNet / 8.972 M Transformer 重训**。只有 MLP 那一行的代码与参数量完全没变。
