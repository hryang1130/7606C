# 本机 DP 训练流程（Windows / RTX 4060 Ti 8 GB）

本目录只有**入口脚本**（`train.py` / `eval.py`）；可复用的库在 `../dp/`，通用工具在 `../tools/`——
两边机器（本机 Windows 与学院 GPU Linux）共用同一份代码，避免漂移。

一套自包含的 Diffusion Policy 训练+评测流程，**不需要 diffusers / wandb / tensorboard，不需要 Vulkan，不需要 CUDA Toolkit**。
训练走 GPU，评测走 CPU 单环境仿真。噪声预测主干可选 **MLP / 官方 UNet / 官方 Transformer(DP-T)**。

---

## 0. 一分钟上手

```bash
# 在仓库根目录（pythonProject1/）下执行即可，脚本会自动找到 dp/ 库

:: 一键：检查数据 → 训练 → 评测 → 出汇总
python tools/run_local.py --env-id PickCube-v1 --backbone unet

:: 只想分开跑
python train_local/train.py --env-id PickCube-v1 --backbone unet
python train_local/eval.py  --ckpt train_local/runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --seed0 2000
```

（也可以 `cd train_local` 后跑 `python train.py`，sys.path 会自动把仓库根挂进来，两种方式等价。）

前提：`~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5` 已存在（你已经有 1000 条的那份）。没有的话见教程第 4 节的 `convert_all.py` / `gen_demos_scripted.py`。

---

## 1. 文件

| 文件 | 作用 |
|---|---|
| `train.py` | 单任务训练，输出 `log.csv` / `loss_curve.png` / `best.pt` / `last.pt`（默认 `--num-workers 0`，Windows 安全） |
| `eval.py` | 在 held-out seeds 上评测（成功率 + 失败分类 + JSON 明细）；`--inference-steps` 做采样步数消融 |
| `runs/` | 本机实验产物；**与农场的 `train/runs/` 分开存**，别混放 |
| `../dp/backbones.py` | 三种主干：`MLPNoisePred` / 官方 `ConditionalUnet1D` / 官方 `TransformerForDiffusion`（见第 6 节） |
| `../dp/dp_lib.py` | DP 模型（DDPM 训练 / DDIM 采样）+ 数据集（episode 切窗 + 归一化统计） |
| `../dp/utils.py` | `EMA` / `plot_curve`（两套 train.py 共用） |
| `../tools/run_local.py` | 一键串起来；支持 `--backbone`、`--data-efficiency`、`--train-dir` |
| `../tools/summarize_runs.py` | 扫描某个 `runs/` 下**全部**实验，汇总为 `SUMMARY_all.md`（`--csv` 可另导 CSV） |
| `../scripts/` | 数据侧脚本：下载/转换/自写控制器生成演示、数据集质检、批量评测、抓官方基线（见各文件头 docstring） |

---

## 2. 训练

> 本节到第 7 节的命令都假设**当前目录是 `train_local/`**（`cd train_local` 后执行）；
> 在仓库根执行也可以，把 `train.py` / `eval.py` 写成 `train_local/train.py` / `train_local/eval.py` 即可。

```bash
:: 默认：30000 iteration（对齐官方 DP baseline），主干 mlp
python train.py --env-id PickCube-v1

:: 快速试验（几十秒）
python train.py --env-id PickCube-v1 --epochs 150

:: 数据效率实验用的单档
python train.py --env-id PickCube-v1 --demo-frac 0.25 --seed 1

:: 换主干（结构对比实验，见第 6 节）
python train.py --env-id PickCube-v1 --backbone unet

:: 换任务（需要先有对应数据集）
python train.py --env-id StackCube-v1 --max-episode-steps 200
```

### 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--env-id` | `PickCube-v1` | 任务 |
| `--backbone` | `mlp` | 噪声预测主干：`mlp` / `unet` / `transformer` |
| `--total-iters` | `30000` | 总迭代数；epoch 数由它自动换算 |
| `--epochs` | 空 | 显式指定 epoch（覆盖上面的换算），快速试验用 |
| `--batch` | `256` | |
| `--lr` / `--wd` | `1e-4` / `1e-6` | AdamW + cosine 退火 |
| `--demo-frac` | `1.0` | 用多少比例的演示 |
| `--seed` | `0` | 训练 seed（影响数据划分 + 权重初始化） |
| `--obs-horizon` / `--pred-horizon` / `--action-horizon` | `2 / 16 / 8` | 即 DP 的 `To / Tp / Ta`；**改进 DP 的消融就改这三个** |
| `--hidden` / `--n-layers` | `256` / `3` | **仅 `--backbone mlp` 生效** |
| `--num-inference-steps` | `10` | 训练时写进 checkpoint 的默认 DDIM 步数（评测可用 `--inference-steps` 覆盖） |
| `--max-episode-steps` | `100` | 写进 checkpoint，评测时用。**PickCube 用 100，StackCube 用 200** |
| `--max-episodes` | `-1` | 只加载前 N 条（调试用） |
| `--device` | `auto` | `auto` → 有 CUDA 就用 |

### 产物

```
runs/<env>_frac<f>_<backbone>_seed<s>/
├── best.pt            验证 loss 最好（含 EMA 权重 + 归一化统计 + 配置）
├── last.pt
├── log.csv            epoch / train_loss / val_loss / lr / sec
├── loss_curve.png     直接贴进报告
└── train_summary.json
```

---

## 3. 评测

```bash
:: 标准：50 个 episode，held-out seeds
python eval.py --ckpt runs\PickCube-v1_frac1.0_mlp_seed0\best.pt -n 50 --seed0 2000

:: 对比 EMA vs 原始权重（报告里的一个小实验）
python eval.py --ckpt ...\best.pt -n 20 --use-raw
```

输出：终端逐 episode 日志 + `eval_seed2000_n50.json`（含每个 episode 的 seed / success / steps）。

**评测铁律**（完整三条见根 `README.md`）：训练 seeds `0,1,2...` 与评测 seeds `2000..2049` **绝不重叠**；每任务 ≥50 episode 且报告写 `n_ok/n_total`；失败分类（超时 / 提前终止）由 `eval.py` 自动统计。

---

## 4. 一键流程 & 数据效率实验

```bash
:: 单档：训练 + 评测 + 出 SUMMARY
python tools/run_local.py --env-id PickCube-v1 --backbone unet

:: 数据效率曲线：1.0 / 0.5 / 0.25 / 0.1 四档
python tools/run_local.py --env-id PickCube-v1 --backbone unet --data-efficiency

:: 只打印命令不执行
python tools/run_local.py --env-id PickCube-v1 --backbone unet --dry-run
```

数据效率实验会输出 `runs/SUMMARY_data_efficiency.md`，形如：

| 演示条数 | demo_frac | 成功率 | 成功/总数 | 平均步数 | best val loss | 训练用时(s) |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

> 做数据效率实验时注意两点：**每档用相同的 `--total-iters`**（别用"每档都训 300 epoch"，那样小数据档反而迭代更多，变量就不干净了），以及**每档至少 2~3 个训练种子**（换种子能差 30+ 个百分点）。

### 全量汇总报告 `tools/summarize_runs.py`

`tools/run_local.py` 只汇总**它本次跑到的那几档**（文件名固定为 `SUMMARY_<env>.md`）。要把 `runs/` 下**已经存在的所有实验**（跨任务、跨主干、跨 seed）一次性汇总，用这个：

```bash
python tools/summarize_runs.py                             # 默认扫 train_local/runs → SUMMARY_all.md
python tools/summarize_runs.py --env-id PickCube-v1        # 只看某个任务
python tools/summarize_runs.py --runs-dir train/runs       # 扫学院 GPU 那套结果
python tools/summarize_runs.py --out runs/REPORT.md --csv  # 指定输出路径 + 额外导出同名 .csv
```

报告共 5 节，可直接贴进报告：

| 节 | 内容 |
|---|---|
| 1. 总览 | 每个实验一行：主干 / demo_frac / 参数量 / iters / 用时 / **iters/s** / best val / 成功率 |
| 2. 按任务对比 | 按任务分组，自动标出**最佳主干**；多种主干时会提示"已横向对比 N 种" |
| 3. 评测明细 | 每个 `eval_*.json` 一行 + **失败分类**（超时 / 提前终止）与失败 seeds 列表 |
| 4. 训练配置明细 | obs_mode / control_mode / backend / To-Tp-Ta / max_steps / DDIM 步数 |
| 5. 数据完整性提示 | 缺 `train_summary.json`、缺评测 JSON、目录名不规范等都会点名 |

细节：目录名两种约定都认（`<env>_frac<f>_seed<s>` 与带主干的），**以 `train_summary.json` 里的 cfg 为准**；同一个实验目录里的多份评测（如 `_steps25`、`_raw`）会全部列出。

---

## 5. 「改进 DP」的消融开关（报告加分项）

Track 3 一般除了复现 baseline，还要至少一个改进/消融。下面这些开关**都已在代码里接好**，改一个参数就是一组对照实验，不用改代码。

| 实验 | 命令 | 对比的变量 |
|---|---|---|
| **主干结构** | `python train.py --env-id PickCube-v1 --backbone mlp` / `unet` / `transformer` | 噪声预测网络（**见第 6 节**） |
| 观测历史 `To` | `python train.py --env-id PickCube-v1 --obs-horizon 1` / `2` / `4` | 用几帧历史做条件 |
| 预测长度 `Tp` | `python train.py ... --pred-horizon 8` / `16` / `32` | 一次预测多长的动作块 |
| 执行长度 `Ta` | `python train.py ... --action-horizon 4` / `8` / `16` | receding horizon 每次真正执行几步 |
| **DDIM 采样步数** | `python eval.py --ckpt runs\PickCube-v1_frac1.0_mlp_seed0\best.pt -n 50 --inference-steps 10` / `25` / `50` | 采样精度 vs 成功率（不改训练） |
| 数据效率 | `python tools/run_local.py --env-id PickCube-v1 --backbone unet --data-efficiency` | 演示条数 1.0/0.5/0.25/0.1 |
| 权重平均的作用 | `python eval.py --ckpt ... --use-raw` | EMA 权重 vs 原始权重 |

> `--inference-steps` 是**只改评测、不动训练**的开关：拿同一个 checkpoint，采样步数越多通常越稳、但越慢。注意：96% 的高基线下它提升空间有限（天花板效应），在欠训练的 checkpoint 上对比才有区分度。
> 结果 JSON 会自动带 `_steps25` 这样的后缀，不会覆盖默认评测。
> **所有对比必须用同一批 held-out seeds（默认 `2000..2049`）**，否则不可比。

---

## 6. 网络结构对比实验（三种主干）

> 完整方案（文献盘点、工作量评估、报告写法）见 `实验三_Track3_完整教程.md` 的 **附录 A：DP 网络结构对比实验（三种主干）**。

`--backbone` 只换噪声预测网络，**其余一切不变**（同一份数据、同一个划分 seed、同样迭代数、同一批评测 seeds）。

```bash
:: 训练
python train.py --env-id PickCube-v1 --backbone unet
python train.py --env-id PickCube-v1 --backbone transformer

:: 评测（eval.py 零改动，主干信息存在 checkpoint 里自动识别）
python eval.py --ckpt runs\PickCube-v1_frac1.0_unet_seed0\best.pt        -n 50 --seed0 2000
python eval.py --ckpt runs\PickCube-v1_frac1.0_transformer_seed0\best.pt -n 50 --seed0 2000
```

| `--backbone` | 主干 | 参数量（实测） | 说明 |
|---|---|---|---|
| `mlp` | 展平拼接 + 3 层 MLP | **0.353 M** | 默认，baseline |
| `unet` | **官方 ConditionalUnet1D 完整移植**（每级 2 个残差块 + 2 个 mid 块 + ConvTranspose1d 上采样 + 时间嵌入/条件拼接后 FiLM） | **66.418 M** | DP 论文的默认主干（A1：无 obs encoder，global_cond = 归一化 obs 展平） |
| `transformer` | **官方 `TransformerForDiffusion` 完整移植**（DP-T：cond memory = 时间 token + To 个 obs token 经 MLP 编码，动作 token 走 TransformerDecoder 因果 self-attn + cross-attn） | **8.972 M** | 论文 DP-T 主干（无 obs encoder，cond = 归一化 obs[:, :To]）；默认 `n_layer=8, n_head=4, n_emb=256, p_drop_attn=0.3, causal_attn=True, n_cond_layers=0` |

量级对比：**MLP : Transformer : UNet ≈ 1 : 25 : 188**（同一任务、同一 To/Tp 下的实测值；换任务或改 `obs_dim` / `Tp` 会变）。

要点：

- **`eval.py` 零改动**：`backbone` 随 `model_kwargs` 存进 checkpoint，反序列化时自动用对的主干；旧的 MLP checkpoint 也能直接加载（已实测）。
- **UNet 超参**在 `train.py` 里用 `--unet-down-dims 256 512 1024`、`--unet-kernel-size 5`、`--unet-n-groups 8`、`--unet-step-embed-dim 256`、`--unet-cond-predict-scale/--no-unet-cond-predict-scale` 调（默认即论文 lowdim 配置）。
- **Transformer 超参**同理：`--tf-n-layer 8`、`--tf-n-head 4`、`--tf-n-emb 256`、`--tf-p-drop-emb 0.0`、`--tf-p-drop-attn 0.3`、`--tf-causal-attn/--no-tf-causal-attn`、`--tf-n-cond-layers 0`（默认即论文 lowdim 配置）。
- **接口与官方一致**：unet 用 `forward(sample, timestep, global_cond=...)`，transformer 用 `forward(sample, timestep, cond=...)`（cond 是 `(B, To, obs_dim)` 的逐步观测），时间嵌入都在主干内部做，`DiffusionPolicy.eps()` 按 backbone 分发；只有 mlp 仍是 `forward(x, t_emb, c)`。
- ⚠️ **2026-09-25 起 UNet 与 Transformer 主干都换成了官方结构**（旧版分别是 2.806 M 的小型 UNet 和 3.348 M 的 GPT 式 Transformer）。旧的 `runs/*_unet_*/best.pt`、`runs/*_transformer_*/best.pt` 结构不兼容、无法再加载，旧成绩需重训；只有 mlp 的旧 checkpoint 不受影响。
- 实验目录名自带主干，三组结果互不覆盖。
- **公平性红线**：做结构对比时别只跑一个训练种子（换种子能差 30+ 个百分点），并且必须固定数据/划分/迭代数/评测 seeds 不变——只让 `--backbone` 变。
- `tools/run_local.py` 现在支持 `--backbone`（目录名与 `train.py` 一致，含主干），`--train-dir train` 还能直接驱动农场那套入口脚本。
- **论文参数口径（答辩常被问）**：论文 Table 8 里 DP-T 的 `#D-params`（扩散网络）= **9 M**，另有 `#V-params`（视觉编码器，**image 版才有**）= 22 M，两者相加 31 M。我们的观测是 state、没有相机，所以对应的是 **9 M** 那一档（实测 8.972 M）——别把 22 M 的视觉编码器算进来。
- **与论文 lowdim 配置的剩余差异**（都为训练/评测侧全局项，尚未对齐）：obs 归一化用 z-score（论文用 min/max → [-1,1]）、推理默认 10 步（论文 lowdim 配置 100 步）、AdamW betas/warmup 与 EMA 动态衰减（论文 transformer 用 wd 1e-3 + 1000 步 warmup）。改这些会同时影响三种主干，属于另一组消融。

---

## 7. 多任务批量（本地跑通后）

```bash
python train.py --env-id PickCube-v1  --backbone unet
python train.py --env-id PushCube-v1  --backbone unet
python train.py --env-id StackCube-v1 --backbone unet --max-episode-steps 200
```

改个小 shell/bat 把「train + eval」串起来过夜跑最省事。**评测必须在同一后端（都 `cpu`）上做**，否则结果不可比。

---

## 8. 本机环境事实（为什么这么写）

| 事实 | 影响 |
|---|---|
| RTX 4060 Ti 8 GB，torch 2.14.0+cu126，`cuda.is_available()=True` | 训练走 GPU；模型 0.35 M（MLP）/ 8.97 M（Transformer）/ 66.4 M（UNet），batch 256 峰值显存实测 ≤ 2.2 GB |
| `physx_cuda` 报 `Could not find module 'cuda.dll'`（没装 CUDA Toolkit） | **不能**用 GPU 并行仿真 |
| `physx_cpu` 不支持 `num_envs>1` | 评测只能单环境串行 |
| 无 Vulkan | 不能渲染/录视频（所以 `eval.py` 只跑数值评测） |
| CPU 仿真实测约 60~200 步/秒 | 50 episode × 100 步 ≈ 25~80 秒，完全可接受 |
| `max_episode_steps` 默认只有 50 | **必须显式传**，脚本已处理 |

---

## 9. 实测结论（PickCube-v1，state 观测 / `pd_ee_delta_pos`）

> 具体成绩表（MLP 60%→96%、旧小 UNet 100% 等）统一放在根 `README.md` 的「当前进展」，这里只记两条结论。

> 💡 **不要用短训练的结果下结论。** 实测：只训 150 epoch（约 4000 iter，官方 1/7）时，动作方向已学对，但夹爪通道会输出 ~0.2~0.7 的中间值而不是 ±1，导致抓取失败、成功率为 0。**夹爪是双峰信号，欠训练时会退化成两峰平均** —— 这是 DP 的典型现象，务必训满再看结果。
>
> **瓶颈定位（三行数据对照）**：迭代数固定 30k 时，数据 200→900 只 +4 pt（噪声内）；同样约 900 条数据，迭代数 30k→82k 直接 64%→**96%**。→ **主要瓶颈是训练迭代数，不是数据量。** 报告里最值得画的一张图：固定数据，扫 `--total-iters`（如 10k / 20k / 40k / 80k）画成功率曲线，正好定量复现上面的"欠训练→夹爪退化"现象。

---

## 10. 排错

| 现象 | 原因 | 处理 |
|---|---|---|
| `没找到 xxx 的 state 数据集` | 数据集不在 `~/.maniskill/demos/<env>/` 下 | 跑 `convert_all.py` 或 `gen_demos_scripted.py` |
| 成功率全 0，训练 loss 却不高 | 欠训练（夹爪退化成两峰平均） | 训满 `--total-iters 30000` 或更多；或提高 `--inference-steps` |
| 成功率全 0 且 episode 都跑满 max_steps | 演示太长 / `max_episode_steps` 太小 | 按官方 `baselines.sh`：PickCube 100、StackCube 200、PegInsertionSide 300 |
| 验证 loss 低但成功率 0 | 数据划分泄漏 | 已按 episode 划分；别自己改成按单步切 |
| UNet 报 `Tp 必须能被 2^(len(down_dims)-1) 整除` | `Tp` 不是 4 的倍数（默认 down_dims 三级） | 用 `Tp=8/16/32`，或 `--unet-down-dims 256 512` 减一档 |
| `未知 backbone: xxx` | `--backbone` 拼错 | 只能填 `mlp` / `unet` / `transformer` |
| `CUDA out of memory` | batch 太大 | `--batch 128`，或把 `--unet-down-dims` / `--tf-n-emb` 调小（官方结构在 8GB 4060 Ti 上 batch 256 实测：UNet 峰值 ~2.2 GB、Transformer ~1.0 GB，一般不会 OOM） |
| 训练 loss 变 NaN | 低方差观测维被放大 | 已在 `compute_stats` 里做 `std<1e-3 → 1.0` 保护；若仍 NaN 检查数据集是否有异常值 |

---

## 11. 两台机器都能跑：跨平台清单

代码层已经做成"一份库 + 两套入口"，本机（Windows / 4060 Ti）与学院 GPU（Linux / 4080）行为一致：

| 关注点 | 本机 Windows | 学院 GPU（Linux） | 代码里的处理 |
|---|---|---|---|
| 库代码 | `dp/` | 同一份 `dp/` | `train_local/` 与 `train/` 都 `import dp.*`，脚本自动把仓库根挂进 `sys.path`，**装不装包都能跑** |
| 依赖 | 现成 `venv/`（也可 `uv sync`） | `uv sync`（读仓库根 `pyproject.toml`） | 同一份依赖声明；torch 走 cu126 官方源 |
| DataLoader | 必须 `--num-workers 0` | 可 `--num-workers 8` | 两套 `train.py` 各自设默认值（本机 0、农场 8） |
| 画图 | 无显示环境 → Agg | 同左 | `dp/utils.py::plot_curve` 强制 `matplotlib.use("Agg")`，没装 matplotlib 只警告不中断 |
| 评测后端 | `physx_cpu` 单环境（无 CUDA Toolkit） | 可 `--sim-backend gpu --num-envs 32` | **同一组对比必须同后端**，否则结果不可比 |
| 路径写法 | 文档示例里可能写 `runs\...` | Linux 用 `runs/...` | 代码**全部**用 `osp.join`，无平台专属拼接；两套约定（仓库根 / `train_local` 内）见第 2 节开头 |
| 文件编码 | UTF-8（含中文注释） | 同左 | Linux 直接跑；本机若控制台乱码，`chcp 65001` 或用 `PYTHONUTF8=1` |
| 评测 seeds | `2000..2049` | 同左 | 训练 seeds `0,1,2...` 与评测 seeds **绝不重叠** |
| 改了目录名 / 搬家 | — | — | venv 里有 3 处写死的绝对路径（`activate.bat` / `pyvenv.cfg` / `_pinocchio_dlls.pth`）：跑 `python tools/fix_venv_paths.py --apply` 修，再在 PyCharm 里重选一次解释器。**代码本身没有绝对路径，不用改** |

自检命令（两台机器都可以先跑这两条，确认环境与代码都没问题）：

```bash
# 1) 库与入口脚本能导入（不训练）
python -c "import dp; print(dp.backbones.BACKBONES)"
python train_local/train.py --help > $null     # Linux 上用 /dev/null

# 2) 真机冒烟：12 条演示、1 epoch（几十秒，产物在 runs/_smoke/ 下）
python train_local/train.py --env-id PickCube-v1 --backbone unet \
    --out train_local/runs/_smoke --exp-name smoke --epochs 1 --max-episodes 12
```

> 冒烟产物体积不小（官方 UNet 每个 ckpt ≈ 500 MB），跑完记得删掉 `runs/_smoke/`。
