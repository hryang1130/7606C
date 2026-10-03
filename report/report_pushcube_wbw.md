# PushCube：数据量（轨道 A）与主干（轨道 B）正式结果

> DASC7606C · Group 11 · scaling + backbone track
> 作者：王博文（学号【待补】）
> 实验平台：HKU HPC（`gpu2gate1.cs.hku.hk`），账户 `u3684254`，工作目录 `~/dp-manip`，结果根目录 `$RUN_ROOT=/userhome/cs5/u3684254/dp-manip/runs`
> 报告日期：2026-10-03

负责任务：Task 3 / `TASK=pushcube`。本文只报告**测试种子**（10000–10099，100 回合）上的
闭环成功率，checkpoint 一律取 `final.pt`（100k 步）。

## 0. 摘要（TL;DR）

1. **数据量**：UNet 的成功率随示范条数单调上升，**45.3% → 82.3% → 94.8% → 96.0%**
   （N = 25 / 50 / 100 / 200）。相邻档两层配对 bootstrap：25→50 = **+0.370 [+0.303, +0.437]**、
   50→100 = **+0.130 [+0.080, +0.183]**，区间均不含 0；**100→200 = +0.012 [-0.014, +0.038]**，
   区间含 0 → **N=100 已平台**，按 `final-plan` §5 的预注册规则，**N=400 不触发**。
2. **主干**：在同样的 N=100 上，**UNet 0.948 ± 0.012**、**MLP 0.922 ± 0.019**、
   **Transformer 0.890 ± 0.006**。配对 bootstrap：Transformer−UNet = **-0.058 [-0.090, -0.026]**
   （显著更差）；MLP−UNet = -0.026 [-0.062, +0.008]（区间含 0，**与 UNet 分不开**）；
   MLP−Transformer = +0.032 [-0.002, +0.066]（临界）。
3. **效率**：MLP 以 **11.7M 总参数量**（UNet 80.8M 的 1/7）、**98 ms 单次推理**
   （UNet 527 ms 的 1/5.4）、**1.28 GB 峰值显存**，拿到与 UNet 分不开的成功率。
4. **与 PickCube 的对照是本报告最重要的组内结论**：同规格、同预算下，PushCube 在 N=25
   就有 45.3%（PickCube 0.7%）、N=100 即平台（PickCube 到 N=200 仍在涨）。两个任务的
   "数据量↔主干"结论方向不同（PickCube 换主干收益远大于加数据；PushCube 加数据到 100
   即够、换主干无收益），**任何跨任务外推都不成立**。
5. **过拟合存在但无害**：UNet 把训练集背到 train loss ≈ 4.5e-4，val loss 从 10k 步的
   0.011 单调升至 100k 步的 0.036——与 PickCube 的 UNet 同款过拟合曲线，但 PushCube
   简单到"背下来也能泛化"，UNet 仍是成功率最高的主干（PickCube 上 UNet 过拟合 = 最差主干）。
   val loss 跨主干排序（MLP 0.033 < UNet 0.036 < Transformer 0.058）与成功率排序不一致，
   **跨结构 val loss 不可比**；同主干内 val loss 随 N 单调降（0.091 → 0.018）则与成功率一致。
6. **成功 ≠ 保持**：`success_at_end` 上 **MLP 0.672 > Transformer 0.628 > UNet 0.536**——
   UNet 进目标区最多但滑出率也最高（once→end 掉 0.41；MLP 只掉 0.25）。

## 1. 实验设置

| 项目 | 值 |
| --- | --- |
| 任务 | `pushcube` / `PushCube-v1`，控制模式 `pd_ee_delta_pos`（4 维动作） |
| 回合长度 | 100 步（`configs/tasks/pushcube.toml`） |
| 仿真后端 | `physx_cpu`（与示范数据一致） |
| 观测 | RGB 128×128×3（ResNet-18，feature 128，`share_camera_encoder`）+ 非特权 proprio |
| 数据 | 官方 demogen 训练示范池 400 条（种子 0–3999，取前 N 条，嵌套 25⊂50⊂100⊂200）；验证示范 50 条（4000+） |
| 训练预算 | 100k optimizer steps，batch 64，AdamW lr 1e-4 + betas(0.95, 0.999)，weight decay 1e-6，cosine + 500 warmup，grad clip 1.0，AMP，DDPM 100 步，EMA |
| 训练种子 | N=25/50 用 1–3；N=100/200 与三个主干用 1–5 |
| 评估 | `SPLIT=test`，回合 10000–10099，`NUM_ENVS=4`，`CHECKPOINT=final.pt` |
| 数据根 | `/userhome/cs5/u3684254/maniskill-demogen/data/dataset`（只读） |
| 运行根 | `/userhome/cs5/u3684254/dp-manip/runs` |
| 代码 | `~/dp-manip`，commit `646117eb3f7538f814922c62a2729373919d424d`（branch `main`，`dirty=false`） |

轨道 B 的自变量只有 `policy.backbone`：三种主干的 resolved config 除该键及其结构键外
完全相同（Gate B 已核对，见 §2）。三种主干的总参数量（含共享 ResNet-18 编码器）见 §5；
裸 UNet 主干为官方 66.4M ConditionalUnet1D。容量差约 7 倍，是 `docs/final-plan.md` §6
预先接受的代价，结论只能限定为「在这三个具体实现之间」。

### 1.1 交接信息（对应 `instructions.md` §13）

| 项目 | 值 |
| --- | --- |
| `TASK` | `pushcube`（Task 3） |
| Git commit | `646117eb3f7538f814922c62a2729373919d424d`（分支 `main`） |
| `DATA_ROOT` | `/userhome/cs5/u3684254/maniskill-demogen/data/dataset` |
| `RUN_ROOT` | `/userhome/cs5/u3684254/dp-manip/runs` |
| 训练作业 | data_size = **135573**；backbone = **135625 → 135661 → 135664**（三次接力，全部完成） |
| 评估作业 | final = **135621**；诊断 10k/30k/60k = **135622 / 135623 / 135624**；backbone = **135681** |
| Gate B 输出 | `data_size: 16 cells ok, control_hash=cf9defc88f09`；`backbone: 15 cells ok, control_hash=bf998cb5b6bd` |
| 文件齐全性 | `summary.json` / `checkpoints/final.pt` / `eval/test_final.json` 均为 **26/26** |
| 人工干预 / 异常 | backbone 训练由三次作业接力完成（135625 未跑满全部新格，135661/135664 走官方断点续跑补齐）；无 failed、无 requeue |

## 2. 验收与完整性

| 检查 | 结果 |
| --- | --- |
| data-size 训练作业 135573 | `completed: 16 skipped: 0 failed: 0 interrupted: 0` |
| backbone 训练作业 135625+135661+135664 | 10 个新格全部完成（5 个 UNet N=100 复用轨道 A） |
| data-size 评估作业 135621 | `completed: 16 skipped: 0 failed: 0 interrupted: 0` |
| 诊断评估 135622/135623/135624 | 各 16 格完成（ckpt = step_010000 / 030000 / 060000） |
| backbone 评估作业 135681 | `completed: 10 skipped: 0 failed: 0 interrupted: 0` |
| Gate B（data_size） | `16 cells ok, control_hash=cf9defc88f09` |
| Gate B（backbone） | `15 cells ok, control_hash=bf998cb5b6bd` |
| 文件计数 | `summary.json` 26 / `checkpoints/final.pt` 26 / `eval/test_final.json` 26 |

三种计数的期望值都是 26（16 个 UNet + 5 个 Transformer + 5 个 MLP），实测一致。

## 3. 轨道 A：数据量（UNet）

### 3.1 汇总

| N | seed 数 | `success_once` 均值 ± SD | `success_at_end` 均值 | `return` 均值 |
| --- | --- | --- | --- | --- |
| 25 | 3 | 0.453 ± 0.026 | 0.430 | 15.2 |
| 50 | 3 | 0.823 ± 0.012 | 0.657 | 27.7 |
| 100 | 5 | **0.948 ± 0.012** | 0.536 | 30.7 |
| 200 | 5 | **0.960 ± 0.017** | 0.552 | 31.0 |

相邻档两层配对 bootstrap（重采样训练种子 + 重采样测试回合，10000 次，RNG seed=0；
配对只在两组共有的训练种子上进行）：

```text
25→50    = +0.370   95% CI [+0.303, +0.437]   → 显著（区间不含 0）
50→100   = +0.130   95% CI [+0.080, +0.183]   → 显著（区间不含 0）
100→200  = +0.012   95% CI [-0.014, +0.038]   → 不显著（区间含 0）→ N=400 按规则不触发
```

### 3.2 逐 run

| run | `success_once` | `success_at_end` |
| --- | --- | --- |
| `unet_n25_s1` / `s2` / `s3` | 0.49 / 0.43 / 0.44 | 0.46 / 0.41 / 0.42 |
| `unet_n50_s1` / `s2` / `s3` | 0.82 / 0.81 / 0.84 | 0.66 / 0.65 / 0.66 |
| `unet_n100_s1` … `s5` | 0.96 / 0.96 / 0.94 / 0.95 / 0.93 | 0.53 / 0.52 / 0.56 / 0.53 / 0.54 |
| `unet_n200_s1` … `s5` | 0.94 / 0.97 / 0.97 / 0.98 / 0.94 | 0.55 / 0.55 / 0.56 / 0.55 / 0.55 |

单 run 100 回合的 Wilson 95% 半宽：50% 档 ±0.096、90% 档 ±0.060、95% 档 ±0.045——
单点数字必须带区间，均值用 seed 间 SD。

### 3.3 训练中途的成功率（诊断评估，10k/30k/60k 检查点）

| N | @10k 步 | @30k 步 | @60k 步 | @100k（final） |
| --- | --- | --- | --- | --- |
| 25 | **0.593** | 0.500 | 0.483 | 0.453 |
| 50 | 0.860 | 0.830 | 0.863 | 0.823 |
| 100 | 0.886 | 0.894 | 0.926 | 0.948 |
| 200 | 0.880 | 0.924 | 0.958 | 0.960 |

N=25 在 10k 步就到顶、之后**单调下滑**——25 条示范下 100k 步已经训练过头；
N≥100 的曲线随步数单调上升，100k 步仍在涨。这是「固定 100k 步、只报 final.pt」
口径对最小数据档最不利的证据（§7）。

### 3.4 过拟合与 val loss（UNet）

| N | train loss@100k | val@10k | val@100k |
| --- | --- | --- | --- |
| 25 | ~3e-4 | 0.031 | 0.091 |
| 50 | ~4e-4 | 0.016 | 0.050 |
| 100 | **4.5e-4** | 0.011 | 0.036 |
| 200 | ~3e-4 | 0.008 | **0.018** |

train loss 全部压到近零（背诵），但 val loss 随 N 单调下降（0.091 → 0.018）、
与成功率排序完全一致——同一主干内 val loss 是可靠的 N 档代理指标；
N=200 的 val 仍比 N=100 低一半，说明数据侧还有余量，但成功率已平台（见 §3.1）。

## 4. 轨道 B：主干（N=100）

| 主干 | `success_once` 均值 ± SD | `success_at_end` 均值 | `return` 均值 |
| --- | --- | --- | --- |
| **UNet (B0)** | **0.948 ± 0.012** | 0.536 | 30.7 |
| MLP (B2) | 0.922 ± 0.019 | **0.672** | 29.9 |
| Transformer (B1) | 0.890 ± 0.006 | 0.628 | 31.0 |

逐 run（`success_once`）：

| run | s1 | s2 | s3 | s4 | s5 |
| --- | --- | --- | --- | --- | --- |
| `transformer_n100` | 0.89 | 0.90 | 0.88 | 0.89 | 0.89 |
| `mlp_n100` | 0.95 | 0.92 | 0.92 | 0.89 | 0.93 |

与 B0（UNet）的配对 bootstrap（同一批训练/测试口径，重采样 seed + 回合，10000 次）：

```text
Transformer − UNet = -0.058   95% CI [-0.090, -0.026]   → 显著更差
MLP         − UNet = -0.026   95% CI [-0.062, +0.008]   → 区间含 0，分不开
MLP   − Transformer = +0.032   95% CI [-0.002, +0.066]  → 临界（区间含 0）
```

**结论**：UNet ≥ MLP > Transformer（按 success_once）。但注意 §0 第 6 条：
`success_at_end` 的排序恰好相反（MLP 0.672 > Transformer 0.628 > UNet 0.536）——
UNet 进目标区最多、滑出也最多。若报告口径只看 `success_once`（final-plan 主口径），
结论是"UNet 与 MLP 打平、Transformer 显著落后"。

## 5. 效率指标（同一型号 GPU：RTX 4080 SUPER）

| 主干 | 总参数量 | 训练 100k 步（单 run，5 seed 均值） | 峰值显存 | 单次推理 |
| --- | --- | --- | --- | --- |
| UNet | 80.8M（主干 66.4M） | 1.79 ± 0.01 h | 2,706 MB | 527 ms |
| Transformer | 20.2M | 1.77 ± 0.04 h | 1,506 MB | 608 ms |
| MLP | **11.7M** | **1.65 ± 0.19 h** | **1,278 MB** | **98 ms** |

参数量为含共享 ResNet-18 编码器的**总策略参数量**（run.json `num_params`）。
三主干训练耗时几乎相同（瓶颈在数据加载与编码器）；推理上 MLP 快 5.4 倍——
单 run 100 回合共 325 次策略调用，UNet 约 171 s、MLP 只需 32 s。
**效率表必须与成功率并列给出**，否则"UNet 最弱"会被误读成"大模型必然差"。

## 6. 关键发现：过拟合在不同任务上后果相反

| 主干（N=100） | train loss@100k | val@10k | val@30k | val@60k | val@100k |
| --- | --- | --- | --- | --- | --- |
| UNet | **4.5e-4**（背诵） | 0.011 | 0.017 | 0.024 | 0.036 |
| Transformer | 1.79e-2 | 0.025 | 0.031 | 0.044 | 0.058 |
| MLP | 4.1e-3 | 0.029 | 0.018 | 0.023 | 0.033 |

1. UNet 与 PickCube 上行为一致：train loss 近零、val loss 一路上升——**同一套过拟合
   签名**。但 PickCube 上 UNet 因此成为最差主干（成功率 0.192），PushCube 上 UNet 仍是
   最高主干（0.948）。任务难度决定"背诵"的后果，不能把"UNet 过拟合 → UNet 差"
   当作跨任务规律。
2. 跨主干 val loss **不可比**：val 排序（MLP 0.033 < UNet 0.036 < Transformer 0.058）
   与 success_once 排序不一致；同主干内（§3.4）val 排序才与成功率一致。
3. Transformer 的 train loss 未收敛到背诵水平（1.79e-2），是三个主干中唯一
   "欠拟合"的，与其成功率垫底对应。

## 7. 局限性与效度威胁

1. **容量不对齐（已在 `final-plan` §6 预注册接受）**：80.8M : 20.2M : 11.7M 相差约 7 倍，
   本文只能主张「这三个具体实现之间」的差异。
2. **固定 100k 步口径**：§3.3 显示 N=25 的最优泛化点在 10k 步附近，final.pt 对最小
   数据档系统性低估；协议规定只报 `final.pt`，报告里必须写明。
3. **N < 100 只测了 UNet**：Transformer / MLP 在 N=25/50 的表现未知。"PushCube 上
   主干排序在低数据量是否翻转"没有直接答案。
4. **单任务、简单任务**：全部结论只在 PushCube-v1 + RGB 上成立；本任务 N=100 即饱和，
   "加数据"或"换主干"的收益空间都很小，与 PickCube（N=200 仍在涨、换主干 +0.71）
   结论方向相反，**组内汇总时两个任务必须分开叙事**。
5. **评测统计精度**：单 run 100 回合 Wilson 半宽 ±0.045–0.096；`success_once` 与
   `success_at_end` 是两个口径，不能混用比较。本任务两者差距巨大（UNet N=100 平均
   每 run 有 41/100 回合"进过目标区但结束时滑出"），失败分类必须引用 at_end。

## 8. 结论

1. PushCube 上 UNet 的成功率随示范条数单调上升（45.3% → 96.0%），**N=100 平台**，
   100→200 的 bootstrap 区间含 0，按预注册规则 **N=400 不触发**。
2. 同数据同预算下，success_once 排序 **UNet 0.948 ≥ MLP 0.922 > Transformer 0.890**；
   MLP 与 UNet 分不开（CI 含 0），而参数量 1/7、推理快 5.4 倍、显存省一半——
   PushCube 场景下 **MLP 是性价比最优**，UNet 只在追求 success_once 上限时值得。
3. 机制是"背诵无害"：UNet 过拟合签名与 PickCube 相同，但任务简单到背诵即可泛化；
   跨主干 val loss 不可比，同主干内 val loss 是可靠的 N 档代理。
4. `success_at_end` 揭示 UNet"进得去、留不住"（滑出率 0.41），MLP 保持最好（0.672）——
   失败分类与结论口径必须写明用的是哪个成功率。
5. 报告结论限定为「这三个具体实现之间」，并同时给出参数量与耗时；同时说明
   N < 100 的主干数据缺失，以及固定 100k 步口径对最小数据档的低估。

## 附录 A：复现命令

```bash
cd ~/dp-manip
export TASK=pushcube
export DATA_ROOT="$HOME/maniskill-demogen/data/dataset"
export RUN_ROOT="$HOME/dp-manip/runs"

# 1) 训练轨道 A（16 run）
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch

# 2) 训练轨道 B（新增 10 run，UNet N=100 复用）
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/backbone.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch

# 3) 闭环评估（final + 三个诊断检查点）
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT="$RUN_ROOT",CHECKPOINT=final.pt,SPLIT=test,NUM_ENVS=4 \
  slurm/eval_dual_gpu.sbatch
sbatch --open-mode=append \
  --export=ALL,TASK="$TASK",EXPERIMENT=configs/experiments/backbone.toml,RUN_ROOT="$RUN_ROOT",CHECKPOINT=final.pt,SPLIT=test,NUM_ENVS=4 \
  slurm/eval_dual_gpu.sbatch

# 4) 验收
.venv/bin/python scripts/sweep.py plan --experiment configs/experiments/data_size.toml \
  --task "$TASK" --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"
.venv/bin/python scripts/check_experiment.py --experiment data_size \
  --task "$TASK" --data-root "$DATA_ROOT" --run-root "$RUN_ROOT"
.venv/bin/python scripts/check_experiment.py --experiment backbone \
  --task "$TASK" --data-root "$DATA_ROOT" --run-root "$RUN_ROOT"
```

条件档（N=400）按 §3.1 的规则**不触发**，未使用。

## 附录 B：文件清单与计数

| 路径 | 数量 | 说明 |
| --- | --- | --- |
| `$RUN_ROOT/<run>/checkpoints/final.pt` | 26 | 正式结果使用的 checkpoint |
| `$RUN_ROOT/<run>/summary.json` | 26 | final step、最终 loss、耗时、峰值显存 |
| `$RUN_ROOT/<run>/eval/test_final.json` | 26 | 100 回合逐回合结果 + 汇总 |
| `$RUN_ROOT/<run>/eval/test_step_{010000,030000,060000}.json` | 48 | 仅 16 个 UNet 格子的诊断评估 |
| 本地报告副本 | — | `results_20260929/grid_new_16cells/`、`backbone/`（与服务器同构） |

## 附录 C：机时统计

每个 run 占一张卡，按各 run 墙钟（summary.json）累加：

| 作业 | run 数 | GPU-min |
| --- | --- | --- |
| 训练 data_size（135573） | 16 | 1,680 |
| 训练 backbone（135625/135661/135664） | 10 | 1,027 |
| 评估 final + 诊断 + backbone（135621–135624、135681） | 74 次 | 216 |
| **合计** | 26 run + 74 eval | **≈ 2,922 GPU-min（48.7 GPU-h）** |

参考：本节点 UNet 单 run（100k 步）约 107 GPU-min；评估单次约 3.4 min。
