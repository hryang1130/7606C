# vision_pool N=100 实验操作记录（UNet：avg vs spatial_softmax）

这是一份**操作记录**（为什么起这个实验、怎么起的、job 号、后面怎么评），不是结果报告。
数字出来后并入 `report_pickcube_yhr.md`。

## 1. 动机

PickCube / RGB / UNet 在 N=100 的 `success_once` 只有 **0.192 ± 0.018**，而同一批数据、同一预算
下 Transformer **0.906**、MLP **0.796**（见 `report_pickcube_yhr.md` §4）。数据与评测流程没有问题，
UNet 是唯一把训练集背到近零、验证 loss 单调恶化的主干。

先把两个"外部"解释排掉：

| 假设 | 检验 | 结论 |
| --- | --- | --- |
| UNet 结构和论文不一样 | 逐比特对照仓库 `conditional_unet1d.py` 与官方 `real-stanford/diffusion_policy`：同配置下 state_dict 键/形状全等，灌同权重同输入输出 `max abs diff = 0` | **排除**，结构一致 |
| `final.pt` 过了最优，取 best checkpoint 能救回来 | job **136096**：N=100 的 s1–s5 各评 10k/30k/60k/100k | **排除**，见下 |

136096 的 `success_once`（5 seed 均值）：10k **0.172** / 30k **0.170** / 60k **0.170** / 100k **0.192**；
per-run 取 max 后的均值 **0.198**，相对 `final.pt` 的 0.192 只差 +0.6pp。也就是说验证 loss 的低点
（10k）**并不对应成功率的峰值**，`final.pt` 的数字不是 checkpoint 选择造成的低估。

于是剩下与论文 image 配置真正不同的地方，最可疑的是**视觉池化**：论文 ResNet-18 用 spatial
softmax —— 原文 *"Replace the global average pooling with a spatial softmax pooling to maintain
spatial information"* —— 而本仓库 baseline 是 `pool = "avg"`。UNet 只能通过 FiLM 条件化那个
全局特征向量，池化把空间信息压掉之后它没有别的通路能取回像素信息；Transformer 的
cross-attention 相对更耐受。这个实验就是把 `vision.pool` 单独换掉。

## 2. 实验定义

- 新增 spec：`dp-manip/configs/experiments/vision_pool_n100.toml`
- 唯一自变量：`vision.pool = avg | spatial_softmax`
- `[fixed] data.num_demos = 100`；seed 1–3（与已有 avg run 配对比较）
- `vision.num_keypoints` 保持 baseline 的 32（控制量，不豁免）
- 其余全部来自 `baseline.toml`：100k steps / batch 64 / AdamW lr 1e-4 cosine + warmup 500 /
  EMA 0.9999 / DDPM 100 步 / obs2 · act8 · pred16 / 单相机 128×128

run 目录名由 `default_run_name` 自动生成：

| arm | run |
| --- | --- |
| avg | `pickcube_rgb_unet_n100_s1..s3`（**复用** data-size 轨道已有 run，`final.pt` 存在即 skip） |
| spatial_softmax | `pickcube_rgb_unet_ss32_n100_s1..s3`（新增，3 次训练） |

复用是安全的：`completion_state` 只比 resolved config（`same_run`），不比实验标签，所以 avg 臂在
`vision_pool_n100` 和 `data_size` 两个 grid 里指向同一个 run 目录。

## 3. 操作记录

| # | 步骤 | 命令 / job | 时间 | 结果 |
| --- | --- | --- | --- | --- |
| 1 | 写 spec | `dp-manip/configs/experiments/vision_pool_n100.toml` | 10-05 ~15:35 | — |
| 2 | 本地干跑 | `scripts/sweep.py plan --experiment configs/experiments/vision_pool_n100.toml --task pickcube ...` | 10-05 15:35 | `6 runs, 3 completed (skipped), 3 pending, 0 conflict` |
| 3 | 本地自检 | CPU 前向 + 反向（spatial_softmax 路径） | 10-05 15:36 | loss / 梯度有限；policy 80.92M（avg 80.96M） |
| 4 | **提交训练** | `sbatch ... EXPERIMENT=configs/experiments/vision_pool_n100.toml ... slurm/train_dual_gpu.sbatch` → **job 136100** | 10-05 **15:36:50** 起跑，节点 `gpu-4080-414` | 跳过 3 个 avg run；worker0/1 = ss32 s1 / s2 |
| 5 | 前置诊断 | **job 136096**（N=100 × s1–s5 × 10k/30k/60k/100k） | 10-05 15:00–15:27 | 见 §1；结果在各 run `eval/test_step_*.json` |
| 6 | 并发限制探针 | **job 136101**（2 分钟空作业，已 scancel） | 10-05 ~15:58 | `PENDING / AssocGrpGRES` → 训练占满 2 卡时提不了第二个 GPU 作业 |

提交模板（先 `plan` 确认复用/pending 数量，再 `sbatch`）：

```bash
cd ~/7606C/dp-manip
DATA_ROOT=$HOME/7606C/maniskill-demogen/data/dataset
RUN_ROOT=$HOME/dp-runs-pickcube

.venv/bin/python scripts/sweep.py plan \
  --experiment configs/experiments/vision_pool_n100.toml \
  --task pickcube --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"

sbatch --partition=batch --open-mode=append \
  --export=ALL,TASK=pickcube,EXPERIMENT=configs/experiments/vision_pool_n100.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch
```

> 必带 `--partition=batch`：集群默认分区 `debug` 上限 18h，而这个 sbatch 申请 48h，
> 不带分区会被 `PartitionTimeLimit` 永久挂起（历史上作业 135936 就是这么被 scancel 的）。

## 4. 进度与预计

实测 **15.3 步/秒**（s1：14900 步 / 975s）→ 单 run（100k 步）**≈ 109 min**。3 个 seed 走两轮
（s1+s2 并行，然后 s3）：

| 事件 | 预计时间 |
| --- | --- |
| s1、s2 的 `final.pt` | ~**17:26** |
| s3 的 `final.pt`（整个 job 结束） | ~**19:15** |

每个 run 落盘：`step_010000.pt` / `step_030000.pt` / `step_060000.pt` / `final.pt`
（`resume.pt` 是断点续训用的原始权重，不能用于评测）。

## 5. 评估计划（待做）

用 `~/7606C/eval_ss32_checkpoint_scan.sbatch`：3 seed × 4 checkpoint = **12 次评估**，
2 卡并行约 25 min。

```bash
sbatch --partition=batch --open-mode=append ~/7606C/eval_ss32_checkpoint_scan.sbatch
```

结果写到各 run 的 `eval/test_{step_010000,step_030000,step_060000,final}.json`（不覆盖 avg 臂的文件）。

对照表（avg 臂 s1–s3 均值，`success_once`，已测得）：

| arm | 10k | 30k | 60k | 100k |
| --- | --- | --- | --- | --- |
| avg | 0.167 | 0.153 | 0.163 | 0.180 |
| ss32 | ? | ? | ? | ? |

## 6. 机时

| 项 | 估算 |
| --- | --- |
| 训练 3 run（UNet 100k 步） | ≈ 3 × 103 = **310 GPU-min** |
| 前置诊断 job 136096 | ≈ **60 GPU-min** |
| 评估 12 次（待做） | ≈ 12 × 3.5 = **42 GPU-min** |
| 合计 | ≈ **410 GPU-min** |

## 7. 后续

1. 出数字后并入 `report_pickcube_yhr.md`（新增一节），或按结果决定是否单独成篇。
2. 若 spatial_softmax 明显拉起来 → 补 seed 4、5；并考虑把 spatial softmax 接到 `layer3`
   （128×128 输入到 layer4 只剩 4×4，`configs/README.md` 已把这一条标为待定）。
3. 若没有变化 → 下一批候选是相机数（论文 2 视角 vs 本仓库 1 视角）和增广方式
   （论文 random crop 76/84 vs 本仓库 `random_shift=4`）。
4. `dp-manip/` 是 subtree：这个 spec 的理想路径是先推到 `hollinsStuart/dp-manip` 上游再
   `git subtree pull`；直接在本仓库提交，下次 subtree pull 时可能冲突。
