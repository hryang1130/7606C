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
| 7 | 闭环评估（第 1 次） | **job 136118**（12 次评估） | 10-05 19:09–19:34 | ss32 = 0.007 vs avg = 0.180；诊断见 §5 |
| 8 | 留证并删除无效 run | 12 份评估汇总存到 `report/exp_vision_pool_n100_deadinit_evals.json`，然后删除 `pickcube_rgb_unet_ss32_n100_s1..s3` | 10-05 ~19:55 | 释放 ~8 GB（可用 23G → 31G）；**必须删**，否则修好后 config 未变会被判 completed 而 skip |
| 9 | 修 keypoint head | `dp-manip/dp_manip/vision.py`（F1） | 10-05 ~20:00 | 见 §7.2；实测 logits std 7.3→0.8、熵 0.748→2.68、坐标跨输入 std 0→0.05、keypoint 梯度比 0.005→1.2 |
| 10 | 加回归测试 | `dp-manip/tests/test_vision_pool.py` 新增两条断言 | 10-05 ~20:05 | 阈值按上面的实测值留了余量 |
| 11 | **重新提交训练** | `sbatch ... EXPERIMENT=configs/experiments/vision_pool_n100.toml ...` → **job 136121** | 10-05 20:20 起跑，节点 `gpu-4080-413` | plan 复核 `6 runs, 3 completed (skipped), 3 pending, 0 conflict`；worker0/1 = ss32 s1 / s2 |
| 12 | 修复后闭环评估 | **job 136198**（12 次评估，默认分区，节点 `gpu-4080-414`） | 10-06 16:08–16:28 | ss32 = **0.127** vs avg = **0.180** @100k → **假设被否掉**，见 §5.4 |
| 13 | 录汇报视频（附带） | **job 136189**（`slurm/record_videos.sbatch`，N=100 UNet s1 `final.pt`） | 10-06 15:07–15:09 | 6 段 1080p 成功/失败视频 + `videos.json`，`all 12 recorded outcomes match test_final.json` |

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

## 5. 评估：结果与诊断（2026-10-05 晚）

评估用 `~/7606C/eval_ss32_checkpoint_scan.sbatch`（3 seed × 4 checkpoint = 12 次，job **136118**，
19:09:10 → 19:34:18，`0 failed`）。结果写在各 run 的
`eval/test_{step_010000,step_030000,step_060000,final}.json`，未覆盖 avg 臂的文件。

### 5.1 结果：ss32 崩了

`success_once`（s1–s3 均值）：

| arm | 10k | 30k | 60k | **100k** | `success_at_end` @100k |
| --- | --- | --- | --- | --- | --- |
| avg | **0.167** | **0.153** | **0.163** | **0.180** | 0.127 |
| ss32 | 0.000 | 0.003 | 0.003 | **0.007** | 0.003 |

三个 seed 的 100k 分别是 0.010 / 0.000 / 0.010 —— 不是"略差"，是**几乎完全不会做**。

### 5.2 诊断：spatial softmax head 在训练中饱和，相机分支变成常数

离线检查 12 次评估用的三个 `final.pt`（24 个窗口，跨 50 条示范）＋对照 avg 臂：

| 指标 | avg | ss32 (s1/s2/s3) |
| --- | --- | --- |
| softmax 熵（nats，均匀 = 2.773） | — | **0.000 / 0.001 / 0.000** → one-hot |
| 关键点坐标跨输入 std | — | **0.00000**（完全不变） |
| argmax 位置"与第 0 个输入相同"的比例 | — | **1.00**（100% 固定） |
| 相机特征跨输入 std | 0.036 | **0.00000**（常数） |
| temperature | — | 0.989 / 0.995 / 0.990（从初值 1.0 基本没动） |
| logits std | — | ≈ 19.5 |

也就是说：**每张图都被映射成同一个相机特征**，encoder 的相机分支是死的，策略只剩 proprio + 时间步可用
（`Dobs = 128 + 29`，坏的正好是那 128 维）。PickCube 的物体位姿不在 proprio 里，所以闭环成功率 ≈ 0；
训练 loss 还能降到 1e-4 是因为示范里的动作很大程度上由机械臂自身状态决定。

**为什么会饱和**：soft-argmax 的坐标是 `softmax(logits/T) @ positions`。一旦 softmax 变成 one-hot，
它对 logits 的导数就归零 —— keypoints 卷积再也收不到梯度，关键点位置被永久冻结。实测
`grad|keypoints conv = 7.4e-4` vs `grad|head Linear = 1.5e-1`（小 200 倍），吻合。

**根因是初始化**：`dp_manip/vision.py` 的 `ResNet18Encoder._initialize` 对**所有** `nn.Conv2d` 都用
`nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")`，其中也包括
`SpatialSoftmax.keypoints = Conv2d(512 → 32, kernel 1)`。`fan_out = 32` → 权重 std ≈ 0.25，作用在 512 维
内积上 → 初始 logits std ≈ **7.3**，而 softmax 只在 16 个格子上、温度 ≈ 1 —— 起始就已经很尖
（熵 0.748 / 2.773），训练中 logits 继续涨到 std ≈ 19.5 就彻底 one-hot，然后梯度消失、关键点冻结。

这个 1×1 keypoint 卷积需要它自己的初始化（零初始化，或至少按 `1/sqrt(512)` 缩放），不能沿用 ResNet 的卷积初始化；
另外温度参数需要下界 / 更好的参数化，防止 logits 无界增长。

### 5.3 结论：这次运行不构成对 pool 假设的检验

本次对照实际比的是「**坏掉的相机分支** vs avg」，不是「spatial softmax vs avg」。因此
**原假设（encoder 池化是 UNet 的瓶颈）仍然未被检验**，不能据此说"spatial softmax 没用"。

而且这个 bug 是**仓库级的**：`configs/experiments/vision_pool.toml`（peginsertionside, N=200,
avg vs ss32）走的是同一条 `ResNet18Encoder` 初始化路径，如果那条线已经跑过，它的 ss32 臂同样无效。

### 5.4 修复后重跑：假设被否掉（2026-10-06）

按 §7.2 的 F1 修好 keypoint head 后重训（job **136121**），并用同一套流程评估（job **136198**，
12 次，16:08–16:28，`0 failed`）。`success_once`（s1–s3 均值）：

| arm | 10k | 30k | 60k | **100k** | `success_at_end` @100k |
| --- | --- | --- | --- | --- | --- |
| avg | **0.167** | **0.153** | **0.163** | **0.180** | 0.127 |
| ss32（修复后） | 0.097 | 0.117 | 0.120 | **0.127** | 0.080 |
| ss32（坏掉那次） | 0.000 | 0.003 | 0.003 | 0.007 | 0.003 |

逐 seed 的 100k：avg 0.170 / 0.180 / 0.190，ss32 0.110 / 0.140 / 0.130 —— **三个 seed 全部低于
avg 的任意一个**，差值在每一档都是 −0.04 ~ −0.06，方向一致。

head 在训练结束时是**健康的**（不是再次塌陷，所以这个数字可信）：

| 指标（ss32 s1 final.pt） | 坏掉那次 | 修复后 | avg 对照 |
| --- | --- | --- | --- |
| softmax 熵（上限 2.773） | 0.000 | **0.592** | — |
| 关键点坐标跨输入 std | 0.00000 | **0.0985** | — |
| 相机特征跨输入 std | 0.00000 | **0.0259** | 0.0390 |
| temperature | 0.99 | 1.0256 | — |

**结论：假设被否掉。** 在这套配置下，把 encoder 从全局平均池化换成 robomimic 式 spatial softmax，
不是"没效果"，而是**每个 checkpoint 都低 4–6 个百分点**（0.180 → 0.127）。所以 UNet 在 N=100 上的
弱**不是 encoder 池化造成的**，至少不是朝这个方向。

三条必须一起说的边界：

1. 这里只测了**从 `layer4` 读**（128×128 输入 → 只有 4×4 = 16 个格子）。`layer3`（8×8）没测，
   仍然是 §7.3 的 B1。
2. avg 的 head 是 `Linear(512→128)`（65k 参数），ss 的 head 是 `Conv(512→32)` + `Linear(64→128)`
   （24k 参数）。所以这个对照里**池化方式和 head 容量是混在一起的**，不能断言"纯池化效应"。
3. 训练后的 softmax 熵从初始化时的 0.748 漂到 0.592（仍未塌陷，但方向一致）——说明**logits 还是
   在慢慢变大**，温度下界只挡住了温度那一条路。以后要重跑 ss 系列，建议对 logits 加约束
   （或改读 `layer3`），否则更长的训练有可能再次饱和。

## 6. 机时

| 项 | 估算 |
| --- | --- |
| 训练 3 run（UNet 100k 步） | ≈ 3 × 103 = **310 GPU-min** |
| 前置诊断 job 136096 | ≈ **60 GPU-min** |
| 评估 12 次（待做） | ≈ 12 × 3.5 = **42 GPU-min** |
| 合计 | ≈ **410 GPU-min** |

## 7. 若 spatial_softmax 收益不高，下一步做什么

### 7.1 决策规则（2026-10-06 更新：假设已被否掉）

§5.4 的修复后重跑给出结论：spatial softmax **每个 checkpoint 都比 avg 低 4–6 个百分点**
（100k：0.180 → 0.127），而且训练结束时 head 是健康的，所以这是有效对照，不是坏图。

**"encoder 池化（avg）是 UNet 的瓶颈"这条假设被否掉。** 后续按下面的顺序走：

1. **A1–A3 的零成本诊断**（§7.2）：先弄清失败到底是"看不见"还是"抓不住"，别再盲目换 encoder。
2. **εθ 侧**（§7.4）：C1 容量/正则（UNet 是唯一把训练集背到 5e-4 的主干）、C2 条件化通路、
   C3 对齐官方超参。
3. **数据侧**（§7.5）：D1 第二视角、D2 加数据。
4. B1（`layer3`）/B2（容量档）**优先级下调**：只有在 A1–A3 或 C 系列指向"条件信号不够"时才回头做；
   而且做之前先按 §5.4 第 3 条给 logits 加约束，否则更长的训练可能再次饱和。

### 7.2 先修 bug（P0，必须先做），以及零成本诊断

**F1. 修 keypoint head 的初始化与温度（已完成，job 136121 用的就是这个修复）**

- 病因：`ResNet18Encoder._initialize` 对所有 `Conv2d` 用 `kaiming_normal_(mode="fan_out")`，
  连 `SpatialSoftmax.keypoints`（512→32，1×1）也吃到了 → 权重 std ≈ 0.25（PyTorch 默认 ≈0.026，
  robomimic 就用默认）→ 初始 logits std ≈ 7.3 → 16 个格子 + 温度 1 → 训练中涨到 std ≈ 19.5、
  熵归零、坐标梯度恒为 0、关键点冻结（§5.2）。
- 已做的修改（`dp-manip/dp_manip/vision.py`）：
  1. 新增 `SpatialSoftmax.reset_parameters()`，用 PyTorch 默认的 `kaiming_uniform_(a=sqrt(5))`
     + 零 bias + 温度 1；
  2. `ResNet18Encoder.__init__` 在 `self.apply(self._initialize)` 之后，对 `pool != "avg"`
     重新调用该初始化，避免 ResNet 的规则误伤 keypoint 卷积；
  3. `forward` 里温度加下界 `clamp(min=0.1)`，堵住"温度→0 也同样会饱和"这条路。
- 实测（修复前 → 修复后，128×128、4 张随机图）：logits std 7.3 → 0.8；softmax 熵 0.748 → 2.68
  （上限 2.773）；关键点坐标跨输入 std 0.00000 → 0.05；`grad|keypoints / grad|head Linear`
  0.005 → 1.2。
- 回归测试：`dp-manip/tests/test_vision_pool.py` 新增 `test_keypoint_head_does_not_start_saturated`
  与 `test_keypoint_conv_gets_a_usable_gradient`（阈值按实测留了余量）。
- **作废的产物**：第 1 次 run 的 `pickcube_rgb_unet_ss32_n100_*` 已删除（汇总留在
  `report/exp_vision_pool_n100_deadinit_evals.json`）；`configs/experiments/vision_pool.toml`
  （peginsertionside）里任何 ss32 臂也是同样无效的，用之前要先确认是否已跑过。

下面是**不需要训练**的诊断：

| # | 做法 | 代价 | 想回答的问题 |
| --- | --- | --- | --- |
| A1 | 统计示范里每步的动作变化幅度 `‖a_{t+1} − a_t‖`，特别是夹爪那一维的取值分布 | 0 | 动作序列到底有多"尖锐" |
| A2 | 用 env 的 reward/info 记录失败回合卡在哪个子目标（接近 / 抓取 / 抬起），复用 `evaluate` 的 observer 钩子 | ~1 次 eval（4 GPU-min） | 失败是"看不见"还是"抓不住 / 握不稳" |
| A3 | dump 少量 `base_camera` 帧，看 128×128 单视角下方块与夹爪的相对位置是否可分辨、有无遮挡 | 0 | 观测本身是否够用 |

**A1 为什么值得先做。** 论文自己写过，CNN backbone

> *"it performs poorly when the desired action sequence changes quickly and sharply through time
> (such as velocity command action space), likely due to the inductive bias of temporal
> convolutions to prefer low-frequency signals"*

PickCube 的动作空间是 `pd_ee_delta_pos`，夹爪闭合就是一个近乎阶跃的事件。如果 A1 显示这个切换很尖锐，
那 UNet 输给 MLP / Transformer 就可能**不是感知问题，而是时序平滑偏置** —— 后面的资源就该转向 εθ 的
时序建模，而不是继续调 encoder。报告 §3.1 里 `success_once` 比 `success_at_end` 高 6.4/100
（"抓到过又掉了"）和"抓取保持不稳定"是一致的。

### 7.3 同一条轴：继续调 encoder

| # | 改动 | 理由 | 代价 |
| --- | --- | --- | --- |
| B1 | spatial softmax 改从 `layer3`（8×8）读，而不是 `layer4`（4×4） | 128×128 / stride 32 → layer4 只有 16 个位置；`configs/README.md` 已把这条标为待定 | 3 run ≈ 310 GPU-min |
| B2 | `vision.num_keypoints` 32→64、`vision.feature_dim` 128→256 | 容量档。config README 明确要求它必须是**独立实验**（`variable = "vision.num_keypoints"`），不能混进同一个矩阵 | 各 3 run |
| B3 | 增广对齐论文：random crop（76/84）替代或叠加 `random_shift=4`，可再加颜色抖动 | 论文的 crop 是它 image 配置的一部分，本仓库只有 DrQ 平移 | 3 run |
| B4 | 预训练视觉编码器（ImageNet-21k / R3M / CLIP），冻结或 10× 小学习率微调 | 小样本下常见的最大杠杆。论文 §4.4.5 在自己的设定里得出"从零训练更好"，但那是 200 条 + 3000 epoch | 中（拉权重 + 3 run） |

### 7.4 εθ 那一侧：动 UNet 自己

| # | 改动 | 理由 | 代价 |
| --- | --- | --- | --- |
| C1 | 容量 / 正则对齐：`unet_dims [64,128,256]`，或 weight decay 1e-6 → 1e-4/1e-2，或加 dropout | UNet 是唯一把训练集背到 `train_loss ≈ 5e-4` 的主干；报告 §7.1 已把"容量对齐档"标为不进主表的补充 | 3 run |
| C2 | 换条件化通路：obs 直接 concat 到 UNet 输入（官方 `obs_as_global_cond=False` 那条），或让 UNet 也吃 obs token | 直接检验本文的"FiLM 带宽"假设，而不只是换特征 | 小改代码 + 3 run |
| C3 | 对齐官方 image UNet 超参：`down_dims [512,1024,2048]`、`diffusion_step_embed_dim 128`（官方 workspace 值；本仓库是 `[256,512,1024]` 和 256） | 排掉"只是超参没对齐"这个可能 | 3 run |

### 7.5 数据 / 任务侧（贵，但可能是最大的）

| # | 做法 | 理由 | 代价 |
| --- | --- | --- | --- |
| D1 | 加第二个视角（wrist / eye-in-hand） | 论文 image 配置是 2 视角；本仓库只有 `base_camera` 一个。对 PegInsertionSide / PlugCharger 这类接触密集任务，单视角很可能是硬伤 | 用 Demogen 重新生成数据（含第二相机）+ 重训，最大 |
| D2 | 加数据 | val loss 在 N=400 仍是干净的 N^−0.668 幂律，UNet 明显还没吃够数据 | 先用外推算"把 0.19 追到 0.9 要多少条"；若外推到不可行就直接排除 |

### 7.6 优先级建议

1. **P0** F1（修 keypoint head）＋ A1–A3 诊断（≈ 0–5 GPU-min）：F1 是所有 ss 实验的前置条件，
   A1–A3 则把"感知 / 时序 / 观测不足"分开。
2. **P1** 修好后的 ss32 重跑、B1（layer3）、C1（容量与正则）—— 各 3 run，最有可能直接命中已经观测到的两个现象
   （encoder 条件瓶颈、UNet 过拟合）。
3. **P2** B3、C2、C3。
4. **P3** B2、B4。
5. **P4** D1、D2：要动数据，应该等前面几条给出方向再做。

### 7.7 与论文的协议差异（若要严格复现）

论文 image 任务训练 **3000 epochs**，表格数字是 **(max performance) / (最后 10 个 checkpoint 平均，
每 50 epoch 存一次)**；本仓库是 829 epochs（100k 步）+ 只报 `final.pt`。§1 已经用 job 136096 排除了
"best checkpoint 能救回来"，但如果要走"严格复现论文"的路线，epoch 数与 checkpoint 口径必须一起对齐。

### 7.8 上游同步

`dp-manip/` 是 subtree：这个 spec 的理想路径是先推到 `hollinsStuart/dp-manip` 上游再 `git subtree pull`；
直接在本仓库提交，下次 subtree pull 时可能冲突。
