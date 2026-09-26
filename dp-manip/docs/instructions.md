# 操作说明：从专家轨迹到测试成功率

> **历史文档（旧本地/state 流程）**：当前集群 RGB 训练入口见 `../README.md`。以下命令仅用于
> 追溯早期 pilot，不要用于新六任务实验。

以 PickCube 为例，写清这个项目目前**怎么生成数据、怎么把数据传到训练机、模型和训练代码在哪、结果存在哪、测试集是什么、怎么跑测试**。命令都是实际跑过的（9.23），换任务时改 `--env-id`、文件名和配置即可。

- 三台机器的分工和安全约束：[README.md](../README.md)、[AGENT.md](../AGENT.md)
- 每次运行的记录与结果：`docs/mmdd-hhmm.md`，例如 [0923-2155.md](./0923-2155.md)

---

## 零、整体流程

```text
 ubuntu（专家数据机，CPU 仿真 + MPlib）         Mac（权威仓库，中转）             wsl（Windows 上的 WSL2，RTX 4090）
 ─────────────────────────────────────         ────────────────────             ──────────────────────────────────
 1. run_cpu.py 运动规划 → 原始轨迹（只有动作）
 2. replay_trajectory → state 轨迹（观测+动作）
 3. 校验脚本
                         ── rsync ──▶  4. demos-batch/ → data/pickcube/ ── rsync ──▶  data/pickcube/
                                          更新 manifests/ 并提交
                                                                                      5. train_dp.py 训练 + 验证
                                                                                      6. eval_dp.py 测试集评估
                                        7. sync.sh pull-results  ◀── rsync ──  results/ checkpoints/ logs/
                                           写运行记录 docs/mmdd-hhmm.md
```

所有同步都从 Mac 发起（`scripts/sync.sh`），远端不会主动连 Mac。下文命令标明在哪台机器上执行；从 Mac 执行时用 `ssh ubuntu '...'` 或 `ssh wsl '...'` 包起来。

---

## 一、生成专家轨迹（ubuntu）

**命令**（在 ubuntu 的 `~/Coding/dp-manip` 下）：

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  .venv/bin/python run_cpu.py --env-id PickCube-v1 --sim-backend physx_cpu \
    --only-count-success -n 100 \
    --traj-name pickcube_batch100 --record-dir demos-batch
```

| 项目 | 说明 |
| --- | --- |
| 脚本 | `run_cpu.py`：就是 ManiSkill 自带的 `mani_skill/examples/motionplanning/panda/run.py`，只多了 `render_backend="cpu"`。它用 ManiSkill 官方写好的运动规划解（`solvePickCube` 等，基于 MPlib）控制 Panda 机械臂完成任务 |
| 控制模式 | 固定为 `pd_joint_pos`：每步动作是 8 维（7 个关节目标角度 + 夹爪开合） |
| 种子 | 从 0 开始，每条轨迹一个种子，依次递增。**种子决定这条轨迹的初始状态**（方块位置、朝向、目标点） |
| `--only-count-success` | 规划失败的种子直接跳过，直到攒够 N 条成功轨迹。100 条那次种子 51 失败，所以实际用的是 0–100 去掉 51 |
| **必须单进程** | 不要加 `--num-procs`。多进程时每个进程从固定起点取种子，跳过失败种子会越界进入下一个进程的种子段，产生重复示范 |
| 输出 | `demos-batch/PickCube-v1/motionplanning/pickcube_batch100.{h5,json}` |
| 耗时 | 100 条约 27 秒 |

**这一步的输出还不能直接训练**：H5 里只有动作，观测（`obs`）是空的。

ubuntu 环境的前提（已配好，不要改）：`mani-skill==3.0.1`、`mplib==0.2.1`，外加补丁 `patches/mani_skill_mplib_0_2_1.patch`；必须设置 `VK_ICD_FILENAMES`、使用 CPU 物理后端。详见 README「各设备环境」。

## 二、重放出观测（ubuntu）

在同一个仿真里，把每条轨迹按原动作重新执行一遍，并把每一步的观测记录下来：

```bash
.venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch100.h5 \
  --obs-mode state --save-traj --use-env-states \
  --max-retry 0 --num-envs 1 --verbose
```

| 项目 | 说明 |
| --- | --- |
| `--obs-mode state` | 观测是 42 维扁平状态向量。PickCube 由这些字段拼接而成（ubuntu 3.0.1 用 `obs_mode="state_dict"` 核对过）：关节角 `qpos` 9、关节速度 `qvel` 9、是否抓住 `is_grasped` 1、末端位姿 `tcp_pose` 7、目标位置 `goal_pos` 3、方块位姿 `obj_pose` 7、末端到方块 `tcp_to_obj_pos` 3、方块到目标 `obj_to_goal_pos` 3 |
| `--use-env-states` | 每一步都把仿真强制设成原始记录的状态，保证观测与原轨迹完全对应 |
| 不写 `--target-control-mode` | 保持 `pd_joint_pos`，动作逐值不变 |
| 输出 | 同目录下 `pickcube_batch100.state.pd_joint_pos.physx_cpu.{h5,json}`：每条轨迹 `obs (T+1, 42)`、`actions (T, 8)` |
| 耗时 | 100 条约 32 秒 |

**训练用的就是这个 `.state...` 文件。**

## 三、检查数据（ubuntu）

```bash
D=demos-batch/PickCube-v1/motionplanning
.venv/bin/python scripts/validate_replay.py $D/pickcube_batch100.h5 $D/pickcube_batch100.state.pd_joint_pos.physx_cpu.h5
.venv/bin/python scripts/inspect_dataset.py  $D/pickcube_batch100.state.pd_joint_pos.physx_cpu.h5
```

检查全部成功、重放没有改动作、没有 NaN/Inf、episode 边界能由 H5 的 `traj_<id>` 组加 JSON 的 `episode_id` 恢复。9.23 还额外检查了种子不重复、都小于 5000、长度分布，以及前 10 条与旧的 10 条批次逐值相同（结果见 STATUS.md「100 条批次」）。

## 四、把数据传到 wsl（从 Mac 执行）

wsl 是 Windows 电脑上的 WSL2，项目在 `~/projects/dp-manip`。ubuntu 和 wsl 之间不直接传，统一经过 Mac：

```bash
# 1. ubuntu → Mac（与 ubuntu 目录结构相同）
rsync -a ubuntu:Coding/dp-manip/demos-batch ./

# 2. 放进 Mac 的训练目录布局（-n：不覆盖已有文件）
S=demos-batch/PickCube-v1/motionplanning
cp -n $S/pickcube_batch100.h5 $S/pickcube_batch100.json data/pickcube/
cp -n $S/pickcube_batch100.state.pd_joint_pos.physx_cpu.{h5,json} data/pickcube/state/

# 3. Mac data/ → wsl data/，并按已有清单校验
scripts/sync.sh data

# 4. 在两台远端重新生成 SHA-256 清单，检查只有新增，再提交
scripts/sync.sh manifest
git diff manifests/                  # 不应有 - 开头的行
shasum -a 256 -c --quiet manifests/ubuntu-demos.sha256 manifests/wsl-data.sha256
git add manifests && git commit -m "..."
```

wsl 上数据的位置：`data/pickcube/`（原始）、`data/pickcube/state/`（训练用）。数据不进 git，一致性靠 `manifests/*.sha256`。

代码的同步是另一条路：在 Mac 提交后执行 `scripts/sync.sh push`（推到 ubuntu 和 wsl）。

---

## 五、代码在哪

| 文件 | 作用 |
| --- | --- |
| `dp_manip/conditional_unet1d.py` | **网络结构**：1D 条件 UNet（约 439 万参数）。输入是加了噪声的动作序列 + 扩散步数，条件是观测，输出预测的噪声。原样取自 ManiSkill 官方 DP 基线 |
| `dp_manip/policy.py` | **Diffusion Policy 模型**（`DiffusionPolicy`）：包装 UNet 和 DDPM 调度器（100 步）；`compute_loss` 是训练损失（预测噪声的 MSE），`get_action` 是推理（从纯噪声去噪 100 步得到动作序列，取其中 8 步执行）；另有保存 / 读取 checkpoint 的函数 |
| `dp_manip/data.py` | **数据**：按 JSON `episode_id` 读 H5；把动作按维度线性缩放到 [-1, 1]（`ActionNormalizer`）；把轨迹切成训练窗口（2 步观测历史 → 预测 16 步动作） |
| `dp_manip/envs.py` | **评估环境**：用 ManiSkill 建 10 个并行的 CPU 仿真环境，控制模式、回合长度与训练数据一致；默认不渲染 |
| `dp_manip/evaluate.py` | **闭环评估**：按给定的种子列表逐个 reset 环境，让策略控制机器人跑满一个回合，记录每个种子是否成功 |
| `dp_manip/config.py` | 配置格式（TOML 各段的字段和默认值），以及验证 / 测试种子的计算 |
| `scripts/train_dp.py` | **训练入口**：训练循环 + 训练中定期验证 + 保存 checkpoint 和日志 |
| `scripts/eval_dp.py` | **测试入口**：读 checkpoint，在测试（或验证 / 训练）种子上评估 |
| `scripts/check_dp_offline.py` | 不需要仿真器的自检：数据窗口、归一化、单步训练、采样、checkpoint 读写 |
| `scripts/replay_check.py` | 在评估环境里开环回放示范动作，确认评估环境与数据一致（应该 100% 成功） |
| `configs/*.toml` | 每个实验的完整设置；训练和评估共用，训练时会被存进 checkpoint |
| `dp_manip/README.md` | 代码出处（ManiSkill 官方 DP 基线 `@62ff3a5`）和我们做的每一处改动 |

没有直接用 Stanford 的 `train.py`，原因见 README「项目流程」。

## 六、训练（wsl）

**命令**（在 wsl 的 `~/projects/dp-manip` 下）：

```bash
.venv/bin/python scripts/train_dp.py \
  --config configs/pickcube_state_jointpos_100.toml \
  --exp pickcube_jointpos_100demo_30k \
  --set data.num_demos=100 --set train.seed=1
```

| 参数 | 说明 |
| --- | --- |
| `--config` | 实验配置。`pickcube_state_jointpos_100.toml` 指向 100 条批次的 state 文件 |
| `--exp` | 实验名，决定输出目录；已存在会拒绝运行（除非加 `--force`） |
| `--set 段.键=值` | 临时覆盖配置，可以写多个。常用：`data.num_demos=N`（取前 N 条，做数据量消融）、`train.seed=S`（训练随机种子） |
| `--no-eval` | 训练中不做验证（不需要 ManiSkill，适合只测训练） |

**训练过程**（`scripts/train_dp.py`）：

1. 读配置、读前 N 条示范；检查数据的任务名、控制模式、观测模式与配置一致，检查验证 / 测试种子不与示范种子重叠。
2. 用这 N 条示范的动作算归一化范围，切出所有训练窗口，整体放进显存。
3. 建 `DiffusionPolicy`；优化器 AdamW（lr 1e-4，前 500 步预热，之后余弦下降），并维护一份 EMA（滑动平均）权重。
4. 循环 30000 步：每步随机抽 1024 个窗口 → 给动作加随机程度的噪声 → UNet 预测噪声 → MSE 损失 → 反向传播。
5. 每 5000 步用 EMA 权重在 **50 个验证种子**上跑一次闭环评估，记录成功率（只用来观察训练是否收敛，**不用来挑模型**）。
6. 结束时保存 `final.pt`，写 `summary.json`。

超参数（horizon、UNet 大小、batch、学习率、扩散步数）与 ManiSkill 官方 PickCube 基线相同，写在配置文件的 `[policy]`、`[train]` 段。

**耗时**：一组约 12.5 分钟（纯训练约 9 分钟 + 6 次验证约 3 分钟），峰值显存约 680 MiB。训练时间和示范条数无关。

长时间运行时用 `nohup` 放后台，日志写到 `logs/`：

```bash
nohup .venv/bin/python scripts/train_dp.py --config ... --exp ... > logs/<exp>.log 2>&1 < /dev/null &
```

多组实验用一个脚本串行执行，例子见 wsl 上的 `~/dp-manip-runs/*.sh`（仓库外）。

## 七、结果怎么保存

每个实验一个名字（`--exp`），输出分两个目录（都不进 git）：

```text
results/<exp>/
  config.json       本次解析后的完整配置 + 示范种子 + GPU / torch 版本 + 开始时间
  summary.json      训练耗时、ms/步、验证耗时、峰值显存、最终损失、验证曲线
  metrics.jsonl     每 1000 步一行：loss、学习率、已用时间
  val_<步数>.json   每次验证：汇总 + 每个验证种子的结果
  test_final.json   测试集结果（eval_dp.py 生成）：汇总 + 每个测试种子是否成功
  train_best.json 等  其他 split / checkpoint 的评估结果
checkpoints/<exp>/
  final.pt          训练结束时的模型（主结果用这个）
  best.pt           验证成功率最高的那次（只作参考，不用于报告）
logs/<exp>.log      终端输出
```

checkpoint 里除了网络权重（普通权重和 EMA 权重），还存了**动作归一化范围和完整配置**，所以评估时不需要再指定配置，也不会用错控制模式或 horizon。每个约 34 MB。

## 八、测试集是什么、怎么跑

### 测试集从哪里来

**测试集不是一份数据文件，而是一组固定的随机种子。** ManiSkill 在 `env.reset(seed=s)` 时，用种子 `s` 随机生成这一回合的初始状态：方块放在哪、朝哪个方向、目标点在哪。同一个种子永远生成同一个初始状态，所以「测试集」就是一组固定的初始状态，让训练好的策略从这些状态出发，自己控制机器人去完成任务。

| 种子范围 | 用途 | 在哪定义 |
| --- | --- | --- |
| 0–100（去掉 51） | 专家示范用的初始状态（训练数据） | 采集时由 `run_cpu.py` 从 0 递增 |
| 5000–5049 | **验证**：训练中每 5000 步评估一次，只看是否收敛 | 配置 `[eval] val_seed_start / val_episodes` |
| **10000–10099** | **测试**：报告用的成功率只来自这里 | 配置 `[eval] test_seed_start / test_episodes` |

三段互不重叠，训练脚本会检查，重叠就拒绝运行。所以测试时的 100 个初始状态，模型在训练中都没见过（held-out）。

**成功判据**（ManiSkill 的 PickCube 定义）：方块离目标点足够近，并且机械臂静止。每个回合跑满 100 步：
- `success_once`：100 步内**任何时刻**成功过；
- `success_at_end`：最后一步仍然成功。

### 怎么跑

```bash
.venv/bin/python scripts/eval_dp.py checkpoints/pickcube_jointpos_100demo_30k/final.pt              # 测试集
.venv/bin/python scripts/eval_dp.py checkpoints/pickcube_jointpos_100demo_30k/final.pt --split train # 诊断
```

| 参数 | 说明 |
| --- | --- |
| `--split test` | 默认；种子 10000–10099，结果写 `results/<exp>/test_final.json` |
| `--split val` | 验证种子 |
| `--split train` | 示范用过的种子，**诊断用**：链路正确时模型应能复现示范；如果这里也失败，说明评估环境和数据不一致，而不是泛化差 |
| `--episodes N` | 只跑前 N 个种子 |
| `--seed` | 扩散采样噪声的随机种子（默认 0） |
| `--save-states` | 保存每个回合的种子和逐步仿真状态，写到 `results/<exp>/states_<split>_<ckpt>/env<i>.{h5,json}`，每 10 个回合约 140 KB；不需要渲染，也不影响结果（9.23 实测 20 个回合逐回合一致）。用于事后高画质渲染，见下 |
| `--video` | 录第一个环境的视频，每个回合一个 mp4，写到 `results/<exp>/videos_<split>_<ckpt>/`（9.23 已在 wsl 验证，见下） |

100 个测试回合约 67 秒。

**wsl 上的视频渲染**：WSL2 没有 NVIDIA 的 Vulkan 驱动（没有 `nvidia_icd.json`），不做处理时 SAPIEN 建渲染器会报 `ErrorIncompatibleDriver`。`dp_manip/envs.py` 的 `ensure_render_icd()` 会在需要录像时，自动把 `VK_ICD_FILENAMES` 设为 Mesa 的 lavapipe（`/usr/share/vulkan/icd.d/lvp_icd.json`，CPU 软件渲染）；如果手动设了 `VK_ICD_FILENAMES`，或者机器上有 NVIDIA ICD（比如 ubuntu），就不会改动。只有第一个环境渲染，每帧 512×512，约 0.2 秒。录像会拖慢这个环境，所以带 `--video` 的运行只用来看视频：它的 `mean_inference_ms` 和耗时不要写进报告，成功率仍以不带 `--video` 的测试结果为准。想录几段回合，可以用 `--episodes`：

```bash
.venv/bin/python scripts/eval_dp.py checkpoints/<exp>/final.pt --split test --episodes 20 --video
```

注意：这会**覆盖** `results/<exp>/test_final.json`。因为它只是部分回合，跑完后要不带 `--video` 重跑完整测试；也可以改用 `--split val`，避免碰到测试结果文件。

**报告/展示用的高画质视频**：正式测试时加上 `--save-states`，然后用 `scripts/render_episodes.py` 离线回放渲染。回放时先按种子 reset，再逐步用 `set_state_dict` 恢复存下的状态，不跑策略也不跑物理，所以画面就是被评估的那个回合；分辨率、机位、shader 都可以事后另选。9.23 验证过：回放帧和评估时的实录逐帧对齐；用存下的状态重算每一步的成功判定，和录制时完全一致。

```bash
.venv/bin/python scripts/eval_dp.py checkpoints/<exp>/final.pt --save-states          # 完整测试 + 存状态
.venv/bin/python scripts/render_episodes.py results/<exp>/states_test_final --seeds 10000 10010   # 默认 1080p 光栅化
.venv/bin/python scripts/render_episodes.py results/<exp>/states_test_final --seeds 10010 \
    --shader rt --eye 0.6 0.7 0.6 --target 0 0 0.35                                      # 光线追踪，自定机位
```

专家示范也能这样渲染：原始轨迹 h5 本身就存了逐步状态和种子，直接把文件传进去即可，输出在 `results/demos/<文件名>/`（不写进 `data/`，以免影响数据清单）：

```bash
.venv/bin/python scripts/render_episodes.py data/pickcube/pickcube_batch100.h5 --seeds 0 1 2
```

输出为 `results/<exp>/renders_<shader>_<W>x<H>/seed<seed>.mp4`（x264，`--crf` 默认 16）。wsl 上用 lavapipe 渲染，1080p 的耗时如下：`default` 每帧约 0.1 秒，一个回合约 7 秒；`rt` 每帧约 25 秒（8 线程），一个回合约 40 分钟，只适合挑几段、在训练空闲时跑；`rt-fast` / `rt-med` 依赖 OptiX 降噪器，lavapipe 没有，画面全是噪点。渲染和训练抢 CPU，**训练进行中不要跑**。

### 报告口径（9.23 决定）

主结果一律是 **`final.pt` 在测试种子上的 `success_once`**，同时列出 `success_at_end`；不按验证成功率挑 checkpoint。每个设置跑 3 个训练种子（`train.seed=1,2,3`），报告平均值和范围。

## 九、取回结果并记录（Mac）

```bash
scripts/sync.sh pull-results     # wsl 的 results/ checkpoints/ logs/ → Mac（不删除 Mac 上已有的）
```

然后写本次运行记录 `docs/mmdd-hhmm.md`（跑了什么、命令、结果、耗时），更新 STATUS.md / TODO.md，提交并 `scripts/sync.sh push`。

---

## 附：一次完整流程的命令清单（PickCube，100 条）

```bash
# ubuntu：生成 + 重放 + 检查
ssh ubuntu 'cd ~/Coding/dp-manip && VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json .venv/bin/python run_cpu.py --env-id PickCube-v1 --sim-backend physx_cpu --only-count-success -n 100 --traj-name pickcube_batch100 --record-dir demos-batch'
ssh ubuntu 'cd ~/Coding/dp-manip && .venv/bin/python -m mani_skill.trajectory.replay_trajectory --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch100.h5 --obs-mode state --save-traj --use-env-states --max-retry 0 --num-envs 1 --verbose'
ssh ubuntu 'cd ~/Coding/dp-manip && D=demos-batch/PickCube-v1/motionplanning && .venv/bin/python scripts/validate_replay.py $D/pickcube_batch100.h5 $D/pickcube_batch100.state.pd_joint_pos.physx_cpu.h5'

# Mac：中转到 wsl + 清单
rsync -a ubuntu:Coding/dp-manip/demos-batch ./
cp -n demos-batch/PickCube-v1/motionplanning/pickcube_batch100.{h5,json} data/pickcube/
cp -n demos-batch/PickCube-v1/motionplanning/pickcube_batch100.state.pd_joint_pos.physx_cpu.{h5,json} data/pickcube/state/
scripts/sync.sh data && scripts/sync.sh manifest     # 检查 git diff manifests/ 后提交

# wsl：自检 + 训练 + 测试
ssh wsl 'cd ~/projects/dp-manip && .venv/bin/python scripts/check_dp_offline.py --config configs/pickcube_state_jointpos_100.toml --device cuda'
ssh wsl 'cd ~/projects/dp-manip && .venv/bin/python scripts/train_dp.py --config configs/pickcube_state_jointpos_100.toml --exp pickcube_jointpos_100demo_30k --set data.num_demos=100'
ssh wsl 'cd ~/projects/dp-manip && .venv/bin/python scripts/eval_dp.py checkpoints/pickcube_jointpos_100demo_30k/final.pt'

# Mac：取回
scripts/sync.sh pull-results
```
