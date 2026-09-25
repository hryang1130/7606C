# dp-manip 当前状态

**最后更新：9.24 00:40（PickCube 数据量消融 × 3 个训练种子）** ｜ 未完成事项见 [TODO.md](./TODO.md) ｜ Day 1 原始记录见 [docs/history.md](./docs/history.md)

> 本文件由 Mac 原 `progress.md` 与 wsl 原 `STATUS.md` 合并而成（9.23）。

---

## 一、结论

- **ubuntu**：PickCube 的环境、专家生成、数据保存、回放链路全部通过；10/10 成功；已重放出 `obs_mode: state` 数据。
- **wsl**：**M0 训练节点验证完成**。Python、CUDA、RTX 4090 小规模训练能力已验证；state 数据在 wsl 上通过了哈希、观测与动作对齐、时间窗口、episode 边界检查。
- **wsl（9.23 B 节阶段 2–3）**：已装 ManiSkill 3.0.1 评估环境与 diffusers；PickCube 评估环境开环回放 10/10 成功，与 ubuntu 逐条一致。**「专家轨迹 → 数据集 → DP 训练 → 闭环评估」链路已在 PickCube 上跑通**：10 条示范训练 30k 步，训练种子 10/10，测试种子只有 2–3%（过拟合，不是 bug）。详见下文「DP 链路」。
- **PickCube 基线（9.23 21:55–22:41）**：100 条示范，测试成功率（success_once，100 回合）best.pt 0.67、final.pt 0.77；数据量消融 10 / 25 / 50 / 100 条 → final 0.02 / 0.14 / 0.47 / 0.77。完整记录与耗时见 [docs/0923-2155.md](./docs/0923-2155.md)。**报告口径（9.23 决定）：主结果固定用 final.pt，不按验证挑选**（50 回合验证选出的 best.pt 在 100 条时比 final.pt 低 10 个百分点）。按此口径，PickCube 基线 = **0.77**。
- **补训练种子 2、3（9.23 22:49 – 9.24 00:40）**：数据量消融 10 / 25 / 50 / 100 条，3 个训练种子平均 **0.02 / 0.13 / 0.56 / 0.67**（范围 0.02–0.03 / 0.09–0.15 / 0.42–0.79 / 0.53–0.77）。**PickCube 基线按 3 个种子平均 = 0.67**（上一条的 0.77 只是种子 1）。训练种子间的波动（50 条标准差 0.20）远大于评估噪声（约 0.05），50 条与 100 条的差距暂不能下结论。完整记录见 [docs/0923-2249.md](./docs/0923-2249.md)。
- **Mac**：9.23 起成为唯一权威 git 仓库，代码、配置、文档都从 ubuntu 和 wsl 汇总到这里（见 [PLAN.md](./PLAN.md)）。

尚未进行：其余五个任务的正式数据收集（9.24 各只试跑了 1 条，见「其余五个任务试跑」）、专家生成过程的录像（策略 rollout 录像已在 wsl 打通）、控制模式转换。

---

## 二、ubuntu：专家数据环境（9.22 重建并验证通过）

| 项目              | 当前状态                                                            |
| ----------------- | ------------------------------------------------------------------- |
| 系统              | Ubuntu Server 26.04                                                 |
| 硬件              | Intel Core i7-8700K、16 GB RAM、GTX 1080 Ti 11 GB                   |
| NVIDIA 驱动       | 580.178.04                                                          |
| 正式项目          | `~/Coding/dp-manip`                                                 |
| Python 环境       | `~/Coding/dp-manip/.venv`，Python 3.11.15                           |
| ManiSkill / MPlib | `mani-skill==3.0.1` / `mplib==0.2.1`                                |
| 依赖记录          | `environment/ubuntu-expert-freeze.txt`                              |
| MPlib 版本覆盖    | `mplib-probe-overrides.txt`                                         |
| 适配补丁          | `patches/mani_skill_mplib_0_2_1.patch`                              |
| CPU 专家入口      | `run_cpu.py`                                                        |
| PickCube 批量数据 | `demos-batch/PickCube-v1/motionplanning/pickcube_batch10.{h5,json}` |

**运行方式**：专家采集使用 `physx_cpu` 物理后端、`render_backend="cpu"`，并通过 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` 指定 Vulkan 驱动。

旧环境 `~/Coding/dp-manip-old-backup/.venv-mplib-probe` 同样报告 Python 3.11.15、ManiSkill 3.0.1、MPlib 0.2.1；原来的 `~/Coding/dp-manip/.venv-mplib-probe` 路径已不存在。

### 已完成事项（9.22）

- ✅ 确定六个 ManiSkill 任务：PickCube、PushCube、PullCube、StackCube、LiftPegUpright、PegInsertionSide。
- ✅ 在 Ubuntu 上跑通 ManiSkill 的 CPU 仿真、Vulkan 初始化和 MPlib 专家规划。
- ✅ 定位原始 `mplib==0.1.1` 在 Panda 规划器初始化时的崩溃问题，改用 `mplib==0.2.1` 并完成所需 API 适配。
- ✅ 将适配保存为独立补丁 `patches/mani_skill_mplib_0_2_1.patch`，避免依赖手工修改第三方包。
- ✅ 发现并排查 uv 硬链接导致的源码污染；在全新项目目录中绕过旧缓存、使用复制安装方式重新搭建环境。
- ✅ 将新项目迁移为正式路径 `~/Coding/dp-manip`，并在最终路径重新创建虚拟环境。
- ✅ 在正式路径验证：关键包版本正确、安装源码为原版、补丁应用成功、适配文件与已验证版本一致。
- ✅ 在正式路径重新生成 1 条 PickCube 专家轨迹，成功率 1/1，并成功回放（`demos-final-verify/`、`demos-rebuild-verify/`）。
- ✅ 采集并验收 10 条 PickCube 专家轨迹：10/10 规划成功、规划失败率 0、平均长度 72.6 步、最长 88 步。
- ✅ 验证这 10 条轨迹的 H5/JSON 内容、动作有效性及 10 个互不相同的随机种子；逐条回放全部成功。

### PickCube 批量验收结果（10 条）

| 指标           | 结果          |
| -------------- | ------------- |
| 规划成功       | 10 / 10       |
| 规划失败率     | 0             |
| 平均轨迹长度   | 72.6 步       |
| 最长轨迹       | 88 步         |
| 随机种子       | 10 个互不相同 |
| 动作有效性检查 | 通过          |
| 逐条回放       | 全部成功      |

---

## 三、wsl：训练节点（9.23 M0 验证完成）

| 项目        | 当前结果                                                                                     |
| ----------- | -------------------------------------------------------------------------------------------- |
| 系统        | WSL2 Ubuntu 24.04.5 LTS；项目盘上次检查约有 951 GB 可用空间                                   |
| 项目 Python | 3.11.15，虚拟环境 `~/projects/dp-manip/.venv`；系统 Python 为 3.12.3                           |
| 环境管理    | uv 0.12.18；`uv pip check` 检查 106 个包，均兼容（9.23 阶段 2 前为 31 个）；虚拟环境未安装 pip 模块 |
| 核心依赖    | PyTorch 2.14.0+cu130、NumPy 1.26.4、h5py 3.16.0                                               |
| 评估与 DP   | mani-skill 3.0.1、sapien 3.0.3、gymnasium 1.3.0（与 ubuntu 一致）、diffusers 0.40.0；mplib 0.1.1 随 mani-skill 装入但不使用 |
| 尚未安装    | torchvision（RGB 观测时再装）                                                                  |
| GPU         | NVIDIA GeForce RTX 4090，计算能力 8.9；驱动 591.86；`nvidia-smi` 显示 CUDA 13.1，PyTorch 运行时 CUDA 13.0 |

**已通过的 GPU 检查**：CUDA 可见性与 4×4 矩阵乘法（和为 120）；5 步 FP32 前向、反向与 AdamW 更新；5 步 fp16 AMP 更新。两种训练检查的损失和梯度均为有限值，参数确实改变，没有 CUDA 错误。FP32 与 AMP 峰值已分配显存分别为 16.44 MiB、16.40 MiB。这些是功能检查，不是性能测试。脚本：`scripts/verify_cuda.py`、`scripts/smoke_train_cuda.py [--amp]`。受限的进程沙箱可能挡住 GPU 访问，即使普通 WSL shell 中可用。

### DP 链路：B 节阶段 1–2（9.23）

**阶段 1（Mac，提交 `c8a8048`）**：`dp_manip/` 改编自 ManiSkill 官方 DP 基线（`haosulab/ManiSkill@62ff3a5`，Apache-2.0），出处与每处改动见 [dp_manip/README.md](./dp_manip/README.md)。修掉了基线用于我们数据时的四个问题：

1. DDPM 采样把动作裁剪到 [-1, 1]，但 `pd_joint_pos` 动作范围是 [-2.344, 2.808] → 按维 min-max 归一化，统计量存进 checkpoint；
2. 基线的 epoch 采样器 `drop_last=True`，726 个窗口 < batch 1024 时一批都没有 → 改为有放回采样；
3. 绝对控制模式的轨迹末尾 padding 未定义，会报错 → 重复最后一个动作；
4. 评估不固定种子，且用评估结果选 best → 分出验证种子 5000–5049（选 `best.pt`）与测试种子 10000–10099（只用于报告），脚本会拒绝与示范种子 0–9 重叠的配置。

Mac 上离线检查 6 项全过；Mac 的仿真环境是 `mani_skill_nightly 2026.8.2`（不是 3.0.1）、numpy 2.4.6，且 macOS 上 ManiSkill 的 `can_render()` 恒为真、`render_backend="none"` 也会建渲染器并因 Vulkan 失败，所以**Mac 不用于评估**。

**阶段 2（远端，锁文件提交 `a0f3fe9`，由 wsl 提交后取回）**：

| 步骤 | 结果 |
| --- | --- |
| ubuntu 开环回放（3.0.1，生成数据的同一环境，只读） | 初始观测 10/10 一致（最大差 6e-8），开环成功 10/10；后段观测最大偏差 1.2e-2，原因是 state 数据用 `--use-env-states` 逐步强制设状态，而这里是纯开环 |
| wsl 备份 | `~/dp-manip-backups/phase2-20260923/`：`freeze-before.txt`（31 包）、`uv.lock.before`、`head.txt`；`pyproject.toml.before` 实为推送后的新版，旧版见 git `e05ac89` |
| `uv lock` | 只有新增，没有旧包改版本或删除 |
| `uv sync --frozen` | 原 31 个包全部不变，新增 75 个；`.venv` 5.3G → 6.3G；`freeze-after.txt` 同目录 |
| 验收 | torch `2.14.0+cu130` 且 CUDA 可用；`verify_cuda.py` 通过；`check_dp_offline.py --device cuda` 6 项通过；`replay_check.py` 10/10，逐条数值与 ubuntu 完全相同 |

偏差：无。wsl 非交互 SSH 的 PATH 里没有 `uv`（在 `~/.local/bin`）和 `nvidia-smi`（在 `/usr/lib/wsl/lib`），远端命令需写全路径或补 PATH。

**阶段 3（wsl，代码 `9ae8ae4`）**：完整运行记录与耗时见 [docs/0923-2016.md](./docs/0923-2016.md)。

| 实验 | 设置 | 结果 |
| --- | --- | --- |
| `pickcube_smoke` | 300 步，验证 10 种子 ×2 次；`eval_dp.py` 测试 10 种子 | 训练、并行验证（10 个 CPU 子环境）、`best.pt`/`final.pt`、测试评估全部正常；成功率 0（预期） |
| `pickcube_jointpos_10demo_30k` | 10 条示范（726 步），官方超参：30k 步、batch 1024、lr 1e-4 cosine、EMA；每 5k 步 50 个验证种子 | 训练 687 s（约 23 ms/步），最终损失 0.0011；验证 success_once 依次 0.02 / 0.04 / 0.04 / 0.06 / 0.06 / 0.06，`best.pt` = 第 20000 步 |

测试集（种子 10000–10099，各 100 回合）：

| checkpoint | success_once | success_at_end | 成功的种子 |
| --- | --- | --- | --- |
| `best.pt`（20k） | **0.03** | 0.01 | 10011、10022、10077 |
| `final.pt`（30k） | **0.02** | 0.01 | 10022、10077 |

**诊断：在训练种子 0–9 上评估同一模型，`best.pt` 与 `final.pt` 都是 10/10。** 模型能在见过的初始条件下完整复现示范，说明观测对齐、动作归一化 / 反归一化、控制模式、动作分块执行都正确；测试集成功率低是 10 条示范只覆盖 10 种方块 / 目标位置导致的过拟合，不是链路 bug。此诊断已固化为 `scripts/eval_dp.py --split train`（Mac 提交，尚未在 wsl 复跑）。

其他记录：
- 推理：100 步 DDPM、10 个环境一批，约 426 ms / 次；一个 100 步回合要推理 13 次，100 个测试回合约 60 s。可作为「推理速度」研究（如 DDIM 减步）的基线。
- 资源：RTX 4090，torch 2.14.0+cu130，UNet 4.39M 参数；checkpoint 每个约 34 MB。
- 产出已用 `sync.sh pull-results` 取回 Mac：`results/`、`checkpoints/`、`logs/pickcube_jointpos_10demo_30k.log`。
- 偏差：临时诊断脚本从标准输入运行时，`forkserver` 子进程找不到主模块而失败，改用单环境完成；以后诊断一律用 `eval_dp.py --split`。

---

## 四、PickCube 数据资产

### 原始专家轨迹（`obs_mode: none`）

```text
ubuntu: ~/Coding/dp-manip/demos-batch/PickCube-v1/motionplanning/pickcube_batch10.{h5,json}
wsl:    data/pickcube/pickcube_batch10.{h5,json}
```

- ubuntu 修改时间 2026-09-22 18:51:22；H5 270,942 字节，JSON 2,887 字节。
- 任务 `PickCube-v1`，来源为运动规划，控制模式 `pd_joint_pos`，仿真与渲染后端均为 CPU；10 个 episode（seed 0–9）全部成功。
- H5 顶层为 `traj_0`–`traj_9`，每组一条轨迹。动作长度 **74、74、50、86、76、88、71、74、49、84**（共 726 步）。
- `actions` 为 `float32 (T, 8)`；`success`、`terminated`、`truncated` 为长度 T 的布尔数组。
- `env_states/actors/{cube,goal_site,table-workspace}` 为 `float32 (T+1, 13)`，`env_states/articulations/panda` 为 `float32 (T+1, 31)`。
- **`obs` 组为空**：原始文件只能用于检查链路，不能训练以观测为条件的策略。
- JSON 的 `max_episode_steps` 是 50，但多条轨迹超过 50 步，且部分逐步 `terminated`/`truncated` 在组结束前已为真。**切分 episode 只能用 H5 组 + JSON ID。**

### state 观测重放（M0）

在 ubuntu 上确认过 ManiSkill 3.0.1 `replay_trajectory` 的参数，省略 `--target-control-mode` 即保留 `pd_joint_pos`。精确命令：

```bash
cd ~/Coding/dp-manip
.venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch10.h5 \
  --obs-mode state --save-traj --use-env-states \
  --max-retry 0 --num-envs 1 --verbose
```

命令对未指定的 backend 参数发出过警告，但输出 JSON 确认 `sim_backend: cpu`，与原始一致；ubuntu 同时提示 GTX 1080 Ti 不受当前 PyTorch CUDA wheel 支持，因为用的是 CPU 仿真，不影响结果。未改动 ubuntu 环境，原始文件不变。

```text
ubuntu: ~/Coding/dp-manip/demos-batch/PickCube-v1/motionplanning/pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json}
wsl:    data/pickcube/state/pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json}
```

- H5 395,070 字节，JSON 2,819 字节。
- `traj_0`–`traj_9` 全部保留；seed、动作长度与原始一致，726 个动作逐值相同；10/10 成功。
- 每条 `obs` 是**单个扁平数组**，没有具名子字段，`float32 (T+1, 42)`；`actions` 为 `float32 (T, 8)`。所有数值 NaN/Inf 均为 0。
- 新 JSON 没有保留原始 JSON 的 `source_type/source_desc`，来源以本页记录为准。
- `scripts/validate_replay.py` 可复查原始与重放的一致性；`scripts/inspect_dataset.py` 默认检查 state 文件。
- `scripts/check_temporal_windows.py` 用真实 `obs` 验证历史 `obs[t-1:t+1]`、未来 `action[t:t+8]` 的首/中/末窗口：10 条轨迹共 646 个有效窗口，7 步短轨迹没有有效窗口，没有跨 episode 窗口。该检查只验证索引，不预设 padding 规则。

### 100 条批次（9.23 B 节下一轮·阶段 B，ubuntu）

采集命令同上文 10 条批次，只改 `-n 100 --traj-name pickcube_batch100`；**单进程**，因为 `run_cpu.py` 多进程时各进程从固定起点取种子，配合 `--only-count-success` 跳过失败种子会越界到下一个进程的种子段，产生重复示范。

| 项目 | 结果 |
| --- | --- |
| 采集 | 13:48:56 → 13:49:23（ubuntu 时间），**27 秒**（含启动；进度条 23 秒，约 4 条/秒）；100 条成功，试了 101 个种子，**种子 51 规划失败被跳过**（日志 2 行 `screw plan failed`） |
| state 重放 | **32 秒**，100/100 保存，命令同上文 |
| `validate_replay.py` | 100/100 成功，动作未变，数值完整，obs `(T+1, 42)` float32 |
| `inspect_dataset.py` | NaN/Inf 为 0；episode 边界可由 H5 组与 JSON id 恢复 |
| 种子 | 0–100 去掉 51，共 100 个，互不重复，全部 < 5000（不与验证 5000+ / 测试 10000+ 重叠） |
| 长度 | 最短 49 / 中位 79 / 平均 77.2 / 最长 99，共 7720 步；40–49: 1，50–59: 3，60–69: 12，70–79: 40，80–89: 38，90–99: 6 |
| 贴近回合上限 | 两条超过 95 步：traj_54（种子 55，96 步）、traj_80（种子 81，99 步）；评估回合上限是 100 步，比示范慢的策略可能来不及完成 |
| 与 `batch10` 对比 | 前 10 条（种子 0–9）的动作与观测**逐值相同** → 10 ⊂ 25 ⊂ 50 ⊂ 100 是真正的嵌套子集，`data.num_demos=10` 与第一次运行用的数据完全一致 |
| 嵌套子集规模 | 前 10 条 726 步；前 25 条 1852 步；前 50 条 3760 步；前 100 条 7720 步 |

文件（`demos-batch/PickCube-v1/motionplanning/`；日志在 ubuntu `logs/pickcube_batch100_{collect,replay}.log`）：

| 文件 | 大小 | SHA-256 |
| --- | --- | --- |
| `pickcube_batch100.h5` | 2,828,836 | `73ebcdec96490e6e609ca7757ab869ecd0c87963d12fd5aa64675087da0e9c3e` |
| `pickcube_batch100.json` | 23,325 | `166e2e58e5429544ba658433a177d5a7abda83faeeed852f3ce24a33af9caf83` |
| `pickcube_batch100.state.pd_joint_pos.physx_cpu.h5` | 4,154,779 | `cfa628ed8f23587d5f929b4285ffd7b4ac6b6f3bebd038b0e95b092e5b7c0240` |
| `pickcube_batch100.state.pd_joint_pos.physx_cpu.json` | 23,257 | `8def8dc02f9d06f1cb3a4b35a3a3bfd60aecff68d01773cf0a02b949c0c42eb4` |

偏差：检查脚本第一次因 f-string 内的转义引号语法错误没跑起来（临时脚本的问题），改写后通过；ubuntu 工作区保持干净，已有数据与 `.venv` 未动。

**阶段 C（数据中转）**：ubuntu `demos-batch/` → Mac（`rsync -a`，4 个新文件哈希与上表一致）→ 用 `cp -n` 复制进 Mac `data/pickcube/` 与 `data/pickcube/state/` → `sync.sh data` 同步到 wsl → `sync.sh manifest` 重新生成清单：`ubuntu-demos.sha256` 8 → 12 个文件，`wsl-data.sha256` 4 → 8 个文件，**只有新增、没有删除或改动**；Mac 副本对两份新清单校验通过。

### 其余五个任务试跑（9.24，ubuntu）

目的：确认另外 5 个任务的专家数据生成与 PickCube 一样可行，现有 MPlib 补丁够用。每个任务单进程生成 1 条成功轨迹（`--only-count-success -n 1`，5 分钟超时保护），再做 state 重放和 `validate_replay.py`。输出在 ubuntu `demos-taskprobe/<env>/motionplanning/probe_*`（共 608 KB，属于 `demos-*`，不进 git，未加入清单），日志 `logs/taskprobe_0924.log`。代码 `91937f8`，ubuntu 工作区保持干净。

| 任务 | 采集 | 重放 | 校验 | 种子 | 轨迹长度 | state 观测维数 | ManiSkill 默认回合上限 | 官方 DP 基线评估长度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PushCube-v1 | 3 s | 4 s | PASS | 0 | 71 | 35 | 50 | 100 |
| PullCube-v1 | 4 s | 3 s | PASS | 0 | 67 | 35 | 50 | 无 |
| StackCube-v1 | 4 s | 4 s | PASS | 0 | 107 | 48 | 50 | 200 |
| LiftPegUpright-v1 | 4 s | 4 s | PASS | 0 | 176 | 32 | 50 | 无 |
| PegInsertionSide-v1 | 7 s | 6 s | PASS | **2**（0、1 执行未成功） | 178 | 43 | 100 | 300 |

结论与影响：

- **六个任务的专家数据生成都已验证可行**，现有补丁 `patches/mani_skill_mplib_0_2_1.patch` 够用，没有遇到未适配的 MPlib 接口。整批 43 秒。
- **评估回合长度要按任务设**：ManiSkill 默认上限（50 / 100）是给强化学习调的，比示范短；StackCube、LiftPegUpright、PegInsertionSide 的示范已有 107–178 步，需要 200–300 步的回合。
- **GPU 时间随之增加**：评估耗时与回合长度成正比。按 PickCube 实测（训练约 9 分钟、100 步回合时 6 次验证约 3 分钟、测试约 1 分钟）估算，每组约为 PushCube / PullCube 13.5 分钟、StackCube（200 步）约 17 分钟、LiftPegUpright / PegInsertionSide（300 步）约 21 分钟；5 个任务 × 3 个训练种子约 **4.3 小时**。这是估算，示范变长后训练窗口数也会变，要实测修正。
- **PegInsertionSide 的专家成功率约 1/3**（这次的进度条显示 success_rate 0.333，失败是执行未成功而非规划报错），采集 100 条约需试 300 个种子。成功与否可能和初始状态有关，采集后要检查成功种子的分布。
- 观测维数各不相同（32–48），`dp_manip` 从数据读取维数，不需要改代码；每个任务需要一份自己的配置文件。

### 哈希

四个文件在 ubuntu、wsl、Mac 三端 SHA-256 一致，完整清单见 [manifests/](./manifests/)（9.23 生成并在 Mac 校验通过）：

| 文件                 | SHA-256                                                            |
| -------------------- | ------------------------------------------------------------------ |
| 原始 H5              | `cf892707024bb18db7eb822fe9fe89f161fbcdd959d66571bd45740dc36e5c18` |
| 原始 JSON            | `86496e8c9669bb59f9823f8bf192251bd6803fae76b8cb5d774e99119a3e5134` |
| state H5             | `e303bffc79aafd902e5e37c12262eb6d9b718541d2cc687cd011f3f0ae509292` |
| state JSON           | `5f395956a439fa1150a47a53fbdcf5d43ad37aaa7c07aa9efc03691b69be6b8a` |

---

## 五、控制模式：分析完成，尚未选择

当前 `pd_joint_pos` 数据是 8 维；ubuntu 上 ManiSkill 3.0.1 的 PickCube 环境实测 `pd_ee_delta_pos` 动作空间为 4 维、范围 `[-1, 1]`。官方 [IL 重放脚本](https://github.com/mani-skill/ManiSkill/blob/main/scripts/data_generation/replay_for_il_baselines.sh) 给 PickCube 采用后者，原则是选择仍能完成任务的最简单控制器；[控制器文档](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)将其描述为 3 维末端位移加 1 维夹爪。

由此推断，4 维有界末端位移可能比 8 维关节目标更容易学会当前的抓取任务，但尚未在本项目上做效果比较。若改用它，需先把专家轨迹重放转换为该模式、验证转换成功率，并让评估环境使用同一控制模式和动作尺度；不能把当前 8 维文件直接交给 4 维策略。最终模式尚未决定。

### 参考实现边界

[Stanford Diffusion Policy](https://github.com/real-stanford/diffusion_policy) 可参考其模型、序列采样器、归一化器和训练配置。其[原始环境](https://github.com/real-stanford/diffusion_policy/blob/main/conda_environment.yaml)固定 Python 3.9、PyTorch 1.12.1、CUDA 11.6，**不照搬**。其 [Robomimic 低维数据读取器](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py) 期望 `data/demo_i`，与 ManiSkill 的 `traj_i` 结构不同；[上游任务指南](https://github.com/real-stanford/diffusion_policy/blob/main/README.md#-adding-a-task)要求先有任务专属的 dataset、runner 和 shape 元数据，才能用 `train.py`。

---

## 六、过程回顾：Mac → Ubuntu → 第一条专家轨迹（9.22）

完整原始记录见 [docs/history.md](./docs/history.md)。

### Mac 上完成了什么

| 项目       | 状态         | 结果                                                      |
| ---------- | ------------ | --------------------------------------------------------- |
| Python     | ✅ 9.22 完成 | 使用 Python 3.11                                          |
| ManiSkill  | ✅ 9.22 完成 | 安装并能够运行（Mac 本地 `.venv`，gitignore）             |
| 图形显示   | ✅ 9.22 完成 | Vulkan / MoltenVK 路径可用                                |
| 机器人仿真 | ✅ 9.22 完成 | 能看到 PickCube 场景中的 Panda 机器人、红色方块和绿色目标 |
| 动作测试   | ✅ 9.22 完成 | 能让机器人执行随机动作；看到的“抽搐乱动”并不是专家策略    |
| Pinocchio  | ✅ 9.22 完成 | `pin` 已安装，`import pinocchio` 验证通过                 |
| MPlib      | ❌ 未完成    | 未能成功安装（`libclang==11.0.1` 缺少 macOS ARM64 wheel） |

因此决定让 Linux x86-64 服务器承担专家数据生成——**这是工作流分工，不是说 Mac 不能运行 ManiSkill。**

### Ubuntu 排障三阶段（均已解决）

1. **让 PickCube 场景正常运行**：专家命令最初直接 Segmentation fault。逐步缩小范围，确认 CPU 物理系统本身可工作，并用 `vulkaninfo` 验证 NVIDIA Vulkan 驱动可识别 GTX 1080 Ti；之后使用 `sim_backend="physx_cpu"` + `render_backend="cpu"`，成功完成 PickCube 创建、`reset()` 和关闭。
2. **找到能够加载 Panda 的 MPlib 版本**：原环境 `mplib 0.1.1` 在初始化 `ArticulatedModel` 时段错误（URDF/SRDF 文件齐全、依赖声明检查通过）；单独降 NumPy 到 1.26.4 又引起其他包冲突。于是创建隔离环境 `.venv-mplib-probe`，安装 `mplib 0.2.1`，成功加载 `panda_v2.urdf` / `panda_v2.srdf` 并初始化 Panda 规划器。
3. **适配新版 MPlib，生成专家轨迹**：ManiSkill 3.0.1 原本固定依赖 `mplib==0.1.1`，仅在测试环境覆盖该约束，并修复 `set_base_pose()` 与 `plan_screw()` 的接口差异；随后 PickCube 专家求解成功率 1/1，生成 H5 和 JSON。

### 临时环境（已被正式环境取代）

|          | 原环境 `.venv`                  | 临时成功的 `.venv-mplib-probe`                           |
| -------- | ------------------------------- | -------------------------------------------------------- |
| MPlib    | 0.1.1（专家规划初始化会段错误） | 0.2.1（PickCube 专家规划成功）                           |
| 适配方式 | —                               | 手工修改 site-packages，未做成可自动应用的补丁           |
| 现状     | —                               | ⚠️ 已被正式环境 `~/Coding/dp-manip/.venv` + 独立补丁取代 |

版本覆盖与 API 适配现已分别固化为 `mplib-probe-overrides.txt` 和 `patches/mani_skill_mplib_0_2_1.patch`。

### 第一条轨迹产出

| 指标       | 结果                                                                                   |
| ---------- | -------------------------------------------------------------------------------------- |
| 成功率     | 1 / 1                                                                                  |
| 轨迹长度   | 74 步                                                                                  |
| 规划失败率 | 0                                                                                      |
| 已保存文件 | H5 轨迹 29,734 字节 + JSON 元数据 871 字节（`demos-probe/`，已被 `demos-batch/` 取代） |
