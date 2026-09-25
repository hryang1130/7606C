# dp-manip

基于 ManiSkill 的专家轨迹生成 + Diffusion Policy 训练项目。本仓库（Mac）是**唯一权威源**，代码、配置、文档都在这里编辑，再同步到 ubuntu 与 wsl；本文件是入口，负责说明项目流程与进度、设备分工、数据流和常用命令。

- 当前状态：[STATUS.md](./STATUS.md)
- 未完成事项：[TODO.md](./TODO.md)
- 课程要求与进度对照：[docs/requirements.md](./docs/requirements.md)
- **操作说明（生成数据 → 传输 → 训练 → 测试，命令与文件位置）**：[docs/instructions.md](./docs/instructions.md)
- 面向编码代理的操作约束：[AGENT.md](./AGENT.md)
- 多设备工作流计划：[PLAN.md](./PLAN.md)
- Day 1 原始记录：[docs/history.md](./docs/history.md)

---

## 一、项目流程（组内三步 ↔ 仓库）

### 组内流程原文

1. 基于 ManiSkill 仿真环境，构造 6 个机器人操作任务（PickCube、PushCube 等），在仿真里生成专家轨迹数据集（观测-动作）。操作：下载 [ManiSkill](https://github.com/mani-skill/ManiSkill)，按 [quickstart](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html) 部署环境、仿真并生成轨迹。
2. 用第 1 步生成的轨迹训练 Diffusion Policy 策略模型。操作：[real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) 里的 `train.py`。
3. 在仿真环境里评估任务成功率；额外选做消融实验，比如轨迹数据量 / 数据质量消融、训练超参 / 网络结构消融，验证各模块贡献。

### 每一步在仓库里对应什么

| 步骤 | 机器 | 代码 / 配置 | 待办 | 进度（9.24） |
| --- | --- | --- | --- | --- |
| **1. 专家轨迹** | ubuntu 生成（运动规划 + state 重放）；Mac 中转；wsl 使用 | `run_cpu.py`、`patches/`；重放用 `mani_skill.trajectory.replay_trajectory`；校验用 `scripts/validate_replay.py`、`inspect_dataset.py` | [TODO C](./TODO.md) | 六个任务已选定：PickCube、PushCube、PullCube、StackCube、LiftPegUpright、PegInsertionSide。PickCube 已采集 100 条并做了 state 重放；其余 5 个任务各试跑 1 条，专家生成与回放全部通过（9.24），正式采集未开始 |
| **2. 训练 DP** | wsl | `dp_manip/`、`scripts/train_dp.py`、`configs/*.toml` | [TODO B](./TODO.md) | 训练代码完成；PickCube 基线：100 条示范，final.pt 测试成功率 **0.67**（3 个训练种子平均，范围 0.53–0.77；见 [docs/0923-2249.md](./docs/0923-2249.md)） |
| **3. 评估 + 消融** | wsl | `scripts/eval_dp.py`（`--split test/val/train`）、`scripts/replay_check.py` | [TODO D](./TODO.md) | 评估完成：固定测试种子、逐种子结果。数据量消融 PickCube 10/25/50/100 条 × 3 个训练种子已完成：平均 0.02 / 0.13 / 0.56 / 0.67 |

当前的做法是先在 PickCube 上把 1→2→3 做深：确认链路可靠、摸清需要多少数据，再铺到另外 5 个任务。

### 和组内流程的两处不同

1. **第 2 步没有直接用 Stanford 的 `train.py`。** 用的是 ManiSkill 官方的 DP 基线（`examples/baselines/diffusion_policy`），网络结构和 Stanford 版相同（1D 条件 UNet + DDPM），能直接读 ManiSkill 数据、直接在 ManiSkill 里评估。Stanford 版要先为每个任务写数据读取和评估模块才能接上 ManiSkill，而且固定 Python 3.9 / torch 1.12 的老环境。在此基础上的改动（动作归一化、验证 / 测试种子分离等）和出处见 [dp_manip/README.md](./dp_manip/README.md)，技术细节见 [STATUS.md](./STATUS.md)「DP 链路」。
2. **第 3 步的消融不是选做。** 课程大纲把「进一步研究」列为必做，单独占 20 分；评估与分析另占 25 分，要求 held-out 种子、分任务成功率、评估回合数和失败分析。详见 [docs/requirements.md](./docs/requirements.md)。

---

## 二、设备分工

| 设备               | SSH 访问        | 项目路径                                      | 硬件                                 | 角色                              | 关键限制                                                            |
| ------------------ | --------------- | --------------------------------------------- | ------------------------------------ | --------------------------------- | ------------------------------------------------------------------- |
| MacBook（本机）    | 本地            | `/Users/hollins/Documents/Coding/dp-manip`    | M3 Max / 36 GB                       | 权威仓库、编辑、文档、编排、数据中转、分析 | 装不了 mplib（`libclang==11.0.1` 无 macOS ARM64 wheel），不生成数据 |
| ubuntu             | `ssh ubuntu`    | `~/Coding/dp-manip`                           | i7-8700K / 16 GB / GTX 1080 Ti 11 GB | 专家轨迹生成 + state 重放         | GTX 1080 Ti（sm_61）与当前 PyTorch CUDA 构建不兼容，**不能训练**    |
| wsl                | `ssh wsl`       | `~/projects/dp-manip`                         | WSL2 / RTX 4090 / 驱动 591.86        | DP 训练与评估                     | 有 ManiSkill 3.0.1 评估环境，但没有 mplib 规划，不生成数据          |

> SSH 别名定义在 `~/.ssh/config`：`ubuntu` = 10.0.0.200，`wsl` = 10.0.0.248。另有 `ubuntu-frp` 走公网 frp，一般只在局域网不可达时使用。

**一句话原则：ubuntu 造数据，wsl 训模型，Mac 做编排和权威仓库；不要把训练放到 ubuntu，不要在 Mac 上折腾 mplib。**

---

## 三、仓库结构

```text
dp-manip/
  README.md AGENT.md STATUS.md TODO.md PLAN.md
  docs/history.md                           # Day 1 原始记录（原 in.txt）
  pyproject.toml uv.lock .python-version    # wsl 训练环境；只有 wsl 可以据此 uv sync
  dp_manip/                                 # Diffusion Policy 实现（改编自 ManiSkill 官方基线，见 dp_manip/README.md）
  scripts/                                  # 训练 / 评估 / 校验脚本（在 wsl 运行）
  configs/                                  # 任务级 dataset / training / evaluation 配置
  run_cpu.py patches/ environment/ mplib-probe-overrides.txt   # ubuntu 专家环境
  manifests/                                # 数据 SHA-256 清单（入库）
  demos-*/ data/                            # 数据，gitignore，rsync 传输
  results/ checkpoints/ logs/               # 训练产出，gitignore，从 wsl 回传
```

`.venv/` 在三端各自独立、互不相同，全部 gitignore。Mac 本地 `.venv` 是 9.22 装的 ManiSkill 仿真环境（无 mplib），ubuntu 的是专家环境，wsl 的是训练环境。

---

## 四、同步方式

所有 git 与 rsync 操作都由 **Mac 发起**，远端从不主动连接 Mac（细节见 [PLAN.md](./PLAN.md)）：

- **代码 / 配置 / 文档**：Mac 提交 → `git push ubuntu main` / `git push wsl main`。远端仓库设置了 `receive.denyCurrentBranch=updateInstead`，远端有未提交改动时 push 会被拒绝。
- **wsl 上的临时修改**：在 wsl 本地提交 → Mac `git fetch wsl && git merge --ff-only wsl/main`。
- **数据**：rsync 传输，传完用 `manifests/*.sha256` 校验。
- **训练产出**：从 wsl rsync `results/ checkpoints/ logs/` 回 Mac，不带 `--delete`。

日常用 `scripts/sync.sh`（仅在 Mac 执行）：

```bash
scripts/sync.sh status         # 三端 HEAD、工作区、数据清单
scripts/sync.sh push           # Mac 提交后推到 ubuntu / wsl
scripts/sync.sh fetch          # 取回 wsl 上的提交（仅 fast-forward）
scripts/sync.sh data           # ubuntu demos-*/ → Mac，Mac data/ → wsl，并校验
scripts/sync.sh manifest       # 新数据产生后重新生成清单，再提交
scripts/sync.sh pull-results   # wsl 训练产出 → Mac
```

---

## 五、数据流

```text
ubuntu                                          wsl
──────                                          ───
run_cpu.py
  └─ 专家轨迹 (motionplanning)  ─── rsync ──▶  data/pickcube/*.h5|json
replay_trajectory --obs-mode state
  └─ state 观测重放              ─── rsync ──▶  data/pickcube/state/*.h5|json
                                                       │
                                                       ▼
                                                 DP dataset → 训练 → 评估
                                                       │
Mac：权威仓库，发起所有同步；数据经 Mac 中转  ◀── rsync ─┘ results / checkpoints / logs
```

原始专家文件只有动作，`obs` 组为空；可训练数据是 `obs_mode: state` 的重放版本。两者都要传递。

---

## 六、各设备环境

### ubuntu（专家数据源）

| 项目          | 值                                                                 |
| ------------- | ------------------------------------------------------------------ |
| 系统/硬件     | Ubuntu Server 26.04，i7-8700K，16 GB RAM，GTX 1080 Ti 11 GB         |
| NVIDIA 驱动   | 580.178.04                                                          |
| Python        | 3.11.15，虚拟环境 `~/Coding/dp-manip/.venv`                         |
| 核心包        | `mani-skill==3.0.1`、`mplib==0.2.1`、`sapien==3.0.3`、`numpy==1.26.4` |
| 依赖冻结      | `environment/ubuntu-expert-freeze.txt`                             |
| MPlib 覆盖    | `mplib-probe-overrides.txt`（`mplib==0.2.1`）                       |
| 适配补丁      | `patches/mani_skill_mplib_0_2_1.patch`（`set_base_pose` / `plan_screw`） |
| 入口          | `run_cpu.py`（CPU 物理 + CPU 渲染）                                 |

运行方式：`sim_backend="physx_cpu"` + `render_backend="cpu"`，并通过 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` 指定 Vulkan 驱动。

> ⚠️ 仓库根目录的 `pyproject.toml` / `uv.lock` 是 wsl 的训练环境。**ubuntu 上禁止 `uv sync`**：uv 会把 `.venv` 精确同步成那份 lock，卸掉 mani-skill 与 mplib。ubuntu 环境只按 freeze 文件与补丁重建。

> 旧环境 `~/Coding/dp-manip-old-backup`（含 `.venv-mplib-probe`）和 `~/Coding/dp-manip-clean-venv-backup` 仅作备份，不要在其上继续工作，也不要删除，除非 TODO E 确认。

### wsl（训练节点）

| 项目        | 值                                                                |
| ----------- | ----------------------------------------------------------------- |
| 系统/硬件   | WSL2 Ubuntu 24.04.5 LTS，RTX 4090（计算能力 8.9）                  |
| Python      | 3.11.15，虚拟环境 `~/projects/dp-manip/.venv`，由 `uv 0.12.18` 管理 |
| 核心依赖    | PyTorch `2.14.0+cu130`、NumPy `1.26.4`、h5py `3.16.0`             |
| GPU 驱动    | 591.86（`nvidia-smi` 报 CUDA 13.1，PyTorch 运行时 CUDA 13.0）      |
| 评估与 DP   | mani-skill 3.0.1、sapien 3.0.3、gymnasium 1.3.0（与 ubuntu 一致），diffusers 0.40.0 |
| 校验脚本    | `scripts/verify_cuda.py`、`scripts/smoke_train_cuda.py`、`scripts/check_dp_offline.py`、`scripts/replay_check.py` |
| DP 脚本     | `scripts/train_dp.py`（训练 + 验证）、`scripts/eval_dp.py`（测试种子评估）、`scripts/render_episodes.py`（按保存的状态离线渲染视频） |
| 数据脚本    | `scripts/inspect_dataset.py`、`scripts/validate_replay.py`、`scripts/check_temporal_windows.py` |

复现环境（wsl 项目根）：

```bash
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv --python 3.11.15
uv sync --frozen
```

venv 无 pip 模块，用 `uv pip ... --python .venv/bin/python` 操作。

### MacBook（本机）

M3 Max / 36 GB。本地 `.venv`（Python 3.11，ManiSkill + Vulkan/MoltenVK）可做仿真与可视化，但 mplib 装不上，所以只承担编辑、文档、数据中转和分析。**Mac 上同样不要 `uv sync`**，否则会把本地 ManiSkill 环境替换成 wsl 的训练依赖。

---

## 七、数据资产（PickCube：10 条批次 + 100 条批次，全部成功）

| 名称                     | 内容                                        | 维度                                                        |
| ------------------------ | ------------------------------------------- | ----------------------------------------------------------- |
| 原始专家轨迹             | `pickcube_batch10.h5` + `.json`             | `actions float32 (T, 8)`；`obs` 为空                        |
| state 重放               | `pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json}` | `obs float32 (T+1, 42)`；`actions float32 (T, 8)` |
| 100 条批次（原始 + 重放） | `pickcube_batch100{,.state.pd_joint_pos.physx_cpu}.{h5,json}` | 同上 |

- 10 条批次：动作长度 74、74、50、86、76、88、71、74、49、84（共 726 步），seed 0–9。
- 100 条批次（9.23）：seed 0–100 去掉规划失败的 51；长度 49–99，共 7720 步；**前 10 条与 10 条批次逐值相同**，取前 N 条即可得到嵌套子集。详见 [STATUS.md](./STATUS.md)。
- 控制模式 `pd_joint_pos`，仿真/渲染后端均为 CPU。
- H5 顶层为 `traj_0`…`traj_9`，每条一个组；JSON `episodes[].episode_id` 是边界映射。逐步 `terminated`/`truncated` 可能提前变真，切分只能用组 + JSON ID。
- 三端 SHA-256 一致，清单见 `manifests/ubuntu-demos.sha256`、`manifests/wsl-data.sha256`；各文件哈希也列在 [STATUS.md](./STATUS.md)。

ubuntu 路径：`~/Coding/dp-manip/demos-batch/PickCube-v1/motionplanning/`
wsl 路径：`data/pickcube/`（原始）与 `data/pickcube/state/`（重放）
Mac：同时保存两种布局（`demos-*/` 与 `data/`），用作中转和备份。

---

## 八、常用命令

### ubuntu：生成专家轨迹

```bash
cd ~/Coding/dp-manip
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
.venv/bin/python run_cpu.py --env-id PickCube-v1 \
  --sim-backend physx_cpu --only-count-success -n 10 \
  --traj-name pickcube_batch10 --record-dir demos-batch
```

### ubuntu：重放为 state 观测

```bash
cd ~/Coding/dp-manip
.venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch10.h5 \
  --obs-mode state --save-traj --use-env-states \
  --max-retry 0 --num-envs 1 --verbose
```

### Mac：数据中转（ubuntu → Mac → wsl）

```bash
rsync -a ubuntu:Coding/dp-manip/demos-batch ./
mkdir -p data/pickcube/state
cp demos-batch/PickCube-v1/motionplanning/pickcube_batch10.{h5,json} data/pickcube/
cp demos-batch/PickCube-v1/motionplanning/pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json} data/pickcube/state/
rsync -a data/ wsl:projects/dp-manip/data/
shasum -a 256 -c manifests/wsl-data.sha256
```

### Mac：重新生成清单（新数据产生后）

```bash
ssh ubuntu 'cd ~/Coding/dp-manip && find demos-* -type f | LC_ALL=C sort | xargs sha256sum' > manifests/ubuntu-demos.sha256
ssh wsl 'cd ~/projects/dp-manip && find data -type f | LC_ALL=C sort | xargs sha256sum' > manifests/wsl-data.sha256
```

### wsl：环境与数据校验

```bash
cd ~/projects/dp-manip
.venv/bin/python scripts/verify_cuda.py
.venv/bin/python scripts/smoke_train_cuda.py --amp
.venv/bin/python scripts/inspect_dataset.py            # 默认检查 state 文件；也可传 path/to/file.h5 [--json path/to/file.json]
.venv/bin/python scripts/check_temporal_windows.py
.venv/bin/python scripts/validate_replay.py \
  data/pickcube/pickcube_batch10.h5 \
  data/pickcube/state/pickcube_batch10.state.pd_joint_pos.physx_cpu.h5
```

`inspect_dataset.py` 会打印元数据、每条轨迹的形状与类型、数值统计、布尔计数和对齐警告；`check_temporal_windows.py` 用历史长度 2、动作 horizon 8 验证时间窗口不跨 episode。

---

## 九、当前状态与下一步

- 流程进度见上文「一、项目流程」；详细状态见 [STATUS.md](./STATUS.md)，待办见 [TODO.md](./TODO.md)，每次运行的记录在 `docs/mmdd-hhmm.md`。
- 刚完成：PickCube 数据量消融 10/25/50/100 条 × 3 个训练种子，平均 0.02 / 0.13 / 0.56 / 0.67，见 [docs/0923-2249.md](./docs/0923-2249.md)。
- 报告口径：主结果固定用 final.pt，不按验证挑选（见 [configs/README.md](./configs/README.md)）。
- 之后：其余 5 个任务的专家数据（TODO C）→ 六任务基线 → 选定并完成研究实验。
