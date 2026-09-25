# AGENT.md

给在这台 Mac 上、需要跨三台设备工作的编码代理的操作说明。先读 [README.md](./README.md) 了解全貌，多设备同步方式见 [PLAN.md](./PLAN.md)；本文件只讲**怎么安全地动手**。

## 0. 设备与角色（先确认你在哪）

| 别名 / 设备    | 连接方式     | 项目路径                                    | 干什么                              | 绝对不能干什么                     |
| -------------- | ------------ | ------------------------------------------- | ----------------------------------- | ---------------------------------- |
| 本机 MacBook   | 直接执行     | `/Users/hollins/Documents/Coding/dp-manip`  | 权威仓库：编辑、文档、编排、数据中转、分析 | 装 mplib；跑专家生成或训练；`uv sync` |
| ubuntu         | `ssh ubuntu` | `~/Coding/dp-manip`                         | 生成专家轨迹、state 重放            | 跑训练；改/升级 venv 或已有数据；改代码；`uv sync` |
| wsl            | `ssh wsl`    | `~/projects/dp-manip`                       | DP 训练与评估、数据校验             | 当作数据生成机（有 ManiSkill 但没有可用的 mplib 规划） |

核心原则：**ubuntu 造数据，wsl 训模型，Mac 做编排和权威仓库。** 三台角色不要串。

## 1. 硬性约束（Hard rules）

1. **不要在 ubuntu 上训练。** GTX 1080 Ti 是 `sm_61`，与当前 PyTorch CUDA 构建不兼容。任何训练/评估都放 wsl。
2. **不要在 Mac 上尝试安装 mplib。** `libclang==11.0.1` 没有 macOS ARM64 wheel，这是已知死路，别再排查。Mac 只做非 mplib 的工作。
3. **ubuntu 的 `.venv` 和 `demos-*` 数据视为只读。** 生成新数据会新建文件；除明确要求外，不删不改不升级已通过验证的文件。
4. **不删除备份。** `~/Coding/dp-manip-old-backup`、`~/Coding/dp-manip-clean-venv-backup` 保留到 TODO E 确认后才处理。
5. **wsl 的 ManiSkill 只用于评估。** 9.23 起装有 mani-skill 3.0.1 / sapien 3.0.3 / gymnasium 1.3.0，版本与 ubuntu 一致，不要单独升级；新增依赖先写进 `pyproject.toml` 再在 wsl `uv lock`，不要直接套用 Stanford DP 的旧环境（Python 3.9 / torch 1.12）。
6. **大文件与产物不入库。** `.gitignore` 忽略 `.venv/`、`data/`、`demos-*/`、`checkpoints/`、`logs/`、`results/`。不要提交 `.h5` 或 checkpoint。数据用 rsync 传输，一致性靠入库的 `manifests/*.sha256`；新数据生成后先更新清单再提交。
7. **按 `sync.sh` 流程提交，push 前需用户确认。** Mac 是唯一权威仓库；不改写已推送的历史，不 `--force`，不绕过 `updateInstead` 的拒绝。
8. **跨设备命令要显式、可复现。** 优先用 `ssh <alias> '<cmd>'` 单条执行，避免在远端做交互式、破坏性操作。
9. **只有 wsl 可以 `uv sync` / `uv lock`。** 仓库根的 `pyproject.toml` / `uv.lock` 是 wsl 训练环境。在 ubuntu 上执行会把 `.venv` 精确同步成这份 lock、卸掉 mani-skill 与 mplib；在 Mac 上执行会替换本地 ManiSkill 环境。ubuntu 环境只按 `environment/ubuntu-expert-freeze.txt` + 补丁重建，也不要 `uv pip install`。
10. **默认只在 Mac 编辑。** wsl 允许临时热修并本地提交，由 Mac `git fetch wsl` 后 `--ff-only` 合入；ubuntu 只运行，不改代码。同步都由 Mac 发起，远端不主动连 Mac。

## 2. 远端访问

```bash
ssh ubuntu 'hostname'   # 10.0.0.200，局域网
ssh wsl    'hostname'   # 10.0.0.248，局域网
ssh ubuntu-frp '...'    # 公网 frp，仅在 ubuntu 不可达时使用
```

SSH 别名定义在 `~/.ssh/config`。批量传输用 `rsync -a`（不带 `--delete`），从 Mac 发起的路径形如 `ubuntu:Coding/dp-manip/...`（`~` 会被省略）。git remote 也用别名：`ubuntu:Coding/dp-manip`、`wsl:projects/dp-manip`。

## 3. 环境事实（不要凭记忆猜版本）

### ubuntu

- Ubuntu Server 26.04，i7-8700K / 16 GB / GTX 1080 Ti 11 GB，驱动 580.178.04。
- venv：`~/Coding/dp-manip/.venv`，Python 3.11.15。
- `mani-skill==3.0.1`、`mplib==0.2.1`、`sapien==3.0.3`、`numpy==1.26.4`（精确冻结见 `environment/ubuntu-expert-freeze.txt`）。
- 关键：ManiSkill 3.0.1 元数据仍要求 `mplib==0.1.1`，这里通过 `mplib-probe-overrides.txt` 覆盖为 0.2.1，并用 `patches/mani_skill_mplib_0_2_1.patch` 适配 `set_base_pose()` / `plan_screw()`。重建环境时必须重新应用，别只重装包。
- 运行必须带 `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`，且 `sim_backend="physx_cpu"`、`render_backend="cpu"`。

### wsl

- WSL2 Ubuntu 24.04.5，RTX 4090，驱动 591.86，`torch 2.14.0+cu130`（运行时 CUDA 13.0）。
- venv：`~/projects/dp-manip/.venv`，Python 3.11.15，由 `uv 0.12.18` 管理；无 pip，用 `uv pip ... --python .venv/bin/python`。
- 复现：`UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_CACHE_DIR="$PWD/.uv-cache" uv venv --python 3.11.15 && uv sync --frozen`。
- 不要重复跑已通过的 CUDA/训练 smoke test，除非环境变了或出错。
- 非交互 SSH 的 PATH 不含 `uv`（`~/.local/bin`）和 `nvidia-smi`（`/usr/lib/wsl/lib`），远端命令要补 PATH 或写全路径。
- 没有 NVIDIA Vulkan ICD。视频渲染走 Mesa lavapipe（`lvp_icd.json`，CPU 软件渲染）+ `render_backend="cpu"`，由 `dp_manip/envs.py` 的 `ensure_render_icd()` 自动设置（9.23 验证）。不要为此装驱动，也不要改系统里的 ICD。

### Mac

- M3 Max / 36 GB。本地 `.venv`（Python 3.11）有 ManiSkill + Vulkan/MoltenVK，可仿真；mplib 不可用。
- 本目录是权威 git 仓库，汇总了三端的代码、配置与文档；`demos-*/`、`data/` 是数据副本（gitignore）。

## 4. 数据约定

- **原始轨迹不能直接训练**：`obs` 组为空，只有 `actions`。必须先在 ubuntu 做 `--obs-mode state` 重放。
- **H5 结构**：顶层 `traj_0`…`traj_9`，每组一条轨迹。`actions` 为 `float32 (T, 8)`（`pd_joint_pos`）。
- **state 重放**：`obs` 为 `float32 (T+1, 42)` 的扁平数组；`actions` 不变。
- **episode 边界**：只用 H5 `traj_*` 组 + JSON `episodes[].episode_id`，**不要**用逐步 `terminated`/`truncated`——它们可能提前变真；JSON 里的 `max_episode_steps=50` 也小于部分实际轨迹长度。
- **对齐**：`obs` 是 T+1，动作是 T；时间窗口按 `state[t-1:t+1]` + `actions[t:t+H]` 组织。
- 完整性：所有数值维 `float32`，无 NaN/Inf。两端的原始与重放文件 SHA-256 必须一致。

当前资产（PickCube，10/10 成功，726 步）：

| 文件 | ubuntu | wsl |
| --- | --- | --- |
| 原始 | `demos-batch/PickCube-v1/motionplanning/pickcube_batch10.{h5,json}` | `data/pickcube/pickcube_batch10.{h5,json}` |
| state | `...pickcube_batch10.state.pd_joint_pos.physx_cpu.{h5,json}` | `data/pickcube/state/...` |

## 5. 常用命令（照抄，改动前想清楚）

### ubuntu — 生成专家轨迹

```bash
ssh ubuntu 'cd ~/Coding/dp-manip && \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  .venv/bin/python run_cpu.py --env-id PickCube-v1 \
    --sim-backend physx_cpu --only-count-success -n 10 \
    --traj-name pickcube_batch10 --record-dir demos-batch'
```

### ubuntu — state 重放

```bash
ssh ubuntu 'cd ~/Coding/dp-manip && \
  .venv/bin/python -m mani_skill.trajectory.replay_trajectory \
    --traj-path demos-batch/PickCube-v1/motionplanning/pickcube_batch10.h5 \
    --obs-mode state --save-traj --use-env-states \
    --max-retry 0 --num-envs 1 --verbose'
```

### Mac — 传输并校验数据

```bash
rsync -a ubuntu:Coding/dp-manip/demos-batch ./
shasum -a 256 -c manifests/ubuntu-demos.sha256
rsync -a data/ wsl:projects/dp-manip/data/
ssh wsl 'cd ~/projects/dp-manip && sha256sum -c -' < manifests/wsl-data.sha256
```

### wsl — 校验

```bash
ssh wsl 'cd ~/projects/dp-manip && \
  .venv/bin/python scripts/verify_cuda.py && \
  .venv/bin/python scripts/inspect_dataset.py && \
  .venv/bin/python scripts/check_temporal_windows.py && \
  .venv/bin/python scripts/check_dp_offline.py --config configs/pickcube_state_jointpos.toml --device cuda && \
  .venv/bin/python scripts/replay_check.py --config configs/pickcube_state_jointpos.toml'
```

## 6. 代码放哪

所有代码都在 Mac 仓库编辑，同步到远端后运行：

- `scripts/`：数据校验现在在此；后续转换、训练、评估入口，以及 `sync.sh`，也放这里（在 wsl 运行，`sync.sh` 在 Mac 运行）。
- `configs/`：任务级 dataset / training / evaluation 配置；观测历史、动作 horizon、归一化规则必须写进配置，训练与评估共用同一份定义。
- `run_cpu.py`、`patches/`、`environment/`、`mplib-probe-overrides.txt`：ubuntu 专家环境，只在 ubuntu 运行。
- `checkpoints/`、`logs/`、`results/`：wsl 本地输出，rsync 回 Mac 存档，忽略入库。

## 7. 汇报与文档

- 每个阶段结束更新 Mac 仓库里的 `STATUS.md` 与 `TODO.md`（统一中文，一套文档，不再分设备维护）。
- 引用代码位置用 `file_path:line`。
- 修改完成后运行对应的校验脚本；改到 wsl 训练代码时至少跑一遍 `verify_cuda.py` 和相关的 `inspect_*` / `validate_replay.py`。
