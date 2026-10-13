# maniskill-demogen

为 DASC7606C Track 3 生成 ManiSkill 3 示范数据（RGB + state），**每人负责一个任务，一条命令生成这个任务的全部数据**。流程：运动规划专家 → 回放并录下观测（rgb、state 各一次；回放未成功的丢弃）→ 修正第 0 帧 → 导出成与 ManiSkill 官方示范相同的格式（训练 400 条、验证 50 条）。

任务：`pickcube`、`stackcube`、`pushcube`、`pullcube`、`peginsertionside`、`plugcharger`、`placesphere`、`liftpegupright`。

新增 `placesphere`（PlaceSphere-v1）沿用官方运动规划专家，导出控制模式为
`pd_ee_delta_pos`，原始示范 440 + 55 条、导出训练/验证 400 + 50 条，种子段与其他任务相同。
运行 `./generate.sh placesphere` 或 `sbatch --export=ALL,TASK=placesphere slurm/generate_task.sbatch`。
该任务尚未进行真实仿真及控制模式转换成功率验证。

`liftpegupright`（LiftPegUpright-v1）由备用任务提升为正式任务，保留官方专家及
`pd_joint_pos`（8 维）配置，原始示范 440 + 55 条、导出训练/验证 400 + 50 条。
运行 `./generate.sh liftpegupright` 或
`sbatch --export=ALL,TASK=liftpegupright slurm/generate_task.sbatch`。
此次接入尚未进行真实生成验证；已有任务的参数与种子段保持不变。

## 快速开始（在集群上，每人一个任务）

需要：Linux x86_64，能上网的登录节点，带 NVIDIA GPU 的计算节点。不需要 sudo、conda、CUDA。

**1. 登录节点：安装 uv、拉取仓库、装环境（一次，约 2 分钟）**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # 已有 uv 可跳过
export PATH="$HOME/.local/bin:$PATH"                 # 非交互式 shell 里需要手动加
git clone https://github.com/hollinsStuart/maniskill-demogen.git
cd maniskill-demogen
./setup.sh
```

最后一行是 `Environment ready. Next: ./check_env.sh` 即成功。环境装在仓库里的 `.venv/`（Python 3.11 由 uv 下载），计算节点通过共享的家目录直接使用。

**2. 计算节点：检查这台节点能不能生成（一次，几分钟）**

先申请一个 GPU 节点（HKU 集群用 `gpu-interactive`；一般 SLURM 用 `srun --gres=gpu:1 --cpus-per-task=4 --pty bash` 或 `salloc`），然后：

```bash
cd ~/maniskill-demogen
./check_env.sh 2>&1 | tee check_env.log
```

最后一行是 `=== check passed` 即可；`=== rendering speed` 一段给出渲染耗时（RTX 4080 约 1–2 ms/步）。

**3. 计算节点：生成自己的任务**

```bash
./generate.sh <task>
```

或者提交成批处理作业（断网、退出登录都不影响）：

```bash
sbatch --export=TASK=<task> slurm/generate_task.sbatch
```

结束时打印训练数据的位置和条数，例如：

```
=== plugcharger (PlugCharger-v1) dataset
train 400 demos, ... action 8 (pd_joint_pos), obs 46, obs_rgb/state 25, images [128, 128, 6]
      data/dataset/train/PlugCharger-v1/motionplanning/trajectory.state.pd_joint_pos.physx_cpu.h5
val    50 demos, ...
preview (first and last frames): data/dataset/train/PlugCharger-v1/motionplanning/sample.png
=== plugcharger done
```

**交给训练的是 `data/dataset/`** 下的 `.h5`（及同名 `.json`、`.export_info.json`）；`data/work/` 是中间文件。完整输出在 `data/logs/<task>.out`。**中断后重新执行同一条命令，会从断点继续**，已完成的步骤不会重做。

预计（单个任务，RTX 4080 节点）：4 维任务约 0.5–1 小时，PegInsertionSide、PlugCharger 约 1–2 小时（由 ubuntu 实测推算）。磁盘每个任务约 2–4 GB；家目录放不下时加 `--out /scratch 下的目录`。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"`（uv 安装脚本只改交互式 shell 的配置） |
| `setup.sh` 下载失败 | 所在节点不能上网，换到登录节点执行 |
| `libGL.so.1` 缺失 | 已处理：仓库用 `opencv-python-headless`；若仍出现，先 `git pull` 再 `./setup.sh` |
| `Failed to find system libvulkan` 警告 | 可以忽略，SAPIEN 自带 libvulkan，照样走 NVIDIA 驱动 |
| `check_env.sh` 渲染失败 | 看 `ls /usr/share/vulkan/icd.d/`：有 `nvidia_icd.json` 就能用；驱动的 json 在别处时 `export VK_ICD_FILENAMES=<路径>` |
| 某一步报 `exists but is not marked done` | 上次被中断留下了半截文件（或另一个进程正在写）。确认没有别的进程在跑后，删掉报错里的那个文件再重跑 |
| 导出报 `only N usable demos, need 400` | 回放成功的不够：删掉 `data/work/<Env>/motionplanning/` 下对应 split 的文件，用更大的 `--n-train` / `--n-val` 重跑 |
| 作业数量受限 | 一个任务只占一个作业；也可在已有分配里 `srun --jobid=<id> --overlap ./generate.sh <task>`（放在 tmux 里防断线） |

## 需要什么

| | 说明 |
| --- | --- |
| 系统 | Linux x86_64（mplib 只有这个平台的 wheel），无需 sudo、conda、CUDA |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh`；安装时要能访问 PyPI 和 download.pytorch.org |
| CPU / 内存 | 每个任务 4 核、16 GB 足够（仿真和运动规划都在 CPU 上，一个任务同一时间只跑一个进程） |
| **渲染（只有 rgb 需要）** | 一个 Vulkan 设备，二选一：<br>• NVIDIA GPU + 带 Vulkan 的驱动（`/usr/share/vulkan/icd.d/nvidia_icd.json`）：快<br>• 纯 CPU 节点 + Mesa lavapipe（`/usr/share/vulkan/icd.d/lvp_icd.json`）：慢 7–17 倍，但 4 核就够 |

不需要 GPU 算力：torch 是 CPU 版，GPU 只用来渲染。

## 进阶用法

`generate.sh` 把参数原样传给 `generate_task.py`，也可以直接调用它：

```bash
.venv/bin/python generate_task.py pickcube --stages expert        # 只跑某些阶段
.venv/bin/python generate_task.py --help                          # 全部参数
```

一个人要跑多个任务、而作业数量有限时，`run_all.sh` 在一次分配里并行跑（同时跑的个数 = CPU 数）：

```bash
sbatch slurm/run_all.sbatch                                  # 一个作业，4 CPU + 1 GPU，八个任务
JOBS=2 ./run_all.sh pickcube stackcube                       # 在已有分配里跑其中几个
```

常用参数（`generate_task.py --help` 有完整说明）：

| 参数 | 作用 |
| --- | --- |
| `--stages expert,rgb,state,first,export,stats` | 只跑其中几个阶段 |
| `--control-mode MODE` | 覆盖 `tasks.py` 里的控制模式 |
| `--n-train / --n-val` | 生成的原始专家条数（默认见 `tasks.py`，PlugCharger 多生成） |
| `--export-train 400 --export-val 50` | 导出前 N 条可用示范（final-plan §2.2） |
| `--out` | 输出根目录，默认 `./data`；八个任务可以共用同一个 |

## 控制模式与条数（`tasks.py`）

| 任务 | 控制模式 | 动作维 | 原始条数（训练 + 验证） | 重放成功率（wsl 实测） |
| --- | --- | --- | --- | --- |
| PickCube、StackCube、PushCube、PullCube | `pd_ee_delta_pos` | 4 | 440 + 55 | ≈100% |
| PegInsertionSide | `pd_joint_pos` | 8 | 440 + 55 | 97% |
| PlugCharger | `pd_joint_pos` | 8 | 600 + 80 | 80% |

7 维任务 9.25 定为 `pd_joint_pos`（转成 `pd_ee_delta_pose` 只剩 60–73%）。**它的动作是关节绝对目标角（弧度），约 30% 的数值在 [-1, 1] 之外**：训练代码必须对动作做归一化，执行前也不能把动作裁剪到 [-1, 1]（VariDP `train/eval.py` 第 156 行正是这样裁剪的；7606-train-template 不做动作归一化，两者都要改）。

导出取每个 split 的前 400 / 50 条可用示范。万一不够，导出会报「only N usable demos」：删掉该任务 `work/` 下对应 split 的文件（和 `.done`），用更大的 `--n-train` / `--n-val` 重跑；生成是确定性的，前面的示范会原样再生成。

## 输出

```
data/work/<Env>/motionplanning/{train,val}.h5                        原始专家（pd_joint_pos，无观测）
data/work/<Env>/motionplanning/{split}.{rgb,state}.<mode>.physx_cpu.h5 两次转换 + .first_obs.h5 旁路文件
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.h5   ← 交给训练的数据
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.json      官方字段
data/dataset/{train,val}/<Env>/motionplanning/trajectory.state.<mode>.physx_cpu.export_info.json  维度、种子、丢弃与修正记录、哈希
data/dataset/train/<Env>/motionplanning/sample.png                   前 5 条的首帧与末帧
data/logs/<task>.log                                                 每次运行的步骤汇总
```

HDF5 与队友验证过的官方示范同结构（`traj_N/obs, actions, success, terminated, truncated, env_states`），另加：

- `traj_N/obs_rgb/rgb`：(T+1, 128, 128, 3×相机数) uint8；StackCube、PegInsertionSide、PlugCharger 有腕部相机，为 6 通道；
- `traj_N/obs_rgb/state`：obs_mode=rgb 的 agent + extra（非特权），与评估时 `FlattenRGBDObservationWrapper(env, rgb=True, depth=False)` 的 `obs["state"]` 逐项对应。

**RGB 策略用 `obs_rgb/rgb` + `obs_rgb/state`，state 策略用 `obs`**（`obs` 含物体位姿）。`traj_N` 按种子排序，前 N 条即嵌套子集。

### 相机视角：RGB 策略看到的画面

![六个任务的相机视角](docs/camera_views.png)

每行一个任务（各取第一条训练示范），左 3 列是固定相机 `base_camera`，右 3 列是腕部相机 `hand_camera`，各取开头、中间、结尾一帧（128×128，图中放大 2 倍）。

- PickCube、PushCube、PullCube 只有 `base_camera`；StackCube、PegInsertionSide、PlugCharger 还有 `hand_camera`（`obs_rgb/rgb` 为 6 通道，前 3 通道是 `base_camera`）。
- PullCube 的固定相机从另一侧看；PegInsertionSide 是贴近桌面的侧视角。
- **PickCube 的目标点对相机不可见**，策略只能从 `obs_rgb/state` 里的 `goal_pos` 得知目标。
- 物体在固定相机里很小（PlugCharger 的插头只有十几个像素），高精度任务要多靠腕部相机。

这张图来自 9.25 的小规模冒烟数据（当时 PegInsertionSide、PlugCharger 用 `pd_ee_delta_pose`）；控制模式不影响相机和场景，画面与现在的 `pd_joint_pos` 数据相同。自己生成的数据可以看 `sample.png`（只有 `base_camera`）。

## 每个阶段做什么

| 阶段 | 脚本 | 说明 |
| --- | --- | --- |
| expert | `scripts/run_cpu.py` | ManiSkill 官方运动规划，只保留成功的；训练池从种子 0、验证示范从 4000 开始 |
| rgb | `mani_skill.trajectory.replay_trajectory -c <mode> --allow-failure -o rgb --shader minimal` | 相机 shader 与普通 `gym.make` 的评估环境一致；每条示范单独保存，失败的在导出时丢弃 |
| state | 同上，`-o state` | CPU 物理是确定性的，两次转换逐步一致，导出时校验 |
| first | `scripts/first_frame_obs.py` | 修正第 0 帧 |
| export | `scripts/export_demos.py`、`scripts/preview_rgb.py` | 丢弃重放未成功的示范，取前 N 条 |
| stats | `scripts/replay_stats.py` | 专家成功率、转换成功率、丢弃的种子、长度、相机（final-plan §2.3） |

每个步骤成功后写 `<输出>.done`，有标记就跳过；有输出却没有标记（被打断或另一个作业在写）时报错，不覆盖，需要人工检查后删除。

> 每个阶段的计数方式、数据流、失败点与每步的数据量，见 [docs/pipeline.md](docs/pipeline.md)。

## ManiSkill 3.0.1 的已知问题（本仓库已处理）

1. `replay_trajectory --use-env-states` 录下的是「从设定状态走一步」的预测，不是状态本身 → state 与 rgb 各自独立转换。
2. 某条示范重放失败时，录制缓冲区不清空，它的步骤会被拼到下一条保存的示范前面（重置处的动作是随机的）→ 重放一律加 `--allow-failure`，每条单独保存，导出时丢弃失败的；另外单步关节跳变超过 0.2 rad 的示范也丢弃（第二道保险）。
3. 第 0 帧的接触类观测是上一条示范的残留（PickCube `is_grasped`）→ `first_frame_obs.py` 重算。
4. 转换文件的 `env_states[0]` 错了一位（记的是 t=1）→ 同上一起替换。

官方下载的示范（队友用过的）同样带有 3、4 两个问题。

## 环境

`pyproject.toml` / `uv.lock` 复刻 dp-manip ubuntu 专家环境的版本：Python 3.11、mani-skill 3.0.1、sapien 3.0.3、mplib 0.2.1（覆盖 ManiSkill 要求的 0.1.1）、numpy 1.26.4、gymnasium 1.3.0、torch 2.14.0 CPU 版；OpenCV 用 `opencv-python-headless`（mani-skill 默认依赖的 `opencv-python` 需要系统的 `libGL.so.1`，HKU 集群登录节点没有）。`setup.sh` 再对 `.venv` 里的 mani_skill 打 `patches/mani_skill_mplib_0_2_1.patch`（mplib 0.2.1 的 API 变化）；uv 用复制模式安装，补丁不会改到 uv 缓存。

## 已验证

- ubuntu（GTX 1080 Ti，NVIDIA Vulkan）：dp-manip 的同一套脚本生成 6 个任务，VariDP 原版代码训练和评估跑通。
- wsl（无 NVIDIA Vulkan，lavapipe）：本仓库从零 `setup.sh` + `check_env.sh` 通过。与 ubuntu 生成的数据相比，同一种子的示范长度、转换成败相同，数值只有浮点级差异（PickCube ≤ 1e-4，PlugCharger ≤ 0.03），图像平均差 0.3 个灰度级。**最终数据应全部在同一种机器上生成。**
- HKU 集群 GPU 节点（RTX 4080 SUPER，4 CPU）：`setup.sh` + `check_env.sh` 通过，渲染每步只多 1.0 ms（单相机）/ 1.7 ms（双相机）。
- 渲染耗时（每步）：lavapipe 1 路相机 14 ms、2 路 63 ms；NVIDIA 1080 Ti 2–4 ms。只有 lavapipe 时的单任务估算：4 维单相机任务不到 1 小时，StackCube 约 1–2 小时，PegInsertionSide 约 1.5 小时，PlugCharger 约 2.5 小时。

`scripts/check_rgb_obs.py`（可选）在训练机上比较导出文件的首帧和评估环境 reset 出来的观测。

## 来源

代码与流程来自 [dp-manip](https://github.com/hollinsStuart/dp-manip)，9.25 在 ubuntu 上验证过。注释和本文中的 final-plan 指 dp-manip 的 `docs/final-plan.md`，`docs/0925-smoke.md` 是那次验证的完整记录。
