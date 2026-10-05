# dp-manip：集群 RGB Diffusion Policy

本目录是六个 ManiSkill 任务的 **RGB-based Diffusion Policy** 训练与评估工程。数据由
`maniskill-demogen` 生成；训练/评估面向 Linux GPU 集群。当前 QOS 每用户只允许 **1 个已提交
作业**，单作业最多 2 张 GPU，因此生产入口是**一个双 GPU 作业内的动态队列**（两个 worker
各绑定一张 GPU，先完成的 worker 立即领取下一个 run），不再使用 Job Array。

保留的研究问题是数据量：对每个任务使用同一训练池的嵌套子集
`25 ⊂ 50 ⊂ 100 ⊂ 200`，只有在 `100 → 200` 仍未饱和时才增加 `400`。子集固定取
按示范 seed 升序排序后的前 N 条，与 HDF5 导出顺序和 `episode_id` 无关；因此不同训练
种子看到完全相同的示范，种子间方差只反映优化随机性。

## 六个任务

| 配置 | 环境 | 控制模式 | 动作维 | 评估步数 |
| --- | --- | --- | ---: | ---: |
| `tasks/pickcube.toml` | PickCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `tasks/stackcube.toml` | StackCube-v1 | `pd_ee_delta_pos` | 4 | 200 |
| `tasks/pushcube.toml` | PushCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `tasks/pullcube.toml` | PullCube-v1 | `pd_ee_delta_pos` | 4 | 100 |
| `tasks/peginsertionside.toml` | PegInsertionSide-v1 | `pd_joint_pos` | 8 | 300 |
| `tasks/plugcharger.toml` | PlugCharger-v1 | `pd_joint_pos` | 8 | 200 |

PegInsertionSide 与 PlugCharger 使用 `pd_joint_pos`，与 `maniskill-demogen/tasks.py` 的最终数据一致。
绝对关节目标会先按训练子集做 min-max 归一化，执行时还原，不裁剪到 `[-1, 1]`。

## 数据契约

每个 split 使用 `maniskill-demogen/data/dataset/` 下的文件：

```text
{train,val}/<Env>/motionplanning/trajectory.state.<control>.physx_cpu.h5
```

训练只读取：

```text
traj_i/obs_rgb/rgb    uint8   (T+1, 128, 128, 3*C)
traj_i/obs_rgb/state  float32 (T+1, P)
traj_i/actions        float32 (T, A)
```

`traj_i/obs` 是特权 state，RGB 策略不会读取。文件名里的 `.state.` 是为了兼容 ManiSkill 官方
示范布局，不能据此判断训练观测类型。

训练文件固定 400 条（种子池 `<4000`），验证示范固定 50 条（种子 `4000–4999`）；闭环验证
使用 `5000–5049`，正式测试使用 `10000–10099`。

`data.num_demos=N` 的选样规则是按 `episode_seed` 升序取前 N 条：同一数据池上
`N1 < N2` 必有 `seeds(N1) ⊂ seeds(N2)`。同一 split 内 `episode_id` 和 `episode_seed`
都必须唯一，重复会直接使数据检查失败。实际选中的 seed 记录在 `run.json` 的
`data_selection` 字段中；`scripts/inspect_dataset.py` 会在提交作业前校验该嵌套不变量。

## 模型与公平性

- 每个相机的 3 通道图像从 HDF5 的通道拼接中拆出；默认共享一套 GroupNorm ResNet-18。
- 每帧视觉特征与 `obs_rgb/state` 的非特权 proprioception 由同一个 observation encoder
  编码成 `(B, To, Dobs)`；`policy.backbone` 选择 noise predictor，目前有 canonical
  `unet`（只在 FiLM 边界把序列 flatten 成 `(B, To*Dobs)`）、从 VariDP 迁移的
  `transformer`（把 `(B, To, Dobs)` 保留为条件 token）和 `mlp`（flatten 后先经该 arm 的 observation MLP `To*Dobs → 256 → 256`，再作为全局条件）。
- 动作预测/执行 horizon 为 `16/8`，DDPM 训练和推理均为 100 步。
- 所有任务、N 和训练种子固定 100k optimizer steps；RGB batch 默认为 64。
- proprio z-score 与 action min/max **只用当前 N 条训练示范**计算。
- 验证去噪 loss 使用独立的 50 条验证示范；主结果只用 `final.pt`，不按 loss 挑 checkpoint。
- 闭环评估固定 `physx_cpu`，与数据生成后端一致；策略推理仍在 CUDA 上。

## 集群快速开始

```bash
# 0. 登录节点：本集群没有 /scratch，数据与输出都放在 $HOME
export DATA_ROOT=$HOME/maniskill-demogen/data/dataset
export RUN_ROOT=$HOME/dp-runs

# 1. 登录节点建环境
./setup.sh

# 2. 在提交作业前检查六套数据（DATA_ROOT 指向 demogen 的 data/dataset）
.venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT"

# 3. Gate B：确认每个实验矩阵的 resolved config 只在声明的实验变量上不同
.venv/bin/python scripts/check_experiment.py --experiment data_size
.venv/bin/python scripts/check_experiment.py --experiment backbone

# 4. 查看运行清单（可选：run name / run 数，不需要再计算数组下标）
.venv/bin/python scripts/sweep.py show --experiment configs/experiments/data_size.toml --task peginsertionside

# 5. 提交一个 task 的数据量实验：Slurm 中只有 1 个作业，作业内 2 张 GPU
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT=$RUN_ROOT \
  slurm/train_dual_gpu.sbatch

# 6. 评估同一 task（completed run 才有 checkpoint；缺 checkpoint 的 run 会在 preflight 明确失败）
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT=$RUN_ROOT \
  slurm/eval_dual_gpu.sbatch
# 训练曲线诊断：CHECKPOINT=step_060000.pt SPLIT=val；SPLIT=train 走所有嵌套子集共有的前 25 个训练 seed
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT=$RUN_ROOT,CHECKPOINT=step_060000.pt,SPLIT=val \
  slurm/eval_dual_gpu.sbatch

# 7. 轨道 B（backbone）：只切换 EXPERIMENT，RUN_ROOT 必须与第 5 步相同
#    unet 格子就是 data-size 的 N=100 格子（同名同目录）：已有 final.pt 时直接跳过并复用。
sbatch --export=ALL,TASK=peginsertionside,EXPERIMENT=configs/experiments/backbone.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT=$RUN_ROOT \
  slurm/train_dual_gpu.sbatch
```

重复提交同一条命令是安全的：`final.pt` 与当前 config 一致的 run 记为 `skipped`，只有
`resume.pt` 的中断 run 通过 `--resume auto` 继续。`RUN_ROOT` 必须在数据量实验与 backbone
实验之间保持一致，否则 N=100 UNet 的结果会被当成另一份运行目录重新训练。

### 运行中查看

```bash
squeue -u "$USER"                                  # 应只有 1 个作业，ID 无 _<array index> 后缀
tail -f slurm-dp-rgb-train-dual-<jobid>.out        # 调度事件 + completed/skipped/failed/interrupted 汇总
tail -f $RUN_ROOT/logs/<run name>.log        # 单个训练 run
tail -f $RUN_ROOT/logs/eval/<run name>.log   # 单个评估 run
```

`[worker 0]` / `[worker 1]` 分别只看到物理 GPU 0/1（子进程内即逻辑 `cuda:0`）。
walltime 前 120s 的 `USR1`（`--signal=B:USR1@120`）或手动
`scancel --signal=USR1 --batch "$JOBID"` 会让队列停止派发新 run、把信号转发给两个 trainer、
等 `resume.pt` 写完后以 75 退出；`train_dual_gpu.sbatch` 随后 `scontrol requeue`，下一次分配
跳过已完成 run、续训中断 run。普通 `scancel "$JOBID"`（不带 `--signal`）会直接取消作业，
不会走 checkpoint → requeue 流程。

### 故障查看

- 顶层 `.out` 出现 `failed <run> exit=N log=<path>`：打开该 run 日志看 traceback。
- 同名目录里已有其他 config 的 `final.pt`：作业会把它记为 `failed`/`conflict` 并最终非零退出；
  提交前可用只读的 `scripts/sweep.py plan --experiment ... --task ... --output-root ...`
  查看 `completed/pending/conflict`，不会启动训练。
- 评估缺 checkpoint：`eval_queue` preflight 打印 `missing checkpoint <path>`，计入汇总 `failed`，
  作业非零退出。
- 双 trainer 并行验证与抢占演练见 `docs/cluster-smoke-test.zh-CN.md`。

统一实验入口（所有实验共用同一个 trainer，只换 config）：

```bash
.venv/bin/python scripts/run_experiment.py \
  --task pickcube --experiment data_size --value 50 --seed 1 \
  --data-root "$DATA_ROOT"

.venv/bin/python scripts/run_experiment.py \
  --task pickcube --experiment backbone --value transformer --seed 1 \
  --data-root "$DATA_ROOT"
```

`--task` / `--experiment` 接受 `configs/tasks`、`configs/experiments` 下的短名或显式路径；
`--value` 按 experiment spec 声明的类型解析（整数 N 或 backbone 名），`--seed`、`--num-demos`
（难任务轨道 B 的 N_B）、`--data-root` 是运行时覆盖。入口本身没有按 experiment 名称的分支，
新实验只需新增 config。

`slurm/train_dual_gpu.sbatch` → `scripts/train_queue.py`：每个 run 的命令由
`dp_manip/runlist.py::train_command` 构造（与 `scripts/sweep.py` 共用同一函数）。单运行入口
`scripts/train_dp.py` / `scripts/run_experiment.py` 与队列最终都调用
`dp_manip/trainer.py::run_training`，所以不会出现第二套 trainer：

```bash
.venv/bin/python scripts/train_dp.py \
  --config configs/tasks/pickcube.toml \
  --experiment configs/experiments/data_size.toml --experiment-value 25 \
  --data-root "$DATA_ROOT" --seed 1
```

作业收到 Slurm 的 `USR1`/`TERM` 后，队列把信号转发给两个 trainer，各自写
`checkpoints/resume.pt` 并以状态 75 退出；`train_dual_gpu.sbatch` 等队列 drain 完成后
requeue，下一次分配自动续训。已存在 `final.pt` 的 run 记为 `skipped`，不会重复训练。

旧入口兼容状态：`slurm/train_array.sbatch`、`slurm/eval_array.sbatch` 和
`scripts/sweep.py train/eval --index` 仍然保留，且与队列共用同一套 `train_command` /
`eval_command`；但多元素 Job Array 在当前 QOS 下会触发 `QOSMaxSubmitJobPerUserLimit`，
不能用作生产流程（单元素 `--index` 调试仍可用）。正式训练/评估使用 `*_dual_gpu.sbatch`，
`scripts/train_dp.py` 与 `scripts/run_experiment.py` 单运行入口不变。

`resume.pt` 除 model/optimizer/scheduler/EMA/scaler/step 外还保存 Python、NumPy、
torch CPU 与 CUDA RNG state；训练 batch 由 `(seed, step)` 直接导出，因此 `resume` 后
第 k 步的 batch 和噪声与连续训练的第 k 步一致，被抢占次数不影响随机轨迹。
Phase 5 之前的旧 `resume.pt` 没有 `rng` 字段，仍可续训，但会打印一次 trajectory
可能偏移的 warning。

## 产物

```text
runs/<task>_rgb_<backbone>_n<N>_s<seed>/   # backbone 目前为 unet / transformer / mlp
  run.json
  metrics.jsonl
  summary.json
  checkpoints/
    step_010000.pt
    step_030000.pt
    step_060000.pt
    final.pt
    resume.pt
  eval/
    test_final.json
```

中间 checkpoint 用于过拟合/训练进程分析；正式表格只使用 `final.pt`。

每个 run 的日志写在 `RUN_ROOT/logs/` 下：训练为 `logs/<run name>.log`，评估为
`logs/eval/<run name>.log`（stdout+stderr 合并）。顶层 Slurm `.out` 只保留调度事件和
`completed/skipped/failed/interrupted` 汇总。

`run.json` 保存完整的 resolved config、实际选中的示范 seed、归一化统计，以及 Phase 14
（§19）的元数据：`experiment_context`（实验名 / variable / value / seed / `control_hash`）、
`git`（commit / branch / dirty）和 train/val 数据集的 `fingerprint`。`control_hash` 由所有非
实验变量的值生成，同一 experiment matrix 的同一任务下必须相同；`summary.json` 同样记录
`control_hash` 和实际训练时长。`train_data`/`val_data`（包括 checkpoint 里的副本）也带回
fingerprint，所以一个 checkpoint 能追溯到具体的数据文件。

## 汇报视频（1080p 成功 / 失败 rollout）

`scripts/record_rollout_videos.py` 按 `eval_dp.py` 的方式（同样的种子顺序、`num_envs` 和推理种子）
重跑一个 checkpoint，从 ManiSkill 的展示相机 `render_camera` 录 1920×1080 的 H.264 MP4（`-crf 16`，
yuv420p，PowerPoint 可直接播放），保留前 `--success` 个成功和前 `--failure` 个失败回合（默认各 3 个），
凑够后提前停止。成败按 `success_once` 判定。策略仍然只看 128×128 的传感器相机；展示相机只多加
`render_mode` 和 `human_render_camera_configs`，不改物理和观测。

脚本默认读取同一 run 的已保存评估 `eval/<split>_<checkpoint>[_h<N>].json`，复用它的 `num_envs` 和
回合步数，并逐回合核对 `success_once`，所以视频就是成功率背后的那些回合；不一致会写进
`videos.json` 并以非零状态退出。有多个评估文件（例如 PlaceSphere 的 50 步和 200 步）时脚本会停下，
需要用 `--reference` 指定。

```bash
# 单个 checkpoint（GPU 节点上）
.venv/bin/python scripts/record_rollout_videos.py "$RUN_ROOT/pickcube_rgb_unet_n100_s1/checkpoints/final.pt"
# PlaceSphere：指定 200 步的评估
.venv/bin/python scripts/record_rollout_videos.py "$RUN_ROOT/placesphere_rgb_unet_n100_s1/checkpoints/final.pt" \
  --reference "$RUN_ROOT/placesphere_rgb_unet_n100_s1/eval/test_final_h200.json"
# 集群：一个单 GPU 作业依次录多个 checkpoint，遇到第一个失败就停
sbatch --export=ALL,CHECKPOINTS="<final.pt> <final.pt>" slurm/record_videos.sbatch
```

输出在 `<run>/videos/<split>_<checkpoint>[_h<N>]/`：`<task>_<split>_seed<seed>_<success|failure>.mp4`
和记录参数、逐回合结果的 `videos.json`。`--shader rt-fast` 换光线追踪（更慢），`--hold-seconds`
控制结尾定格时长（默认 1 秒），`--fps` 默认等于任务控制频率（实时速度）。

## 代码导航

- `dp_manip/data.py`：demogen schema 校验、流式统计、temporal windows；默认启动前把所选 episode
  一次性解码进内存，`data.preload=false` 时回到 HDF5 懒加载（样本逐位相同，见 `configs/README.md`）。
- `dp_manip/vision.py`：不依赖 torchvision 的 GroupNorm ResNet-18 与随机平移增强。
- `dp_manip/observation_encoder.py`：共享 RGB + proprio observation encoder，固定输出 `(B, To, Dobs)`。
- `dp_manip/backbones/`：`NoisePredictor` 接口、`policy.backbone` 注册表、包装 canonical UNet
  的 `UNetBackbone` 与从 VariDP 迁移的 `TransformerBackbone`、`MLPBackbone`。
- `dp_manip/policy.py`：动作归一化、DDPM；把 `(B, To, Dobs)` 序列原样交给 noise predictor；`DiffusionPolicy.from_checkpoint` 是评测与测试共用的 checkpoint 装载口。
- `dp_manip/trainer.py`：唯一训练 pipeline（resume、采样器、日志、checkpoint、评估 loss）。
- `dp_manip/invariants.py`：Gate B 声明差异、config diff/prune 与 `control_hash`（checker 和 run 元数据共用）。
- `dp_manip/metadata.py`：run 元数据辅助（git revision、dataset fingerprint）。
- `scripts/run_experiment.py`：统一实验入口 `--task/--experiment/--value/--seed`，只解析 config。
- `scripts/check_experiment.py`：Gate B checker：对比实验矩阵的 resolved config（`--run-root` 时对比实际 `run.json`），输出 `control_hash`。
- `scripts/train_dp.py`：单运行训练 CLI，与统一入口共用 `dp_manip.trainer`。
- `scripts/eval_dp.py`：固定种子 RGB 闭环评估。
- `scripts/record_rollout_videos.py` / `dp_manip/rollout_video.py` / `slurm/record_videos.sbatch`：1080p 成功/失败 rollout 视频，逐回合核对已保存的评估。
- `scripts/sweep.py`：show/plan/index 的薄 CLI；`slurm/train_array.sbatch` 与
  `slurm/eval_array.sbatch` 是旧 Job Array 入口，仅保留兼容与本地调试，当前 QOS 下不可生产使用。
- `dp_manip/completion.py`：`final.pt`/`run.json` 完成状态判断（trainer 与调度器共用）。
- `dp_manip/runlist.py`：运行清单、task 过滤、`train_command`/`eval_command` 与 `plan_runs`。
- `dp_manip/scheduler.py`：双 worker 动态队列、抢占信号 drain、per-run 日志与失败汇总。
- `scripts/train_queue.py` / `scripts/eval_queue.py`：单作业双 GPU 的训练/评估队列入口。
- `slurm/train_dual_gpu.sbatch` / `slurm/eval_dual_gpu.sbatch`：当前 QOS 下推荐的生产入口。
- `docs/cluster-smoke-test.zh-CN.md`：提交正式 sweep 前的集群 smoke 检查表。
- `slurm/bench_dataload.sbatch` / `scripts/dataload_bench.py`：单 GPU 上 lazy 与 `data.preload` 的
  DataLoader 吞吐对比（交替运行、测量窗口内的 steps/s、GPU/CPU 利用率、内存），输出在 `bench/`。
- Failure-aware 研究（`exp/failure-aware-dp`，计划见 `docs/failure_aware_finetuning_plan.md`）：
  `scripts/collect_rollouts.py`（rollout 收集与数据集 build）、`scripts/finetune_dp.py`（从 baseline
  checkpoint 微调）、`scripts/failure_study.py`（逐阶段 CLI 与 lock 文件）、
  `scripts/failure_pipeline.py`（可续跑的总驱动，含 smoke），均通过 `slurm/failure_aware.sbatch` 提交。
- `dp_manip/training.py`：EMA、RNG state 存取、`(seed, step)` 确定性 sampler、resume checkpoint 组装。
- `baselines/phase0/pickcube_rgb.json`：重构前 RGB baseline 的机器可读 regression reference。
- `legacy/`：Phase 16 归档的 state-based 工作流（含 `run_cpu.py` 与旧 WSL/Ubuntu 清单）；
  `tests/test_legacy_boundary.py` 保证正式代码不引用它或 VariDP。
- `docs/phase0-rgb-baseline.md`：Phase 0 行为清单、真实 smoke 结果和复现命令。
- `PLAN.md`：数据量实验矩阵和运行口径。

旧的本地 state-based 工作流已归档到 `legacy/`（运行记录、旧操作说明、示范生成脚本和旧环境清单），
只作历史/调试参考，正式代码不得引用；`VariDP/` 是 backbone 的 frozen donor，canonical 实现已在
`dp_manip/backbones/`。本 README、`PLAN.md` 和 `configs/baseline.toml`、`configs/tasks/` 与
`configs/experiments/` 是当前权威定义；`configs/*_rgb.toml` 仅为旧命令保留兼容跳转。

## 上游与署名 / Upstream and attribution

dp-manip originated as Holly Stewart's standalone repository and was subsequently
developed as part of the DASC7606C group project. The unified implementation includes
contributions from multiple project members, with substantial integration and refactoring
work performed by Holly Stewart. This standalone repository is now the canonical upstream
for the dp-manip component; hryang1130/7606C remains the course-level integration repository.

- 本仓库是 `dp-manip` 组件的 canonical upstream：改动先落在这里，再由课程集成仓库
  `hryang1130/7606C` 以 git subtree（`--squash`）同步到 `dp-manip/`。
- 数据生成工具 `maniskill-demogen`、backbone donor `VariDP` 以及集群侧的运行记录保留在课程
  集成仓库，不在本仓库内。
- 贡献者与署名依据见 [`CONTRIBUTORS.md`](./CONTRIBUTORS.md)。
