# PegInsertionSide 完整 state 诊断对照

该实验用于检查加入物体位姿等完整状态信息后，策略能否改善 RGB 基线的对孔失败。
正式代码通过 `task.obs_mode="state"` 显式选择观测；默认仍为 RGB。

## 固定设置

- `configs/experiments/state_n100.toml`：N=100、UNet、训练 seed=1，仅一个 run。
- 同一专家池按示范 seed 升序取前 100 条，验证示范取 50 条。
- 读取 `traj_i/obs` 完整 **43 维 state** 和 **8 维 actions**；不加载 RGB，也不创建视觉编码器。
- 控制模式 `pd_joint_pos`、物理后端 `physx_cpu`，评估上限 300 步。
- 保留 baseline 的 100k optimizer steps、batch 64、obs/act/pred horizon 2/8/16、100 步 DDPM、AdamW 与 EMA。只比较观测可用性，不同时改变训练预算或 UNet 宽度。
- state z-score 与 action min/max 只从选中的 100 条训练示范计算。沿用共享接口的 `proprio` / `proprio_mean` / `proprio_std` 字段名；state 模式下它们表示完整 state，checkpoint 中的 `obs_mode="state"`、`observation_key="obs"` 和维数记录其含义。

现有导出文件已包含完整 state，不必重新生成或复制示范。
预计 checkpoint 长期占用约 1.98 GiB，单 GPU 运行建议预留 4 GiB（含覆盖写入临时文件与日志）。
此空间预算不含视频或 failure-aware 轨迹；现有 failure-aware RGB 轨迹收集器不适用于 state 对照。

## 集群运行

先通过 Git 同步本次代码到集群仓库，再从集群仓库根目录执行：

```bash
export DATA_ROOT=$HOME/maniskill-demogen/data/dataset
export RUN_ROOT=$HOME/dp-runs-peginsertionside-state

# 检查真正使用的 100 条完整 state 与 50 条验证示范
.venv/bin/python scripts/inspect_dataset.py \
  --config configs/tasks/peginsertionside.toml \
  --data-root "$DATA_ROOT" --num-demos 100 --set task.obs_mode=state

# 确认只有一个 run
.venv/bin/python scripts/sweep.py show \
  --experiment configs/experiments/state_n100.toml --task peginsertionside

# 单卡训练，完成后自动评估 val 50 回合、test 100 回合
sbatch --export=ALL,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/state_single_gpu.sbatch
```

该脚本申请 1 GPU / 4 CPUs / 16 GiB RAM，在同一个作业中顺序训练和评估，不增加提交作业数。
训练中断通过现有 `resume.pt` 和 requeue 机制恢复；重复提交跳过已完成的训练并重新评估。
集群只允许一个已提交作业，提交前需等现有作业结束。

产物位于：

```text
$RUN_ROOT/peginsertionside_state_unet_n100_s1/
  run.json
  metrics.jsonl
  summary.json
  checkpoints/{step_010000.pt,step_030000.pt,step_060000.pt,final.pt,resume.pt}
  eval/{val_final.json,test_final.json}
```

训练日志位于 `$RUN_ROOT/logs/`，评估日志位于 `$RUN_ROOT/logs/eval/`。
需要训练 seed 闭环诊断时，可另提交现有评估入口或在 GPU 分配内执行：

```bash
.venv/bin/python scripts/eval_dp.py \
  "$RUN_ROOT/peginsertionside_state_unet_n100_s1/checkpoints/final.pt" \
  --split train --episodes 25
```

单次训练直接调用统一入口的等价命令为：

```bash
.venv/bin/python scripts/train_dp.py \
  --config configs/tasks/peginsertionside.toml \
  --experiment configs/experiments/state_n100.toml --experiment-value state \
  --data-root "$DATA_ROOT" --output-root "$RUN_ROOT"
```

此命令须在 GPU 分配内执行。旧 RGB checkpoint 缺少 `task.obs_mode` 时按历史 RGB 模式加载；
state 与 RGB 的运行目录、配置身份和 checkpoint 模型结构彼此独立。
