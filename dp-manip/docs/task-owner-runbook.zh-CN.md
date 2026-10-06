# 任务负责人操作手册：从生成数据到 failure 实验

适用于在 HKU 集群上**从零**跑通一个任务的人（例如 LiftPegUpright 的负责人）。按顺序做完
§1–§8，得到这个任务的示范数据、数据量主线结果，以及 failure-aware 实验的结果。PlaceSphere
就是按这条流程跑的；文中给出的 PlaceSphere 实测数字可以作为参照。

方法、对照组和统计规则见 [failure-aware 方案](failure_aware_finetuning_plan.md)，这里只写操作。

## 0. 约定

- **QOS 每个用户同时只能有 1 个作业**（排队中的也算）。下面每一步都是一个作业，前一个结束
  才能提交下一个。`squeue -u $USER` 为空时才能提交。
- debug 分区单个作业最长 18 小时。训练、评估、failure 流程被时限或抢占打断后都会自动
  requeue，从断点继续，不需要手动处理。
- 家目录配额 100 GB。一个任务的全部产物大约 35–45 GB（见各步），开始前先 `df -h ~`。
- 下文统一用这些变量（每次登录后先设置）：

```bash
export TASK=liftpegupright                               # 你的任务
export DATA_ROOT=$HOME/maniskill-demogen/data/dataset    # 示范数据
export RUN_ROOT=$HOME/dp-runs-$TASK                      # 训练与评估产物
```

## 1. 环境（只做一次，登录节点）

```bash
export PATH="$HOME/.local/bin:$PATH"     # 没有 uv 先装：curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hollinsStuart/maniskill-demogen.git
cd ~/maniskill-demogen && ./setup.sh
git clone -b exp/failure-aware-dp https://github.com/hollinsStuart/dp-manip.git ~/dp-manip
cd ~/dp-manip && ./setup.sh
```

dp-manip 目前要用 `exp/failure-aware-dp` 分支：PlaceSphere/LiftPegUpright 的配置、回合长度
检查和 failure 流程都在这个分支上。合进 main 之后改用 main。之后每次开始前 `git pull`。

## 2. 生成示范（demogen，约 1–2 小时，2–4 GB）

```bash
cd ~/maniskill-demogen
sbatch --export=ALL,TASK=$TASK slurm/generate_task.sbatch
```

结束后看 `data/logs/$TASK.out` 末尾，要有 `train 400 demos` 和 `val 50 demos`。

**回放成功的示范不够时**，导出会报 `only N usable demos, need 400`。`tasks.py` 里的默认条数
（LiftPegUpright 为 440 / 55）还没在真实生成中验证过。PlaceSphere 用默认值时只有 68% 的
专家示范回放成功，只能导出 298 / 37 条，改成 `--n-train 700 --n-val 120` 才够。处理方法：

1. 在日志里找 `usable X/Y = Z%`，得到回放成功率 Z；
2. 删掉 `data/work/<Env>/motionplanning/` 下对应 split 的文件；
3. 用 `--n-train ≈ 400 / Z × 1.1`、`--n-val ≈ 50 / Z × 1.1` 重新生成：

```bash
sbatch --export=ALL,TASK=$TASK,ARGS="--n-train 700 --n-val 120" slurm/generate_task.sbatch
```

同一行日志里的 `len a-b-c` 是示范长度的最小值、中位数、最大值，下一步要用。

## 3. 确定评估回合长度（最容易出错的一步，登录节点）

每个任务配置里的 `max_episode_steps` 是**闭环评估的回合上限**。训练本身不用它，但 checkpoint
会把它记下来，之后所有评估都在这一步截断。ManiSkill 的注册默认值往往比专家示范还短：
PlaceSphere 的默认值是 50，而示范有 90–150 步，结果 16 个 run 训练全部正常，评估却全是 0，
而且没有任何报错。

规则（`docs/final-plan.md` §1）：

```text
max_episode_steps = max(注册默认值, 2 × 示范平均长度)，向上取整到 50 的倍数
```

用检查脚本读出示范长度，并确认配置可用：

```bash
cd ~/dp-manip
.venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT" --config configs/tasks/$TASK.toml
```

- 输出 `PASS ...; demo length 最小-平均-最大 (min-mean-max), max_episode_steps N`：按上面的
  规则检查 N。不对就改 `configs/tasks/$TASK.toml` 的 `max_episode_steps`，再跑一遍。
- 输出 `FAIL ... shorter than the longest demonstration ... e.g. M`：回合上限比最长的示范还短，
  按提示把 `max_episode_steps` 改成 M（或按规则算出的值）。训练启动时也会做同样的检查并报错。

改完 commit 并 push 到 dp-manip（只改自己任务的配置文件），在 commit 信息里写上示范长度和
算出的值。**之后不要再改**：训练好的 checkpoint 都会记录这个值。

## 4. 提交前检查（登录节点）

```bash
cd ~/dp-manip
.venv/bin/python scripts/check_experiment.py --experiment data_size     # 期望 Gate B ok
.venv/bin/python scripts/sweep.py show --experiment configs/experiments/data_size.toml --task $TASK
```

`sweep.py show` 应列出 16 个 run：N=25/50 各 3 个种子，N=100/200 各 5 个种子。

## 5. 训练数据量主线（约 14 小时、约 29 GPU-h、约 20 GB）

```bash
cd ~/dp-manip
sbatch --export=ALL,TASK=$TASK,EXPERIMENT=configs/experiments/data_size.toml,DATA_ROOT="$DATA_ROOT",RUN_ROOT="$RUN_ROOT" \
  slurm/train_dual_gpu.sbatch
```

一个作业里两张 GPU 轮流训练 16 个 run，每个 100k 步。PlaceSphere 每个 run 约 1.8 小时，
全部跑完约 14 小时。进度：`tail -f slurm-dp-rgb-train-dual-<jobid>.out`，单个 run 的日志在
`$RUN_ROOT/logs/<run>.log`。最后一行是 `completed: 16 ... failed: 0` 才算完成。

## 6. 评估主线（test 约 1 小时，val 约 0.5 小时）

两次分开提交（一次只能一个作业）：

```bash
cd ~/dp-manip
sbatch --export=ALL,TASK=$TASK,EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT="$RUN_ROOT",SPLIT=test \
  slurm/eval_dual_gpu.sbatch
# 上一个结束后：
sbatch --export=ALL,TASK=$TASK,EXPERIMENT=configs/experiments/data_size.toml,RUN_ROOT="$RUN_ROOT",SPLIT=val \
  slurm/eval_dual_gpu.sbatch
```

结果在每个 run 的 `eval/test_final.json`（100 回合，seed 10000–10099）和
`eval/val_final.json`（50 回合，seed 5000–5049），成功率在 `summary.success_once`。

如果训练时的回合上限错了（例如 PlaceSphere 在 2026-10-03 之前训练的 run），不要重新训练，
评估时加 `MAX_EPISODE_STEPS=<正确值>`：结果文件变成 `test_final_h<值>.json` /
`val_final_h<值>.json`，原来的结果不会被覆盖。之后 failure 流程也要加
`--max-episode-steps <值>`（§7、§8）。

**能不能做 failure 实验**，看 val 结果：N=100 五个种子的平均成功率在 [0.15, 0.85) 之间就可以；
否则看 N=200。两个都不在这个区间，failure 实验就不做，报告里写明原因（§8 的
`--no-low-success` 会自动在这里停下）。test 结果只用于报告，不用于做这个判断。

## 7. failure 流程 smoke（不到 1 GPU-h）

正式实验前先用缩小版协议把整条流程在真实环境里跑一遍。它用单独的锁定文件和输出目录，
结果直接丢弃，不会影响正式实验：

```bash
cd ~/dp-manip
export SMOKE=$HOME/dp-failure-smoke-$TASK
sbatch slurm/failure_aware.sbatch scripts/failure_pipeline.py --task $TASK --run-root "$RUN_ROOT" \
  --protocol configs/failure_aware/smoke_protocol.toml --lock "$SMOKE/$TASK.toml" --rollout-root "$SMOKE" \
  --no-low-success --budget-confirmed --include-test
```

（如果 §6 用了 `MAX_EPISODE_STEPS`，这里加 `--max-episode-steps <值>`。）

通过的标准：作业日志 `slurm-dp-failure-aware-<jobid>.out` 最后是 `DONE: every test result is written.`，
中间有 `alpha = 0 reproduces the baseline`。

smoke 每类只有 4 条数据，offline gate 很可能不通过，日志出现
`STOPPED: the offline gate failed`，后面的 dry-run、调参、测试就没有跑到。**只在 smoke 里**，
把临时锁定文件里的 gate 改成通过，再提交同一条命令，让剩下的步骤也跑一遍：

```bash
cd ~/dp-manip && .venv/bin/python - "$SMOKE/$TASK.toml" <<'PY'
import sys, tomllib
from pathlib import Path
from dp_manip.failure_lock import dumps
path = Path(sys.argv[1]); data = tomllib.loads(path.read_text())
data["gate"]["passed"] = True; path.write_text(dumps(data))
PY
```

正式实验（§8）绝对不能这样做。smoke 通过后 `rm -rf "$SMOKE"`。

## 8. 正式 failure 实验（约 10–12 GPU-h、约 8–11 GB）

```bash
cd ~/dp-manip
sbatch slurm/failure_aware.sbatch scripts/failure_pipeline.py --task $TASK --run-root "$RUN_ROOT" --no-low-success
```

（同样，§6 用了 `MAX_EPISODE_STEPS` 的话加 `--max-episode-steps <值>`。）

`--no-low-success` 只给作为重复验证的任务用（LiftPegUpright）。PlaceSphere 是主任务，按方案
§1 允许低成功模式，不加这个参数。

流程会自动依次执行：选档 → 采集 checkpoint 1 的 rollout → **暂停①** → pilot 微调 → 采集
checkpoint 2、3 → 其余微调 → offline gate → dry-run → 调 α → **暂停②** → 测试。每次暂停后，
**重新提交同一条命令并加上对应参数**就会接着往下跑，已完成的步骤不会重做：

| 日志里看到 | 含义 | 下一步 |
| --- | --- | --- |
| `PAUSED (budget ...)` 和 `projected total ... GPU-h` | 实测的 rollout 成本和总预算估计 | 估计不超过 24 GPU-h 就加 `--budget-confirmed` 重新提交；超过的话按方案 §10.3 的顺序缩减，并记进方案 §11 |
| `STOPPED: the offline gate failed` | 失败模型没学到区分失败的信息 | 实验到此结束，作为负结果写进报告（方案 §6.7），不要调参重跑 |
| `PAUSED (test): the design is locked` | α、模型等全部锁定 | 把 `configs/failure_aware/$TASK.toml` commit 并 push，再加 `--budget-confirmed --include-test` 重新提交 |
| `DONE: every test result is written.` | 完成 | 见下 |
| `preempted ...; resubmit` 或作业被 requeue | 被时限或抢占打断 | 不用管，会自动 requeue；如果没有，重新提交同一条命令 |

任何时候都可以查看进度和已用的 GPU 时间：

```bash
.venv/bin/python scripts/failure_study.py status --task $TASK
```

结果文件在 `$RUN_ROOT/failure_aware/$TASK/s<种子>/eval/test/*.json`，每个（组别, checkpoint）
一个文件。锁定文件 `configs/failure_aware/$TASK.toml` 记录了所有选择，报告里的每个数字都
应该能对上它。

## 9. 常见问题

| 现象 | 处理 |
| --- | --- |
| `sbatch: ... QOSMaxSubmitJobPerUserLimit` | 已有一个作业在跑或排队，等它结束 |
| 训练报 `shorter than the longest demonstration` | 回到 §3 改 `max_episode_steps` |
| 评估成功率全是 0 | 先看 `eval/*.json` 里的 `max_episode_steps` 是否比示范长；再看日志里有没有回合全部跑满 |
| `select-cell` 报 `missing` | 缺 val 结果，先做 §6 的 val 评估；用了 `MAX_EPISODE_STEPS` 的话 failure 流程要加同样的 `--max-episode-steps` |
| `select-cell` 报 `study is not run on this task` | baseline 不在 [0.15, 0.85)，按方案 §12.3 不做 failure 实验，写进报告 |
| `record-collection` 报 `collected at ... steps` | 采集时的回合长度和锁定的不一致；不要单独手动跑 `collect_rollouts.py`，用 §8 的流程命令 |
| `alpha = 0 guidance did not reproduce the baseline` | GPU 计算有非确定性，配对比较不成立；停下来，联系框架维护者 |
| 磁盘满 | `du -sh $RUN_ROOT/*`；`step_*.pt` 中间 checkpoint 每个约 310 MB，过拟合分析用完可以删，`final.pt` 不能删 |
