# dp-manip 操作约束

## 当前权威流程

1. 默认 RGB 主线只读取 `maniskill-demogen` 导出的 `obs_rgb/rgb + obs_rgb/state`。显式设置 `task.obs_mode="state"` 的诊断对照读取 `traj_i/obs` 完整 state，不读取图像；不得在 RGB 模式下回退到完整 state。state 对照通过正式统一流程运行，使用独立的 `_state_` run 目录。
2. 任务、控制模式和文件名以 `configs/*_rgb.toml` 与 `maniskill-demogen/tasks.py` 为准。
3. 数据量研究的嵌套子集、训练种子数、100k 固定步数和 held-out seed 段不得随结果改动。
4. 训练在集群 GPU 节点运行；训练代码本身不得 import ManiSkill。只有闭环评估依赖 ManiSkill。
5. 评估使用 `physx_cpu`，因为训练数据由该后端生成。不能把 `physx_cuda` 数字混进同一比较表。
6. `pd_joint_pos` 动作不能裁剪到 `[-1,1]`；只能在 DDPM 内归一化，送入环境前必须还原。
7. 不提交 HDF5、checkpoint、run 目录或 Slurm 日志。
8. `legacy/` 与 `VariDP/` 是冻结的历史/donor 代码，不参与正式实验；正式代码不得 import 或执行它们（`tests/test_legacy_boundary.py` 检查）。

## 修改后的最低验证

```bash
python3 -m compileall -q dp_manip scripts
python3 - <<'PY'
from pathlib import Path
from dp_manip.config import load
for path in Path('configs').glob('*_rgb.toml'):
    load(path)
print('configs ok')
PY
```

有真实数据和集群环境时再运行：

```bash
.venv/bin/python scripts/inspect_dataset.py --data-root "$DATA_ROOT"
srun --gres=gpu:1 .venv/bin/python scripts/verify_cuda.py
```

代码只在仓库中编辑；集群端通过 Git 同步，不在计算节点热改。历史 `docs/` 中的 state-based
命令已经过时，不能据此恢复旧入口。
