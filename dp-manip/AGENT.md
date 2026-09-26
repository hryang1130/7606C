# dp-manip 操作约束

## 当前权威流程

1. 只训练 `maniskill-demogen` 导出的 `obs_rgb/rgb + obs_rgb/state`，不得回退到 `traj_i/obs`。
2. 任务、控制模式和文件名以 `configs/*_rgb.toml` 与 `maniskill-demogen/tasks.py` 为准。
3. 数据量研究的嵌套子集、训练种子数、100k 固定步数和 held-out seed 段不得随结果改动。
4. 训练在集群 GPU 节点运行；训练代码本身不得 import ManiSkill。只有闭环评估依赖 ManiSkill。
5. 评估使用 `physx_cpu`，因为训练数据由该后端生成。不能把 `physx_cuda` 数字混进同一比较表。
6. `pd_joint_pos` 动作不能裁剪到 `[-1,1]`；只能在 DDPM 内归一化，送入环境前必须还原。
7. 不提交 HDF5、checkpoint、run 目录或 Slurm 日志。

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
