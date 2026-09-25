# `scripts/` — 数据侧一次性脚本

训练/评测的日常入口在 `train_local/`、`train/`、`tools/`；这里放"偶尔才会跑一次"的数据与批处理脚本。
每个脚本头部 docstring 都有完整用法，本文只做索引。

| 脚本 | 用途 | 需要什么 |
|---|---|---|
| `fetch_demos.py` | 从 hf-mirror 镜像下载 ManiSkill 官方演示到 `~/.maniskill/demos/<Task>-v1/` | 联网 |
| `convert_all.py` | 重放转换成训练要的 `state` + `pd_ee_delta_pos/pose`（7 维任务自动切 `pd_ee_delta_pose`） | 仿真 + pinocchio（本机 venv 已移植） |
| `gen_demos_scripted.py` | 自写端点闭环控制器 + waypoint 程序，**从零生成**演示（不依赖 mplib） | 仿真（无需渲染/Vulkan） |
| `check_datasets.py` | 数据集质检（条数 / 维度 / 成功率 / 长度 / 光滑性 / 动作范围）→ `report.md` + `report.csv` | 无（纯读 h5） |
| `fetch_ms_baseline.py` | 需要时抓 ManiSkill 官方 DP 基线代码到 `third_party/ms_dp_baseline/` | 联网 |
| `eval_all.py` | 批量评测 `runs/*/best.pt`（逐个调 `train_local/eval.py` 或 `train/eval.py`） | 仿真 |

典型顺序（新任务带来数据时）：

```bat
python scripts\fetch_demos.py StackCube-v1          :: 1) 下官方演示（或直接跳到第 2' 步自己生成）
python scripts\convert_all.py --tasks StackCube-v1  :: 2) 重放转换成 state + pd_ee_delta_pos
python scripts\gen_demos_scripted.py --env-id PickCube-v1 -n 200   :: 2') 或者自己生成
python scripts\check_datasets.py                    :: 3) 质检，报告在 train_local\runs\_reports\
python train_local\train.py --env-id StackCube-v1 --backbone unet --max-episode-steps 200   :: 4) 训练
python scripts\eval_all.py -n 50                    :: 5) 批量评测
python tools\summarize_runs.py --csv                :: 6) 汇总成报告
```

> 数据放在项目外（`~/.maniskill/demos/`），所以换机器/改目录名都不影响；
> 详见根 `README.md` 与 `train_local/README.md`。
