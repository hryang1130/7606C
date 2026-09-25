# `train/` — 学院 GPU 机器上的训练 / 评测

> 适用环境：学院机器 **单张 RTX 4080 (16 GB) + Linux**。不是集群、没有 DDP——训练就是单进程单卡。
> 环境搭建见根目录 `学院_uv安装指南.md`（uv 或原生 `python3 -m venv` 均可）。

## 目录分工（`dp/` 库 + 两套入口脚本）

**`dp/` 是唯一代码真源**，`train/` 与 `train_local/` 都只是薄入口：

- `dp/backbones.py`、`dp/dp_lib.py`、`dp/utils.py` 提供模型 / 数据集 / EMA / 画图；
  两套 `train.py` 都 `import` 同一份（不再有复制粘贴，也不再用 `importlib` 去加载另一个脚本）。
- 训练循环、checkpoint 格式、log.csv 格式两边**完全一致**，产物互通
  （农场训的 ckpt 可以拷回本机用 `train_local/eval.py` 评，反之亦然）。
- 默认值差异：`--device cuda`、`--num-workers 8`、`--out train/runs`（与本机 `train_local/runs` 分开），
  并开启 `pin_memory` + `persistent_workers` + `cudnn.benchmark`。
- `train/eval.py` 额外支持 **并行仿真**（`gym.make(..., num_envs=N)`），这是本机（Windows）做不到的。
- 三个主干（`--backbone mlp/unet/transformer`）的接口与超参在两套脚本里一致，见 `dp/backbones.py` 头注释。

## 训练

```bash
cd train
python train.py --env-id PickCube-v1 --backbone unet --seed 0
# 常用可调：--demo-frac 0.5（数据效率实验） --total-iters 30000 --batch 256
#           --max-episode-steps 200（StackCube） / 300（PegInsertionSide）
#           --unet-* / --tf-*（主干超参，默认即 DP 论文 lowdim 配置）
```

产物（`train/runs/<env>_frac<f>_<backbone>_seed<s>/`）：
`best.pt`、`last.pt`、`log.csv`、`loss_curve.png`、`train_summary.json`（与本机 `train_local/runs/` 同格式）。

## 评测（并行仿真）

```bash
python eval.py --ckpt runs/PickCube-v1_frac1.0_unet_seed0/best.pt \
    -n 50 --seed0 2000 --num-envs 32 --sim-backend gpu
```

| 参数 | 说明 |
|---|---|
| `--num-envs N` | 并行环境数；GPU 仿真建议 32~64，cpu 后端建议 1 |
| `--sim-backend auto\|gpu\|cpu` | `auto` = 有 CUDA 就用 gpu。**同一组对比实验必须同一后端** |
| `--seed0 2000` | held-out 评测 seeds `2000..2049`，与训练 seeds 永不重叠 |
| `--inference-steps K` | 覆盖 DDIM 步数（默认用 ckpt 里存的值） |

输出 JSON 与 `train_local/eval.py` 同格式，文件名带 `_gpuN` / `_cpuN` 后缀，不会覆盖本机结果。

## 两套 runs 目录，别混放

- 本机结果：`train_local/runs/`
- GPU 机器结果：`train/runs/`

汇总报告用机器无关的 `tools/summarize_runs.py`，指明扫哪套 runs 即可：

```bash
# 在农场（仓库根目录）
python tools/summarize_runs.py --runs-dir train/runs
# 或拷回本机后一起汇总
python tools/summarize_runs.py --runs-dir train/runs --out train/runs/SUMMARY_all.md --csv
```

一键流程（训练 → 评测 → SUMMARY）也可以直接驱动本目录：
`python tools/run_local.py --train-dir train --device cuda --backbone unet`。

## 与 `train_local/` 的产物互认

两边 ckpt 的 `config.model_kwargs` 里带着 `backbone` 与主干超参，所以任一侧训练、另一侧评测都能自动识别
（`eval.py` 零改动）。注意旧的 unet / transformer ckpt 已因主干升级而不兼容，需重训。

## 排错速查

| 症状 | 处理 |
|---|---|
| `num_workers` 报错 / 卡住 | 降到 `--num-workers 0` |
| `physx_cuda` 加载失败 | `--sim-backend cpu`（评测照常能跑，只是慢） |
| 显存 OOM | 降 `--batch`（256→128）；评测降 `--num-envs` |
| 结果与本机不可比 | 检查仿真后端是否一致（见铁律） |
