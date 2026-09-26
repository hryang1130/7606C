# RGB 数据量实验计划

## 固定研究问题

研究六种操作任务在 RGB 观测下需要多少条成功专家示范。唯一主自变量是训练示范数 N；
模型、optimizer steps、验证集、闭环测试种子和评估后端保持不变。

## 核心矩阵

| N | 训练种子 | 每任务次数 | 六任务次数 |
| ---: | --- | ---: | ---: |
| 25 | 1, 2, 3 | 3 | 18 |
| 50 | 1, 2, 3 | 3 | 18 |
| 100 | 1, 2, 3, 4, 5 | 5 | 30 |
| 200 | 1, 2, 3, 4, 5 | 5 | 30 |
| **合计** | | **16** | **96** |

`slurm/train_array.sbatch` 的数组索引 0–95 固定映射到这 96 组。每个 N 都取 400 条训练池的
前 N 条；更换训练 seed 不重抽数据。

## 条件 N=400

先完成 N=100/200。如果某任务的 `100 → 200` 提升在预先规定的统计检验下仍未饱和，才为该
任务增加 N=400、seed 1–5。`slurm/train_optional400_array.sbatch` 提供完整 30 项映射；提交时
用 `--array` 只选中满足条件的任务段。

## 统一训练口径

- RGB + 非特权 proprioception；共享 ResNet-18 + 条件 1D UNet。
- `To/Tp/Ta = 2/16/8`，DDPM 100 步。
- 100k optimizer steps，batch 64，AdamW 1e-4，cosine + 500 warmup，EMA。
- N 对应的归一化统计只来自该 N 条训练数据。
- 独立 50 条验证示范只计算固定噪声的去噪 loss。
- 10k、30k、60k 保存分析 checkpoint，100k 保存 `final.pt`。
- 只报告 `final.pt` 在测试 seeds 10000–10099 的结果。

## 评估口径

- 主指标：`success_once`；次指标：`success_at_end`。
- `physx_cpu`，固定推理 seed 0，同一任务/格子的 `num_envs` 保持一致。
- 每次训练报告成功数/100；每个 N 报训练种子均值、样本标准差和范围。
- 相邻 N 使用原计划的两层 bootstrap；不能根据结果临时增加种子。
- 训练 seeds 的前 25 条闭环评估只用于过拟合诊断，不是主结果。

## 阶段门槛

1. `inspect_dataset.py` 六任务全部 PASS。
2. 先跑 PickCube N=25 seed=1 的短 smoke（CLI 覆盖 total_iters），确认显存、吞吐和 resume。
3. 提交核心 96 组。
4. 对 10k/30k/60k/final 做诊断评估；主表只读取 final。
5. 判断是否触发 N=400，再提交可选数组段。
