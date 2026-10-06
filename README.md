# 7606 proj of Group11

DASC7606C 小组项目（Track 3）：在六个 ManiSkill 任务（PickCube、StackCube、PushCube、PullCube、
PegInsertionSide、PlugCharger）上训练 **RGB-based Diffusion Policy**。**数据量**与**模型结构**
两条研究轨道都由 `dp-manip` 的同一套 RGB pipeline 运行；`VariDP` 只是 backbone 实现的 donor，
已冻结，不参与正式实验。

| 目录 | 做什么 |
| --- | --- |
| [maniskill-demogen](./maniskill-demogen) | **Demo Gen**：运动规划专家生成 RGB + state 示范（每任务训练 400 条、验证 50 条），一条命令生成一个任务，支持断点续跑与 Slurm 批处理 |
| [dp-manip](./dp-manip) | **统一 pipeline**：训练与闭环评估。数据量轨道 N = 25 / 50 / 100 / 200（嵌套子集，未饱和时加 400）；backbone 轨道 UNet / Transformer / MLP。集群上以单作业双 GPU 动态队列运行 |
| [VariDP](./VariDP) | 冻结的 donor：UNet / DP-T / MLP 三种主干已迁移进 `dp-manip/dp_manip/backbones/`，本目录只作实现对照（见 [VariDP/LEGACY.md](./VariDP/LEGACY.md)） |
| [report](./report) | **实验报告与交接**：StackCube-v1 数据量轨道结果 **N = 25/50/100/200/400 全档已实测**（[scaling 报告](./report/report_stackcube_scaling.md)，v2 补入 N=400）、N=100 B0 baseline 交付说明（[handoff](./report/handoff_lusen_n100.md)） |

- 当前实验口径（矩阵、种子、训练参数）：[dp-manip/PLAN.md](./dp-manip/PLAN.md)，以
  `dp-manip/configs/` 为准
- 原始实验设计与分工（state-based 时期起草，部分参数已被 RGB 实现取代）：
  [dp-manip/docs/final-plan.md](./dp-manip/docs/final-plan.md)
- 旧 state-based 工作流归档：[dp-manip/legacy/](./dp-manip/legacy/)

环境、数据契约与集群用法详见 [maniskill-demogen/README.md](./maniskill-demogen/README.md) 和
[dp-manip/README.md](./dp-manip/README.md)。

## Subtree 上游与同步

`maniskill-demogen/` 与 `dp-manip/` 都通过 **Git subtree** vendored 进本仓库，canonical upstream
分别是：

| 目录 | upstream | 引入方式 |
| --- | --- | --- |
| `maniskill-demogen/` | [`hollinsStuart/maniskill-demogen`](https://github.com/hollinsStuart/maniskill-demogen) | 保留完整历史（无 `--squash`） |
| `dp-manip/` | [`hollinsStuart/dp-manip`](https://github.com/hollinsStuart/dp-manip) | `--squash` |

本仓库 `hryang1130/7606C` 是**课程级集成仓库**：两个组件的代码与实验改动先在各自 upstream 完成，
再同步回这里；不要直接修改 vendored 目录（本仓库自己的集成与文档细节除外）。

Git remote 是**本地配置**，新 clone 不会继承，需要在本地加一次。以 `dp-manip` 为例：

```bash
git remote add dp-manip git@github.com:hollinsStuart/dp-manip.git
git fetch dp-manip
git switch -c integration/update-dp-manip
git subtree pull --prefix=dp-manip dp-manip main --squash
```

`maniskill-demogen` 同理（remote 名 `demogen`，`--prefix=maniskill-demogen`，**不加** `--squash`，
与首次引入方式保持一致）。

同步之后必须：在对应目录下装环境并验证（`dp-manip` 跑完整测试；`maniskill-demogen` 没有测试套件，
至少在计算节点跑通 `./check_env.sh`）、检查 `git diff`（该目录之外不得有变化），然后创建 PR；
PR 使用**普通 merge commit**，不要 squash 或 rebase。
`dp-manip` 的贡献者与署名依据见 [dp-manip/CONTRIBUTORS.md](./dp-manip/CONTRIBUTORS.md)。
