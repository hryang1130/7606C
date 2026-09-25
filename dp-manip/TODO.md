# TODO（未完成事项）

**更新时间：9.23** ｜ 当前状态见 [STATUS.md](./STATUS.md) ｜ 课程要求见 [docs/requirements.md](./docs/requirements.md)

> 六个任务：PickCube、PushCube、PullCube、StackCube、LiftPegUpright、PegInsertionSide（PickCube 已完成环境与 10 条示范验收，并已重放出 state 数据）。
>
> 本文件由 Mac 原 `TODO.md` 与 wsl 原 `TODO.md` 合并（9.23）。训练设备问题已经解决：训练与评估放在 wsl（RTX 4090），ubuntu 的 GTX 1080 Ti 不再用于训练。

---

## A. 多设备工作流（9.23 完成，见 [PLAN.md](./PLAN.md)）

- [x] 阶段 0：ubuntu / wsl 打包备份 `~/dp-manip-pre-git-20260923.tgz`。
- [x] 阶段 1：Mac 建仓，写 `.gitignore`，添加 `ubuntu` / `wsl` 两个 remote。
- [x] 阶段 2：导入代码与文档，数据 rsync 到 Mac，生成 `manifests/*.sha256` 并校验，合并中文文档，首次提交。
- [x] 阶段 3：挂接 ubuntu / wsl；两端 HEAD 与 Mac 一致，数据与 `.venv` 未变。
- [x] 阶段 4：编写 `scripts/sync.sh`（`push` / `fetch` / `pull-results` / `data` / `manifest` / `status`）。
- [x] 阶段 5：往返验证（Mac → 远端、wsl → Mac → ubuntu、远端有改动时拒绝 push）。

## B. DP 接入准备（P1，9.23 完成：链路已跑通）

- [ ] 在 `pd_joint_pos` 8 维与 `pd_ee_delta_pos` 4 维之间正式选定控制模式。（9.23：跑通链路先用现有 `pd_joint_pos` 数据，靠动作归一化处理范围；`pd_ee_delta_pos` 之后转换，可作为「动作表示」对比。）官方 IL 示例采用后者，但目前只完成了比较分析；若选后者，先转换一小批并验证成功率，再规划大规模转换。
- [ ] 明确训练用的 42 维扁平 state 向量定义、动作表示和缩放方式；不要仅凭文件名推断各维语义。训练数据与评估环境必须使用同一控制模式。
- [x] 确定观测历史长度、动作 horizon、padding 与归一化规则，记录在 `configs/`；训练和评估共用同一份定义（`configs/pickcube_state_jointpos.toml`，规则见 `dp_manip/README.md`）。
- [x] 编写 ManiSkill H5/JSON → 训练样本的 dataset adapter（`dp_manip/data.py`）；以 H5 `traj_*` 组和 JSON ID 划分 episode，不依据可能提前变真的逐步 `terminated`/`truncated`。
- [x] 选定 DP 实现来源：ManiSkill 官方基线 `examples/baselines/diffusion_policy`（@62ff3a5），与原版 DP 同一 UNet；出处与改动见 `dp_manip/README.md`。
- [x] 接入所需的 Diffusion Policy 模型与采样组件，补齐与训练控制模式一致的 ManiSkill 评估环境（9.23 阶段 2：wsl 装 mani-skill 3.0.1 + diffusers，开环回放 10/10）；只按确定的需求新增依赖，不照搬上游旧环境。
- [x] 在小样本上做加载、单批前向/反向和评估接口检查，然后用少量 PickCube 数据跑通 **「专家轨迹 → (observation, action) 数据集 → DP 训练 → 策略评估」** 完整链路。（9.23：10 条示范 30k 步，训练种子 10/10，测试种子 2–3%。）
- [x] 下次推送后在 wsl 复跑 `eval_dp.py --split train` 验证新参数。（9.23：`best.pt` 训练种子 10/10，与临时诊断一致。）
- [x] 训练脚本记录峰值显存，用来判断同一张卡上能否并行跑多个实验。（9.23：`summary.json`，reserved 约 680 MiB。）
- [x] 采集足量 PickCube 示范（先 100 条，与官方基线同量级），得到第一个可信的 PickCube 基线。（9.23：100 条，测试 best 0.67 / final 0.77，见 `docs/0923-2155.md`。是否转换 `pd_ee_delta_pos` 仍未决定。）
- [x] **定报告口径**：主结果固定用 final.pt，不按验证挑选（9.23 决定，写在 `configs/README.md`）。
- [ ] 录制并查看 PickCube「成功后又失去」的回合视频（100 条 final 有 19/100），确认是否因为示范在成功那一刻就结束。（9.24：看了种子 10000，是到达目标后方块被带偏、机械臂随后静止在偏离位置，不是停不下来；其余同类回合还没看。）
- [ ] 失败分析：100 条 3 个训练种子都失败的 5 个测试种子（10004、10031、10038、10097、10098）有什么共同点。
- [ ] 把 wsl `~/dp-manip-runs/` 下的批处理脚本放进仓库 `scripts/runs/`。
- [ ] 实测同一张卡并行两组实验时的 ms/步 与评估耗时（显存峰值约 680 MiB，限制在 CPU）。

## C. 完成六任务专家数据集

- [x] 对剩余五个任务各生成 1 条专家轨迹，分别确认规划成功和动作回放成功。（9.24：5 个任务全部通过，见 STATUS「其余五个任务试跑」。）
- [x] 检查其他任务是否触发尚未适配的 MPlib API；如需修改，更新项目补丁并重新验证。（9.24：现有补丁够用，5 个任务都没有报接口错误。）
- [ ] 为每个任务定评估回合长度 `max_episode_steps`：ManiSkill 默认值（50 / 100）比示范短；官方 DP 基线用 PushCube 100、StackCube 200、PegInsertionSide 300，PullCube 与 LiftPegUpright 没有官方值，按 100 条示范的最长长度留余量来定。
- [ ] PegInsertionSide 的专家成功率约 1/3（种子 0、1 失败，种子 2 成功），采集 100 条约需试 300 个种子；采集后检查种子分布是否有偏。
- [ ] 对剩余五个任务各采集 10 条成功轨迹，检查随机种子、动作有效性和逐条回放结果；按选定控制模式做 state 重放。
- [ ] 根据首轮训练结果确定正式数据量；当前规划起点为每任务约 200 条成功示范，并非课程硬性要求。
- [ ] 固定训练/验证数据划分及独立的评估随机种子，记录各任务数据量、轨迹长度分布和质量检查结果；新数据同步更新 `manifests/`。
- [ ] 为六个任务分别写明观测、动作空间和成功判据，并说明各自考察的操作技能（重点区分 PushCube 与 PullCube）。

## D. DP 基线、研究实验与交付

- [ ] 确认提案（选的 track、题目、目标、成员分工）已在 Lecture 5 前提交，并把提交情况记录到仓库。

- [ ] 为六个任务分别训练、评估 DP baseline，记录训练配置、checkpoint、日志与计算资源。
- [ ] 在 held-out 评估种子上评估，报告各任务成功率、评估回合数和失败案例；对比要公平，数据量或算力预算不同时写明（评估与分析占 25 分）。
- [ ] **（必做，占 20 分）** 选定并完成一项受控研究实验（数据量、数据质量、超参数或架构）；可以考虑使用 25 / 50 / 100 / 200 条示范构造嵌套子集，研究数据效率。尽早定题，因为它决定每个任务要采多少条数据。（9.23：PickCube 10/25/50/100 条各 1 个训练种子已完成，final 0.02/0.14/0.47/0.77；写报告前需补训练种子。9.24：种子 2、3 已补完，3 个种子平均 0.02 / 0.13 / 0.56 / 0.67，见 `docs/0923-2249.md`。50 与 100 条之差小于种子波动，下一步二选一或都做：50 / 100 条补到 5 个种子；加 200 条一档。）
- [ ] 整理六任务的专家生成过程和策略 rollout 视频（展示时必须有）；需要打通 RGB 渲染和录制。9.23：wsl 的策略 rollout 录像已打通（`eval_dp.py --video`，走 lavapipe），也能离线高画质回放（`--save-states` + `scripts/render_episodes.py`）；专家生成过程的录像还要在 ubuntu 上试。
- [ ] 完成报告（5–8 页）、10 分钟展示加 Q&A、代码与数据、训练 checkpoint、复现说明及贡献声明。
- [ ] 编写 LLM Usage Statement（提案和终稿都要）：逐项列出 Claude 等工具在环境排障、同步脚本、文档、训练代码等部分的用途，以及人工如何核验。
- [ ] （可选，+5 分）评估能否使用 Moore Threads GPU，并在报告中附 profiling 与 benchmark 结果。

## E. 项目收尾与可复现性

- [ ] 检查 `~/Coding/dp-manip-old-backup` 中是否仍有独有的 Git 历史、未提交改动、脚本或轨迹。
- [ ] 检查 `~/Coding/dp-manip-clean-venv-backup` 是否还有需要保留的信息。
- [ ] 将干净安装、版本覆盖、补丁应用、Vulkan 环境变量和专家采集命令写入复现文档。
- [ ] 依赖拆分：`pyproject.toml` 改为 `train` / `expert` 依赖组并重新 lock（会改动 wsl 环境，需单独验证）。
- [ ] 确认备份内容已妥善保留后，再删除旧目录、临时虚拟环境、wsl 的 `.git.bak` 与 `dp-manip-pre-git-*.tgz`；删除前须用户确认。
