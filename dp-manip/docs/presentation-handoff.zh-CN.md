# 汇报 PPT 交接（2026-10-05）

新对话负责制作期末汇报 PPT。本文件汇总要求、结构建议、已有结论与材料位置、尚缺材料。研究细节以各文档为准，本文件不重复推导。

## 1. 约束

- **时长 10 分钟**，**英文 PPT**，范围是**整个项目**（6 个任务）。
- 未出结果的部分不得用预期数据代替；各结论按文档中的预定措辞，不夸大。
- 用户偏好：中文交流、直接行动、简洁更新。集群操作只做被要求的事，出错先停下报告（见 memory `cluster-stop-on-failure`）。

## 2. 课程评分要求（用户原文要点）

1. **Task selection**：至少 6 个不同技能的任务；说明观测、动作空间、成功判定。
2. **Expert demonstrations**：6 个任务都要生成示范；说明过程、数据量、质量检查。
3. **Policy training**：每个任务训练并评估 DP 基线；说明架构、预处理、训练设置、计算资源。
4. **Further investigation**：至少一个研究问题，可以聚焦部分任务（需说明理由）。
5. **Evaluation and analysis**：held-out 评估种子；报告每个任务的成功率、评估回合数、失败案例；比较要公平，说明数据/算力预算差异。

## 3. 10 分钟结构建议（约 10 页正文 + 附录）

| # | 页 | 对应要求 | 时间 |
| --- | --- | --- | --- |
| 1 | 研究问题与概览 | — | 0.5 min |
| 2 | 6 个任务：技能、观测、动作空间、成功判定；PegInsertionSide/PlugCharger → PlaceSphere/LiftPegUpright 的替换 | R1 | 1 min |
| 3 | 示范生成（maniskill-demogen 运动规划 → 控制模式转换 → RGB 渲染；train 400 / val 50；质量检查） | R2 | 0.75 min |
| 4 | DP 架构与训练设置、算力 | R3 | 0.75 min |
| 5 | **每个任务的基线成功率表**（N=100，held-out test，回合数）+ 失败案例 | R5 | 1.25 min |
| 6 | 研究 A：数据量（success–N 小多图）与 backbone 对比 | R4 | 1.5 min |
| 7 | 研究 B①：失败数据负向引导无效 + 失败诊断（约 90% 为抓空） | R4/R5 | 1 min |
| 8 | 研究 B②：专家中途接管（v1 覆盖不足 → v2 抓空接管）方法图 | R4 | 1 min |
| 9 | 研究 B③：纠正训练结果（森林图）+ 机制验证/汇率曲线 | R4/R5 | 1.5 min |
| 10 | 公平性说明 + 结论、局限、下一步 | R5 | 0.75 min |

附录：各任务逐种子表、种子区间、预注册门槛、Insertion 负结果细节、预加载 2.34× 等工程结果。

## 4. 已有结论与数字（可直接用）

**PlaceSphere 数据量**（从零训练 10 万步，test，horizon 200，5 个训练种子均值）：N=25/50/100/200/400 → 4.7% / 25.3%（3 种子）/ 36.0% / 76.6% / **96.2%**。val：7.3/21.3/37.2/80.4/95.2%。来源 `docs/experiment-progress.zh-CN.md` §6、§15；集群 `~/dp-runs-placesphere/placesphere_rgb_unet_n*_s*/eval/{test,val}_final_h200.json`。

**PegInsertionSide**：data-size 16 + backbone 10 + vision_pool 3 + control_mode 3 个 run，val/test 闭环均约 0；专家回放对照通过；特权 state 对照仅 2%。来源 §5、§9。

**Failure-aware 负向引导（PlaceSphere，N=100，s1–s3）**：第一轮 F−B +1.7pp，95% 区间 [−2.0, +5.0]，不可区分；第二轮加大失败数据/调参仍未达推进门槛。来源 §10、§12。

**专家接管**（`docs/failure-expert-takeover-probe.zh-CN.md`）：
- v1 持球后接管：对齐/专家都可行，但只覆盖约 9% 的 baseline 失败，预定门槛未过（§7）。
- 失败诊断：s1 的 488 回合中约 90% 的失败是抓空（闭合时没抓住或只擦碰）。
- v2 抓空接管 probe 通过：16/18 失败被触发，专家成功 75%，重放逐位对齐（§10.1）。
- **纠正训练**（§10.3，`decision.json`）：B / C（+100 个纠正片段）/ D（+等量 63 条普通示范）合计 600 回合成功率 32.8% / **46.7%** / 32.3%；C−B **+13.8pp** [+9.3, +18.3]，D−B −0.5pp [−4.7, +3.5]，C−D **+14.3pp** [+9.5, +19.0]；三个 checkpoint 方向一致、各自显著。
- 森林图：`docs/figures/takeover-correction-forest.png`（英文标注，可直接用）。
- 局限：单任务、N=100 基线、只覆盖抓空、触发器与专家使用仿真特权状态（只用于采集，策略不使用）；微调与从零训练不可直接比较（§10.4）。

## 5. 机制验证 + 汇率曲线（Job 136133，2026-10-06 04:03 已完成）

**已完成**，结论见 `docs/failure-expert-takeover-probe.zh-CN.md` §12，图 `docs/figures/takeover-curve.png`。要点：C25/C50/C100 = +4.3/+6.0/+13.8pp；D63/D150/D300 = −0.5/+3.7/+18.2pp；纠正数据单位效率约为示范的 4 倍（粗估）；B 与 D 组落空后几乎从不成功，C 组学会补救（C100 10.7%），"因落空而失败"率 −13.2pp。注意修正：普通示范给足后同样有效，不能说"示范无效"。以下为原记录。


- 设计见 `docs/failure-expert-takeover-probe.zh-CN.md` §11、`configs/failure_aware/takeover_curve.toml`；入口 `scripts/takeover_curve.py`；作业 `slurm/takeover_curve_pipeline.sbatch`；输出 `~/dp-runs-placesphere-takeover/curve/`（最终 `curve_decision.json`）。
- 2026-10-05 22:00 开始，预计 10-06 早上 5–6 点结束；结束后作业自行退出，不会再提交新作业。
- 已出（仅 s1，待三 checkpoint 汇总后再下结论）：B 抓空率 64%、抓空后成功 0%；C 抓空率 54%、抓空后成功 9%、平均抓取尝试次数更多。
- 查看（`hku.sh` 为 hku-cluster-access skill 的脚本）：`hku.sh run 'squeue -u $USER; grep -E "^== |mechanism:|^s[123] [CD][0-9]+: |Traceback|Error" ~/dp-manip/slurm-dp-takeover-curve-136133.out | tail -30'`
- 结果出来后需要：读 `curve_decision.json`，画汇率曲线图（两条"新增训练窗口 → 成功率"曲线：C25/C50/C100 与 D63/D150/D300，B 为起点）和机制图，按 §11 预定问题 Q1/Q2/M1 写结论，补进文档 §12。森林图脚本可参考本地 `results/takeover-correction/` 的做法。

## 6. 尚缺材料（需向用户/队友要）

1. **任务 1–4**（PickCube / StackCube / PushCube / PullCube）的数据量与 backbone 结果位置——仓库文档仍标"待汇总"。
2. **LiftPegUpright**（队友）：示范生成统计、基线成功率、failure-aware 结果；配置 horizon 仍为 50，需核对。
3. **PlugCharger** 的替换理由与实验记录。
4. 除 PlaceSphere/PegInsertion 外各任务的示范质量检查记录（demogen `replay_stats.py` 输出等）。

## 7. 参考与工具

- 总进度：`docs/experiment-progress.zh-CN.md`；PPT 早期建议结构 §7、材料清单 §8。
- 集群访问：`hku-cluster-access` skill；VPN 断开需用户 2FA 重连。
- 本地结果副本：`results/takeover-correction/`、`results/takeover-probe-v1/`、`results/failure-round2-review/`（gitignore）。
- 代码尚未 commit（交接与研究文件均为未跟踪/已修改状态），不要 reset/clean。
