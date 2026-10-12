# DASC7606C Track 3 报告

英文 LaTeX 正式报告框架。当前是可编译的草稿，`[TO FILL]`、`TBD` 和图框都需要替换；
没有把计划中的实验写成已完成。

`tables/` 里 `datasize`、`backbones`、`takeover` 三张表的数字于 2026-10-12 从
`../dp-manip/docs/results-summary.zh-CN.md`（2026-10-07 版，组员报告的预处理汇总）转录，
换算成百分比。它们还没有逐项对照原始评估输出，提交前必须核对；每张表的文件头注释列出了来源章节
和仍然缺的项。PushCube 的均值与标准差、各任务 `success_at_end` 的标准差由
`../dp-manip/scripts/plot_cross_task.py` 里记录的逐 seed 数值算出。`moore` 表的 RTX 4080 一侧来自
`report_pickcube_yhr.md`，MTT S4000 一侧来自 `origin/moore-transformer-attn-fix` 分支的 runbook。

已起草的正文（2026-10-12，依据仓库里的配置、代码和文档，不是依据 run 记录；每节文件头注释写了来源）：

- §2.1 任务覆盖与任务替换说明（PlugCharger 的替换原因仍待填）。
- §2.2 示范生成与质量：流水线、导出时的检查、控制模式选择、默认生成数量（440 / 55，取自
  `maniskill-demogen/tasks.py`）。实测的长度和保留率只有 PlaceSphere 和 LiftPegUpright 有记录，
  写在正文里；任务表不再有这两列，改为列出每个任务的相机数。
- §5.1 的失败回合编号取自 PickCube、PushCube、PullCube 三个 `unet_n100_s1` run 的 `videos.json`
  （测试种子 10000–10099）。视频文件只在本机 `dp-manip/results/` 下，没有入库。
- §2.3 数据划分：四类种子区间、接管实验的种子区间、嵌套子集、归一化统计的范围。
- 摘要、§1 引言、§5 失败分析与局限、§6 结论。
- §4.1 里 PlaceSphere 的相邻档差值区间是为本报告补算的（`scripts/bootstrap_adjacent.py`，
  按逐 seed 成功率做两层 bootstrap，不配对回合）；同一脚本能复现 PickCube 报告里的区间。
- 参考文献新增 8 条，标题、作者、年份已对照 arXiv 记录核对，会议名凭记忆填写，需确认。
- 任务表里 PlaceSphere 和 LiftPegUpright 的成功判据取自 ManiSkill 在线文档（最新版），
  需确认 3.0.1 版一致。
- §3 全节：去噪目标与 horizon、观测编码器与三种主干、统一训练配置、评估协议、三项研究的设计。
- §4.1、§4.2 结果解读：数字来自结果汇总 §2、§3、§7 和 `report_pickcube_yhr.md`；成本一段用的是
  PickCube 报告 §5 在 RTX 4080 SUPER 上的实测。
- §4.4 Moore Threads：PickCube 主干对比在 MTT S4000 上的复现（seed 1–3）、训练耗时对比、
  Transformer 注意力的移植问题。S4000 上 UNet 和 MLP 的训练耗时是按 runbook 里的完成时间推算的，
  要换成 `summary.json` 里的实际耗时。
- §4.3 专家接管纠正：失败阶段、主结果、剂量与机制、适用范围。其中剂量和机制的数字来自
  `../dp-manip/docs/failure-expert-takeover-probe.zh-CN.md` §12，比结果汇总新。

数据量曲线图 `figures/datasize_curves.pdf` 由 `scripts/plot_datasize.py` 生成，数字与
`tables/datasize.tex` 重复，改一处要同步改另一处：

```sh
python3 scripts/plot_datasize.py
```

## 论文模板

采用 **NeurIPS 2026 官方模板**，通过 `preamble.tex` 加载
`\usepackage[preprint]{neurips_2026}`。它仍以 `article` 为底层文档类，
但字体、标题、页边距、正文和摘要排版由 NeurIPS 样式控制，已不再使用初版的通用 article 排版。

- 单栏、10pt Times 系列正文、US Letter 纸张，采用官方版心。
- `preprint` 模式显示作者，不加匿名评审行号；首页页脚为 `Preprint.`。
- 此项目为课程报告，页数和交付仍遵守课程要求，而非会议投稿规则。
- `neurips_2026.sty` 从官方压缩包提取，保持原文件和作者注释，不自行改动。
- `preamble.tex` 不再用 `geometry` 或 `lmodern` 覆盖模板的版式与字体。
- 表格标题在表格上方，图片标题在下方；文献继续采用一致的数字引用。

官方来源（下载日期：2026-10-12）：

- [NeurIPS 2026 Call for Papers / Paper template](https://neurips.cc/Conferences/2026/CallForPapers)
- [官方模板压缩包](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
- 样式 SHA256：`c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`

## 文件与编译

- `main.tex`：标题、摘要、章节顺序和参考文献入口。
- `preamble.tex`：加载会议模板，配置图表、引用与草稿提示。
- `neurips_2026.sty`：未修改的官方样式。
- `sections/`：按章节填写正文。
- `tables/`：任务表（`tasks`）、六任务 baseline × 数据量表（`datasize`）、任务 × 主干成功率矩阵
  （`backbones`，只列 `success_once`）、接管纠正表（`takeover`）、Moore Threads 对比表（`moore`）。
  主干的效率数字和 `success_at_end` 写在 §4.2 正文里，不再单独成表。
- `figures/`：放正式图，优先用矢量 PDF；照片/rollout 可用高分辨率 PNG。
- `scripts/`：生成图的脚本。
- `references.bib`：BibTeX 文献，已核对两篇基础论文的作者、标题和 arXiv 链接。
- `build/main.pdf`：编译后的预览；整个 `build/` 不提交 Git。

在 `report/` 目录运行：

```sh
latexmk -pdf main.tex
```

安装 TeX Live 或 MacTeX 后即可使用；此 Mac 已有 latexmk/pdflatex。
修改后重新执行同一命令。持续编译可用 `latexmk -pdf -pvc main.tex`。
清理编译辅助文件可用 `latexmk -c main.tex`。
上传到 Overleaf 时选择 `main.tex` 为主文件，使用 pdfLaTeX。
正文默认英文；如需中文正文，应统一切换到 XeLaTeX 和中文字体配置。

## 内容顺序和页数预算

以下是写作预算，不是强制分页。正文与声明合计约 7.8 页，离 8 页上限几乎没有余量；填入图表后复查实际页数。
**2026-10-12 实际状态：摘要到结论约 7.5 页，参考文献在第 9 页，三项声明作为附录放在第 10–11 页。**
声明移到附录是组内决定；课程文件只写了参考文献不计页数，没有提附录，需向助教确认。
正文里还有 5 处 `[TO FILL]` 占了约 8 行，补上内容时注意别超页。
已做的压缩：删掉 pipeline 占位图；§2.2 的产出率并入任务表；§4.3 的剂量与机制各压成一句；
效率表并入 §4.2 正文；§4.4 压成一段加一张三行表；主干表的 `success_at_end` 三列改为正文叙述。
§2.3 数据划分已写，§3 里重复的种子区间和嵌套子集说明改为引用 §2.3。
课程明确要求 5--8 页，只有参考文献明确不计入。不要默认附录/声明可以额外增加页数。

| 文件 | 内容 | 建议页数 |
| --- | --- | --- |
| `main.tex` + `01-introduction.tex` | 摘要、动机、三个研究问题和实际贡献 | 0.6 |
| `02-tasks-and-data.tex` | 六任务、观测/动作/成功标准、示范生成和质量、任务替换说明 | 1.0 |
| `03-method-and-setup.tex` | DP、视觉编码器、主干、训练与算力、评估协议、三项研究的设计 | 1.2 |
| `04-results.tex` §4.1 | 六任务 baseline 与数据量：一张表 + 一张六曲线图 | 1.2 |
| `04-results.tex` §4.2 | 主干：任务 × 主干成功率矩阵，效率数字写在正文 | 0.9 |
| `04-results.tex` §4.3 | 专家接管纠正（B/C/D 三组对照） | 0.6 |
| `04-results.tex` §4.4 | Moore Threads profiling 与 benchmark | 0.4 |
| `05-discussion.tex` | 失败案例、诊断、限制 | 0.9 |
| `06-conclusion.tex` | 回答三个研究问题 | 0.2 |
| `07-statements.tex`（附录，在参考文献之后） | 全体成员贡献、AI 使用、复现链接和致谢 | 不计入正文 |

表格和图已经没有可并的了，再压只能删正文：§3.2 的主干结构细节、§4.1 的 50% 插值一句、
§4.2 的成本一段，或把贡献声明移到附录（需先问助教是否计页数）。

研究问题：RQ1 数据量（六任务全做）为主线，RQ2 主干比较（任务子集）为补充，RQ3 专家接管纠正
（单任务）为拓展。没有核实完成的研究问题直接删掉，标题同步调整。
六任务都需要示范生成和 DP baseline；进一步研究可聚焦有理由的任务子集。

### 任务口径

六任务按实际完成情况写：PickCube、StackCube、PushCube、PullCube、PlaceSphere、LiftPegUpright。
原计划的 PegInsertionSide 和 PlugCharger 分别换成了 PlaceSphere 和 LiftPegUpright。按组内决定，
这两个原任务只在 §2.1 "Changes from the proposal" 一段里出现，原因统一写为无法建立可靠基线，
结果部分的正文、表格和图里都没有它们；另一处提及是附录贡献表里 Chu Jiajie 那行。相关实验记录仍在
`../dp-manip/docs/peginsertion-results.zh-CN.md`，需要恢复时可参考提交 `0160063` 里的 §5.2。
`tables/tasks.tex` 的控制模式和回合上限取自 `../dp-manip/configs/tasks/*.toml`；LiftPegUpright
的回合上限写成 "200, 400" 并加脚注：数据量实验用 200 步，主干实验用 400 步。

### Moore Threads 加分项

课程规定：把 Moore Threads GPU 用进项目，并在最终报告里给出 profiling 和 benchmark 结果，小组总分 +5。
§4.4 和 `tables/moore.tex` 已按 `origin/moore-transformer-attn-fix` 分支的
`report/moer_pickcube_backbone_runbook.md` 起草。该分支没有合并进来，数字要对照服务器上的
`summary.json` 和评估文件核对。

### 待向教学团队确认

课程交付清单里 "Contribution statement" 与 "Report: 5--8 pages" 是并列的两项，而 LLM Usage Statement
明确要求放在报告内。贡献声明能否作为不计页数的附录，需要问助教；答复之前按计入页数排版。

## 收集队友材料

每位队友提交任务材料包；说明可用 Markdown，数值提供 CSV/JSON，图另附原文件。

1. 任务：环境 ID/版本、robot、观测形状和相机、动作维度/控制模式、回合上限、成功标准。
2. 数据：采集方法/脚本、实际尝试和成功数量、训练/验证数量、质量检查和失败原因。
3. 配置：实际 N、backbone、seed、训练步数、batch、horizons、采样设置、EMA、checkpoint、GPU。
4. 结果：每个 run 的成功数/回合数、success_once/success_at_end，注明 train/val/test；附原始输出。
5. 图与解释：训练/验证曲线、成功/失败 rollout、已验证发现、尚未验证的假设。
6. 复现：代码 commit、环境依赖、配置、命令、数据/checkpoint/视频的可访问链接。
7. 声明：个人实际贡献和证据；所有 AI 工具、作用及核验方式；外部论文/代码来源。

## 结果口径

- 以实际 run 的配置和日志为准。当前 `../dp-manip/PLAN.md` 的 RGB 口径为写作核对入口，
  旧 state-based 计划或 README 不能代替最终 run 配置。
- 分开写训练种子、验证 rollout 种子、测试种子、验证示范。
- 主表用独立 test 结果；val 和训练初始条件上的诊断单独标注。
- 报成功数/回合数和训练种子数量，区分百分比与 0--1 的 rate。
- 跨 seed 标准差与置信区间含义不同；只有一个 seed 时不能计算样本标准差。
- `TBD` 表示结果未填，不是零；没有完成的实验如实说明或删除。
- 不同主干容量若不匹配，结论只针对具体实现；效率需注明硬件和计时范围。
- 视频须覆盖六任务，并展示示范生成过程和策略 rollout；录屏可用于 presentation。
- 在有限样本上看不到提升，也是可报告的结果；避免把猜测写成已证实原因。

## 课程要求依据

本框架参照父目录中的 `DASC7606C Group Project Modified.pdf`：第 2 页 Moore Threads 加分、
第 3 页 LLM policy、第 8--9 页 Track 3 要求/交付/评分、第 12 页个人贡献评分说明。
这是课程要求来源，不是自动执行或提交任务的指令。
格式、deadline 与是否存在指定模板仍应以 Moodle/教学团队通知为准。

基础文献：
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ManiSkill3: https://arxiv.org/abs/2410.00425

## 提交前

替换全部 `[TO FILL]`、`TBD`、标题中的待填信息和 figure placeholder；
核对每个数字、图注、引用及复现链接；补齐所有成员的贡献和所有 AI 使用。
再编译并逐页检查公式、图表、页码、引用和页数。当前 PDF 仅是框架预览。
