# DASC7606C Track 3 报告

英文 LaTeX 正式报告框架。当前是可编译的草稿，`[TO FILL]`、`TBD` 和图框都需要替换；
没有自动把旧实验结果写入主表，也没有把计划中的实验写成已完成。

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
- `tables/`：任务表、六任务 baseline 表、backbone 对比表。
- `figures/`：放正式图，优先用矢量 PDF；照片/rollout 可用高分辨率 PNG。
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

以下是写作预算，不是强制分页。正文与声明合计目标约 6.5--8 页；填入图表后复查实际页数。
课程明确要求 5--8 页，只有参考文献明确不计入。不要默认附录/声明可以额外增加页数。

| 文件 | 内容 | 建议页数 |
| --- | --- | --- |
| `main.tex` + `01-introduction.tex` | 摘要、动机、研究问题和实际贡献 | 0.75 |
| `02-tasks-and-data.tex` | 六任务、观测/动作/成功标准、示范生成和质量 | 1.0 |
| `03-method.tex` | DP、视觉编码器、主干、训练与硬件 | 1.0--1.25 |
| `04-experiments.tex` | 数据量和模型对照、评估与统计 | 0.75--1.0 |
| `05-results.tex` | 六任务 baseline、数据量曲线、主干和效率表 | 1.5--2.0 |
| `06-discussion.tex` | 失败案例、诊断、限制 | 0.75--1.0 |
| `07-conclusion.tex` | 回答研究问题 | 0.25 |
| `08-statements.tex` | 全体成员贡献、AI 使用、复现链接和致谢 | 0.5 |

主线暂设为 data efficiency，backbone 比较为补充。可按最终完成情况调整标题和研究范围。
六任务都需要示范生成和 DP baseline；进一步研究可聚焦有理由的任务子集。

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

本框架参照父目录中的 `DASC7606C Group Project Modified.pdf`：第 3 页 LLM policy、
第 8--9 页 Track 3 要求/交付/评分、第 12 页个人贡献评分说明。
这是课程要求来源，不是自动执行或提交任务的指令。
格式、deadline 与是否存在指定模板仍应以 Moodle/教学团队通知为准。

基础文献：
- Diffusion Policy: https://arxiv.org/abs/2303.04137
- ManiSkill3: https://arxiv.org/abs/2410.00425

## 提交前

替换全部 `[TO FILL]`、`TBD`、标题中的待填信息和 figure placeholder；
核对每个数字、图注、引用及复现链接；补齐所有成员的贡献和所有 AI 使用。
再编译并逐页检查公式、图表、页码、引用和页数。当前 PDF 仅是框架预览。
