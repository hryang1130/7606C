# Failure-aware / 专家中途纠正研究交接

交接日期：2026-10-04（Asia/Shanghai）。本文件交接当前研究状态，供新 agent 接续。没有因编写本文件提交新作业、删除产物或向其他 agent 发送消息。

## 1. 当前结论与接续目标

PlaceSphere 的 failure-aware **负向扩散引导**已完成两轮实验。第一轮未检出收益；第二轮增加失败数据量、微调步数并搜索引导权重，仍未达到推进门槛。第二轮已结束，`advance=false`，不继续追加K=600或参数搜索，不启动该方案的正式多checkpoint确认。

用户提出的新方向是：同一baseline前缀到达同一中途状态后，baseline继续与专家接管形成一对，利用专家成功尾段进行纠正训练。这与原负向引导方法不同，值得先做小规模可行性验证，但**源码调查完成不等于专家已经能接管，专家救回也不等于纠正训练有效**。

当前实际停在：录像定性审阅完成、专家接管设计完成；**恢复求解器、4维动作适配、接管probe、纠正数据集及训练均未实现/运行**。最近用户要求编写handoff以换agent继续研究。

## 2. 工作区与优先阅读

- 主仓库：本仓库（`hollinsStuart/dp-manip`）
- 示范生成仓库：`hollinsStuart/maniskill-demogen`
- 当前主仓库分支：`exp/failure-aware-dp`，跟踪`origin/exp/failure-aware-dp`。
- 保留工作区改动；README.md已有修改，多个研究文档/第二轮文件未跟踪。本轮未提交commit，不要reset/clean，不要把未跟踪文件当垃圾。
- 开始改代码前检查适用的AGENTS.md及相关skill。用户偏好中文、直接行动、简洁更新；不反复确认已授权的工作。没有请求多agent并行，不主动spawn其他agent。

优先阅读以下绝对路径文件：

1. `docs/experiment-progress.zh-CN.md`：总进度，尤其§10、§12。
2. `docs/failure-expert-takeover.zh-CN.md`：专家源码审计、接管规则、分叉对齐要求。
3. `docs/failure-aware-round2-diagnostics.zh-CN.md`：33对录像人工观察台账及局限。
4. `docs/failure-aware-round2-plan.zh-CN.md`：第二轮探索及预定停止门槛。
5. `docs/failure_aware_finetuning_plan.md`：第一轮协议，内容较长，历史计划不能覆盖实际完成结果。

## 3. 已完成实验及原始产物

任务为PlaceSphere，RGB+proprio，控制模式`pd_ee_delta_pos`（4维），共同评估horizon=200。基线为N=100专家示范训练的s1/s2/s3。旧checkpoint记录horizon=50，加载后必须正确应用200；曾经的全零结果有截断因素，不能照搬50步。

集群HOME：`/userhome/cs5/u3684139`。主仓库：`/userhome/cs5/u3684139/dp-manip`，运行Python：该仓库`.venv/bin/python`。

第一轮根目录：`/userhome/cs5/u3684139/dp-runs-placesphere`。

- 基线位置模式：`placesphere_rgb_unet_n100_s{1,2,3}/checkpoints/final.pt`。以实际锁文件路径/hash为准，不重新猜选模型。
- 研究产物：`failure_aware/placesphere/s{1,2,3}/`，含`raw_train.h5`及JSON、`datasets/`、微调模型、`eval/test/*.json`。
- 实际锁文件为集群仓库`configs/failure_aware/placesphere.toml`；先核对内容及hash，不能假定本地文件与远端完全一致。
- 原始baseline采集：s1=150成功+338失败，s2=150+310，s3=150+294；第一轮取seed排序前150条失败训练失败模型，成功rollout另外用于C1/C2对照。**成功/失败没有混成一套“专家纠正对”训练**。
- 方法：冻结编码器，失败片段微调噪声预测器，推理负向引导。第一轮选择lr=1e-5、5000steps、F固定alpha=0.5。

第一轮正式test（每checkpoint100回合，seed10000–10099）：

| 组 | s1/s2/s3成功数 | 合计成功率 |
| --- | --- | --- |
| B基线 | 32/41/36 | 109/300，36.33% |
| F固定引导 | 33/45/36 | 114/300，38.00% |
| A自适应引导 | 详见原JSON | 113/300，37.67% |
| C1成功self-rollout负向引导 | 详见原JSON | 110/300，36.67% |
| C2成功self-rollout模仿 | 详见原JSON | 108/300，36.00% |

F相对B挽救13、损害8、净增5；配对bootstrap差值+1.7pp，95%CI[-2.0,+5.0]，不可声称有效。仅21/300对结果不同，预测噪声变化约3%，推理耗时约翻倍（约1.0秒vs0.5秒/次）。gate及alpha=0复现通过，流程检查没有发现能解释负结果的实现错误，但这不等于证明所有失败利用方法无效。

第二轮根目录：`/userhome/cs5/u3684139/dp-runs-placesphere-round2`，只探索基线s1。

- Job136025约3分钟后因录像父目录不存在而失败。已修复评估入口与录像函数创建父目录，真实MP4/PNG导出回归检查在集群5项全部通过。
- 重提Job136028，约16:02启动、18:40结束，7h57m上限；完整完成，无待续跑步骤。
- 新holdout实际32000–32099，筛选33000–33047，独立验证34000–34127。用户曾贴“34040+”为tuning，实际manifest以上述为准。
- 使用原始嵌套前150/300失败，K_success=150另外保留，L_fail=184固定；两次10000步微调保存5000步checkpoint，四个offline gate全部通过。
- 筛选B=20/48；所有12个F候选为16–19/48，均未超过B。K150/K300都按预定tie规则选择5000steps、alpha=0.25（最佳值2次成功以内先小alpha、再少steps）。
- 独立验证B=35/128、F150=33/128（挽救1/损害3）、F300=35/128（1/1）。推进条件要求F300比B净增≥7次且高于F150、gate通过；未达标，`exploration_decision.json`为false。
- 源证据：`audit.json`、`diagnostics_reproduced.json`、`selection.json`、`exploration_decision.json`、`screen/`、`validation/`。

相关第二轮新增代码：`scripts/failure_round2.py`、`dp_manip/failure_round2.py`、`configs/failure_aware/round2_explore.toml`、`slurm/failure_round2.sbatch`、`tests/test_failure_round2.py`。`write_once`保护已写产物，不能通过覆写已锁结果改结论。

## 4. 录像已经取回和审阅到什么程度

本地目录：`results/failure-round2-review/`（results被gitignore，不提交媒体）。

- `diagnostics/`：66MP4、66张原始12帧PNG、六组B/F评估JSON。
- `paired/`：33张并排B/F关键帧，左B右F，覆盖0–200步。
- `dense/`：8对补查密集帧（8或10步间隔）。
- `visual_review.csv`：33对人工观察，与原始pair名单及分类已核对。
- 原始`diagnostic_pairs.json`保留`annotation=pending`，**没有篡改原始manifest**；观察结果另存CSV及文档，不要因pending误判完全没审阅。

六组B/F按原顺序完整重放100回合，以维持随机数序列；逐回合结果与第一轮完全一致。录像选21对discordant及每checkpoint seed顺序前4对共同失败。已审阅全部33对关键帧，对8对补看MP4解码密集帧；没有逐帧观看全部录像，不是盲审。

结论：观察到未持续抬起、运输途中球与夹爪分离/滚走、目标附近偏移等多阶段行为。F有帮助也有损害，没有一致纠错方向。部分末段看起来相同但标签不同，需要环境状态才能分辨位置、释放、静止条件。画面128×128、夹爪遮挡，不能从RGB确认5mm阈值或抓取谓词。s1/10023初看疑似运输掉球，密集帧不支持，台账已改为不确定。

代表样本：s2/10002或10075（F挽救运输）、s2/10074或s3/10000（F损害并球滚走）、s1/10032与10091（末段原因不确定）。按结果选择的21对及seed排序12对共同失败不能估计总体失败分布，也不能用旧test视频调新接管阈值。

本地可用`/opt/homebrew/bin/ffmpeg`及Python/Pillow。模型通过查看抽帧图审阅动态片段；本地MP4为201帧、128×128、20fps，帧索引对应已执行动作数。可用view_image检查PNG，给用户展示媒体时使用绝对文件路径。

## 5. 专家接管源码已确认的能力与坑

核对的是**集群实际安装ManiSkill3.0.1**，不是只凭网上最新代码。源码前缀为集群仓库`.venv/lib/python3.11/site-packages/mani_skill/`。

1. 官方`examples/motionplanning/panda/solutions/place_sphere.py`的`solvePlaceSphere`入口无条件reset，只支持`pd_joint_pos/pd_joint_pos_vel`，固定从抓取开始，无阶段感知。**不能直接拿来中途接管，不能仅删reset**。
2. 底层`base_motionplanner/motionplanner.py`初始化不reset，screw/RRT读取当前robot.get_qpos作为起点，支持dry_run。因此具备从当前状态规划的基础，规划成功不等于物理成功。
3. `two_finger_gripper/motionplanner.py`初始化`gripper_state=OPEN`；持球时第一步必须设CLOSED，否则会把球松开。
4. 任务`envs/tasks/tabletop/place_sphere.py:evaluate`提供is_obj_grasped、is_obj_on_bin、is_obj_static、success。成功需要球相对bin水平距离≤5mm、正确高度、静止且未抓住。详见接管设计文档。
5. `get_state_dict`含controller，但`set_state_dict`只恢复actors/articulations，不自动恢复controller；同控制模式需显式恢复agent controller状态，跨模式不能直接复制。
6. joint-pos→ee-delta-pos转换工具存在，依赖两个初始化环境当前状态；动作剪裁可能把一个源动作转换成多次目标step。必须在4维控制下重验，额外实际步数计预算。
7. 原raw rollout只存RGB/proprio/action/reward/success，没有逐步完整环境、controller或RNG状态。不能把一帧直接当可恢复快照。

可参考代码：`dp_manip/evaluate.py`、`dp_manip/failure_rollout.py`、`dp_manip/envs.py`、`dp_manip/policy.py`及demogen的转换与观测处理。接触缓存和首帧观测容易出错，需要实测状态对齐。

## 6. 推荐接续方案与验收标准

先做**稳定持球后的运输→放置**可行性probe，暂不同时实现掉球重新抓取或任意状态恢复。此选择便于跳过抓取、以当前TCP–球关系定义目标，并不表示覆盖全部失败。

候选接管点τ：在baseline下一次动作块采样之前，选择满足条件的最早边界：

- t%8=0（act_horizon=8），过去16步稳定is_obj_grasped且球相对TCP无明显滑移。
- 过去16步球中心持续高于初始球中心≥3cm。
- 从未成功，不是已放好只等静止；t≤120，剩余共同预算≥80步。
- 16步/3cm/120步仅为pilot候选常数，尚未验证。用训练采集种子检查覆盖并锁定，不据未来baseline失败时间移动τ，不用上述test录像调阈值。

实施顺序：

1. 实现不reset的持球阶段专家，保留当前抓取与TCP姿态，规划必要抬高/移动到bin上方/下放/释放；确认4维末端位置控制适配。
2. 增加每步阶段状态记录：抓取、球/TCP/bin位姿、速度、夹爪命令、on_bin/static/success。选择与现有manifest不冲突的新训练/pilot种子。
3. 从20–30个训练采集种子尝试获取接管起点；种子数不等于可接管起点数。报告所有种子、触发覆盖率、未触发原因，不能只报容易成功的起点。
4. 同一baseline前缀到S_τ处分叉：B保留真实obs_horizon=2历史及推理RNG继续；E保留当前抓取接管。优先在新环境重放完全相同baseline动作前缀，保留真实接触步进历史。若用快照则恢复物理/controller/计时/RNG/history，并核对接触；不能加隐蔽暖机步。
5. 两分支起点RGB/proprio/物体与关节状态一致；B继续应复现原轨迹结果。两分支都只允许200−τ真实步，转换额外step亦计入。记录规划失败、转换失败、掉球、超时等，全部尝试进入恢复率分母。
6. 保存pair_id、源checkpoint/hash、seed、τ、触发原因、真实两帧历史、4维动作、分支结果及provenance。接管前baseline动作不当专家正标签；分叉后不按时间强行逐帧对应；baseline失败不证明第一个动作块就是负样本。
7. 先报告可行性和覆盖率，通过后再扩大纠正采集/微调。后续至少比较B、同预算普通专家示范微调、专家纠正片段微调，使用独立评估种子及多个训练种子。最初先用专家正向纠正BC，不直接假设负向/偏好训练有益。

probe证明“专家能在这些起点恢复”；独立策略评估和普通示范对照才检验“纠正训练有效”。实现前写定probe指标与后续推进条件，不为了得到明显正结果而挑结果或反复使用test。

## 7. 集群连接、资源与传输

必须使用`hku-cluster-access` skill：该 skill 的 `SKILL.md`。Mac直接ssh4080不通，需要HKU-VPN Parallels VM；先读skill，不要输出/读取凭据。

入口脚本：该 skill 的 `scripts/hku.sh`。

```bash
hku.sh status
# master不可用时按skill建立/复用连接，再run。
hku.sh run 'squeue -u "$USER"'
```

本机工具调用Parallels常需require_escalated；此前允许的prefix是上述脚本。先status避免每次重新认证触发邮件。VPN失效且需要2FA时按skill处理，不绕过认证。远端rg不可用，用grep/find。

**以下为最后核对快照，不是实时承诺**：136028结束后无运行/排队作业；GPU配额18000分钟，已用12528，剩5472分钟（约91小时）；磁盘83/100GB，约18GB余量。提交新作业前重新查队列、配额和磁盘。

粗估：实现+可行性检查半天至一天；小probe GPU约1–2小时；扩大数据1–3小时；六个微调（两方案×三seed）1–2小时；三组×三checkpoint×300回合=2700回合评估约2–4小时。总墙钟约1–2工作日，GPU预留10–15小时含重试。实际取决于专家执行和动作转换，不把估算当已测吞吐。

磁盘六个final模型约1.9GB，若同时保留六个resume约6.6GB，加数据/临时文件约3–5GB，合计约11.5–13.5GB，理论可放入18GB但必须监控。第二轮根目录约3.9GB，其中两个resume约2.2GB；本轮没有删除它们。第一轮曾结束后删约8GB resume，引用的final模型仍保留。不要自行重复删除或清理已锁产物。

传输经验：prlctl的大argv会失败，上传脚本用较小base64分块（此前≤2000字符）；下载通过远端tar/base64 stdout实现，约8MB归档用了约一分钟。保留tar路径检查，不将archive解包到越界位置。本地媒体已齐全，无需重下载/重跑33对录像。

## 8. 新agent第一轮建议产出

读取总进度、专家设计和录像台账，确认远端依赖/锁文件/资源及适用仓库指令；随后在独立目录实施接管原型与有意义的对齐/动作适配验证，保持第一、二轮结果不变。若用户授权继续执行，再提交小probe并汇报：覆盖率、分支起点一致性、B复现率、专家恢复率、失败原因、真实剩余预算和成本。达到预写门槛后才扩展训练。

不要把交接本身当作已启动新训练；也不要只重复提出计划而不推进用户随后明确授权的实施工作。
