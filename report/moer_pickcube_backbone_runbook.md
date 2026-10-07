# 摩尔服务器 · PickCube backbone 消融训练记录（2026-10-05 起）

> 任务：Track 3 轨道 B（主干结构消融）在摩尔线程 MUSA 机器上的执行记录与运行手册。
> 执行人：本机 agent（WorkBuddy）；口径确认：seed 1–3、单卡（第 8 张，ID 7）。
> 本地参考脚本 `moer_train.sh` 只作模板，**实际执行以本记录为准**。

---

## 1. 实验口径

| 项 | 值 | 来源 |
| --- | --- | --- |
| 任务 | `pickcube` / PickCube-v1，`pd_ee_delta_pos`（4 维动作） | `configs/tasks/pickcube.toml` |
| 实验 | `configs/experiments/backbone.toml`（自变量 `policy.backbone`） | 同上 |
| 主干 | unet / transformer / mlp | 同上 |
| 训练种子 | **1, 2, 3**（与本人 2026-10-05 确认；官方 `backbone.toml` 声明 1–5，本次只取 1–3） | 本次决定 |
| N_B（训练示范数） | 100 | `docs/final-plan.md` §6 |
| 训练预算 | 100k optimizer steps，batch 64 | 固定口径 |
| 产物 | `runs/pickcube_rgb_<backbone>_n100_s<seed>/`，主结果看 `checkpoints/final.pt` | 固定口径 |
| 运行规模 | 3 主干 × 3 种子 = **9 个 run**，单卡串行 | 本次决定 |

---

## 2. 环境事实（已实测核对）

| 项 | 实测值 |
| --- | --- |
| 服务器 | `10.123.0.45`，宿主机 `worker00045`，登录用户 `group11` |
| 容器 | `dp`（`registry.mthreads.com/presale/devtech/vllm-mt:d0930`），已运行，**不要重建/不要改环境** |
| 挂载 | **只有一条**：宿主 `/home/group11/7606C` → 容器 `/workspace/code`（`7606C-main` 没有挂进去） |
| 工程路径 | 容器内 `/workspace/code/dp-manip`（= 宿主 `~/7606C/dp-manip`） |
| 数据根 | 容器内 `/workspace/code/maniskill-demogen/data/dataset` |
| Python / torch | `/usr/bin/python` 3.10.12，torch 2.9.0 + **torchada 0.1.90**（把 `torch.cuda.*` 重定向到 MUSA，解释器级生效，无需手动 import） |
| 设备 | 8 × MTT S4000（48GB/卡），`torch.cuda.device_count()=8` |
| 环境自检 | `python scripts/check_musa.py` → **11/11 PASS**（matmul / conv2d 反传 / GroupNorm / RNG / amp / GradScaler / pin_memory） |
| 磁盘 | `/home` 191G，已用 57G，余 125G |

### GPU 布局（启动时）

| 卡 | 状态 |
| --- | --- |
| 0–3 | 队友占用（stackcube 等训练），**不要动** |
| 4–6 | 空闲（本次未使用；如需加速可在征得同意后各起一个实例） |
| **7** | **本次使用**（"第 8 个 GPU"），启动后 94% 利用率、约 4.1GB 显存 |

### 与本地 `moer_train.sh` 的差异（脚本确实"不一定对"，已逐条修正）

| 本地脚本 | 实际情况 / 本记录 |
| --- | --- |
| `DATA_ROOT=.../maniskill-demogen/data/dataset`，假定 PickCube 数据就位 | 容器里原本**没有** PickCube：数据在 `~/7606C-main/data_yhr/data/dataset/{train,val}/PickCube-v1`，**容器挂载之外** |
| 未处理数据桥接 | 已把数据**复制**进 `~/7606C/maniskill-demogen/data/dataset/{train,val}/`（158M + 20M）。注：**软链接行不通**——链接目标在挂载范围外，容器内会解析失败 |
| `CONCURRENCY` 默认 3 | 改为 **1**：队友实测同卡多进程吞吐只剩独享的 ~30%（单 run 5.4h → 36h），单卡必须串行 |
| `SEEDS` 默认 `1 2 3` | 本次确认沿用 **1 2 3**（官方 `backbone.toml` 是 1–5，本次不取 4/5） |
| 无 `GPU_INDEX` 默认可用值 | 固定 **7** |
| 未提及 torchada | 无需额外处理，直接 `python scripts/train_dp.py` |

---

## 3. 启动前的验证（都已通过）

1. `mthreads-gmi` → 确认卡 7 空闲、卡 0–3 是队友的
2. `python scripts/check_musa.py` → 11/11 PASS
3. `python scripts/inspect_dataset.py --data-root /workspace/code/maniskill-demogen/data/dataset --config configs/tasks/pickcube.toml`
   → `PASS PickCube-v1: train 400 demos/31009 steps; val 50; RGB (128,128,3) ['base_camera']; proprio 29; action 4; action range [-1.000, 1.000]`
4. `DRY_RUN=1 bash run_backbone_pickcube.sh` → 列出 9 个 run，无 CONFLICT
5. `SMOKE=1 bash run_backbone_pickcube.sh` → 2000 步、3 分钟跑完，loss 1.13 → 0.12，写出 `runs_smoke/.../final.pt`
   （输出到 `runs_smoke/`，**不污染正式 `runs/`**）

---

## 4. 完整操作步骤（可复现）

```bash
# ---- 0. 本机连服务器（或直接在服务器上操作）----
ssh group11@10.123.0.45                  # 密码见 账号.txt

# ---- 1. 进容器 ----
docker exec -it dp bash
cd /workspace/code/dp-manip

# ---- 2. （已完成，勿重复）把 PickCube 数据复制进容器可见的数据根 ----
cp -r ~/7606C-main/data_yhr/data/dataset/train/PickCube-v1 ~/7606C/maniskill-demogen/data/dataset/train/
cp -r ~/7606C-main/data_yhr/data/dataset/val/PickCube-v1   ~/7606C/maniskill-demogen/data/dataset/val/

# ---- 3. 数据与环境自检 ----
python scripts/inspect_dataset.py --data-root /workspace/code/maniskill-demogen/data/dataset --config configs/tasks/pickcube.toml
python scripts/check_musa.py

# ---- 4. 启动脚本（已放在工程根：/workspace/code/dp-manip/run_backbone_pickcube.sh）----
DRY_RUN=1 bash run_backbone_pickcube.sh    # 先只看清单
SMOKE=1   bash run_backbone_pickcube.sh    # 试跑 2000 步到 runs_smoke/

# ---- 5. 正式启动（后台，断开 SSH 不影响；已在 2026-10-05 20:22 执行）----
docker exec -d dp bash -lc 'cd /workspace/code/dp-manip && \
  nohup bash run_backbone_pickcube.sh > logs/backbone/launcher_pickcube.log 2>&1 &'
```

脚本默认值（均可用环境变量覆盖）：`TASK=pickcube`、`GPU_INDEX=7`、`NUM_DEMOS=100`、
`BACKBONES="unet transformer mlp"`、`SEEDS="1 2 3"`、`CONCURRENCY=1`、`DATA_ROOT=/workspace/code/maniskill-demogen/data/dataset`。

---

## 5. 怎么看"是不是正在跑"

### 5.1 三条命令，30 秒判断

```bash
ssh group11@10.123.0.45
docker exec -it dp bash
cd /workspace/code/dp-manip

# ① 队列在干什么：哪一个 run 在跑、哪些已完成、有没有 FAILED
tail -5 logs/backbone/launcher_pickcube.log
#   期望：最后一行是 "[start] pickcube_rgb_<bb>_n100_s<seed> (pid ...)"
#   每完成一个会出现一行 "[done]   pickcube_rgb_..."；全部结束会打印 "全部 run 正常结束"

# ② 当前 run 的进度：步数 / loss / 速度（这是最直接的"活着"证据）
tail -f logs/backbone/pickcube_rgb_unet_n100_s1.log
#   期望：持续刷 "[00xxxx/100000] loss=... elapsed=...s"，每 100 步一行，约 18 秒推进一步

# ③ 卡上有没有我的进程
mthreads-gmi | head -40      # ID 7 那一段应显示 ~94%、约 4 GB
```

### 5.2 更省事的批量检查

```bash
# 所有 pickcube run 的步数进度一览（未开始的 run 没有目录/日志）
for f in logs/backbone/pickcube_rgb_*.log; do
  printf '%-42s %s\n' "$(basename $f .log)" "$(grep -oE '\[[0-9]{6}/100000\]' $f | tail -1)"
done

# 已产出的 run 目录与 checkpoint
for d in runs/pickcube_rgb_*; do
  echo "== $d"; ls "$d/checkpoints" 2>/dev/null | tr '\n' ' '; echo
done

# 我自己的训练进程
ps aux | grep train_dp | grep pickcube
```

### 5.3 完成与失败的判据

| 状态 | 判据 |
| --- | --- |
| 正在跑 | run 日志的步数在增长；`mthreads-gmi` 卡 7 有 GPU 利用率；`[start]` 出现但还没有对应的 `[done]` |
| 该 run 完成 | 日志出现 `finished ... in X h; checkpoint: .../final.pt`；目录里有 `checkpoints/final.pt` 与 `summary.json` |
| 全部完成 | launcher 日志打印 `全部 run 正常结束`；`runs/` 下有 9 个 `pickcube_rgb_*_n100_s*` 目录，各有 `final.pt` |
| 失败 | launcher 日志出现 `[FAILED] <name> → logs/backbone/<name>.log`；打开该日志搜 `Traceback` |
| 重启后继续 | 脚本用 `--resume auto`：有 `final.pt` 的 run 直接跳过复用，只有 `resume.pt` 的中断 run 会续训 |

### 5.4 中断 / 恢复

- 训练进程挂掉或机器重启：重新执行第 4 步的第 5 条命令即可，已完成 run 会被跳过。
- **不要**给 train 命令加 `--set` 覆盖训练超参：`run.json` 记录完整配置，与既有 run 不一致时 trainer 判 CONFLICT 直接抛 `FileExistsError`。
- **不要**为了"省时间"先短跑再全量：短跑会写出 `final.pt`，正式跑必然撞 CONFLICT。试跑一律用 `SMOKE=1`。

---

## 6. 进度与时间预期

| 时间 | 事件 |
| --- | --- |
| 20:16 | PickCube 数据复制完成（train 158M / val 20M），`inspect_dataset` PASS |
| 20:19 | SMOKE（mlp s1，2000 步）3 分钟通过 |
| 20:22 | 正式启动：`pickcube_rgb_unet_n100_s1` |
| 20:24 | 实测 600 步 / 109 秒 ≈ **5.5 步/秒**（unet）；卡 7 利用率 94%、4.1GB |
| 20:29 | 2900 步 / 462 秒 ≈ **6.5 步/秒**（含启动约 18s） |

### 实测速率（本机 MUSA，均为 100k 步口径）

| 主干 | 速率 | 单 run 预计 | 依据 |
| --- | --- | --- | --- |
| unet | 6.5 步/秒 | **≈ 4.3 h** | 本次 pickcube s1 实测；另参考已完成的 stackcube_rgb_unet_n200（5.39h）、pushcube_rgb_unet_n25/50（4.22h） |
| transformer | ≈ 6.5 步/秒 | **≈ 4.3 h** | 摩尔上 `stackcube_rgb_transformer_n200_s1` smoke 实测（2000 步 / 307s） |
| mlp | ≈ 13.6 步/秒 | **≈ 2.1 h** | 本次 pickcube mlp smoke 实测（2000 步 / 147s） |

> 单 run 除训练本身外还有约 5 秒数据 preload（100 条示范 → 0.54 GiB 进内存）+ 十几秒启动。

### 总时长（9 个 run 串行，卡 7 独占）

| 阶段 | 顺序 | 时长 | 预计完成时刻 |
| --- | --- | --- | --- |
| unet × 3（s1→s3） | 第 1–3 个 | ≈ 13 h | 10-06 上午 09:30 左右 |
| transformer × 3（s1→s3） | 第 4–6 个 | ≈ 13 h | 10-06 晚上 22:30 左右 |
| mlp × 3（s1→s3） | 第 7–9 个 | ≈ 6.5 h | **10-07 凌晨 05:00 左右** |
| **合计** | | **≈ 32–34 h** | 从 10-05 20:22 起算 |

误差主要来自 transformer 在摩尔上的实际速率（目前只有 2000 步的短测）与卡间干扰；
第一个 run 跑完（约 10-06 00:40）后可用 `summary.json` 里的真实 wall time 校准后续估计。

> 若想压缩到 ~11 小时：需要额外占用卡 4/5/6（各起一个实例、`GPU_INDEX=4/5/6`、把 `BACKBONES`/`SEEDS` 切开分给不同实例）。
> 本次未使用——用户明确只授权卡 7，且那些卡是否属于其他队友尚未确认。


产物目录（容器内 / 宿主）：

```
/workspace/code/dp-manip/runs/pickcube_rgb_<backbone>_n100_s<seed>/
  run.json            # 完整 resolved config、选中的示范 seed、归一化统计、git/dataset 指纹
  metrics.jsonl       # 每步 loss（画曲线用）
  summary.json        # 训练时长、peak 显存、最终 train/val loss
  checkpoints/
    step_010000.pt  step_030000.pt  step_060000.pt  final.pt  resume.pt
宿主对应：~/7606C/dp-manip/runs/...
日志：~/7606C/dp-manip/logs/backbone/{launcher_pickcube.log, pickcube_rgb_*.log}
```

## 7. 闭环评估（RGB）在摩尔上的可行性与用法

**结论：可以评，已实测跑通**（2026-10-05 20:36）。踩坑与配方如下。

### 7.1 为什么一开始跑不起来

| 现象 | 原因 | 解法 |
| --- | --- | --- |
| 容器里 `import mani_skill` 失败 | 训练镜像没有 ManiSkill | 组内已把评估依赖 `pip --target /workspace/code/evalsite`（mani_skill 3.0.1 / sapien 3.0.3 / gymnasium 1.3.0），用 `PYTHONPATH=/workspace/code/evalsite` 即可，**不需要动容器** |
| `RuntimeError: Failed to load vulkan library!` | 容器 `/usr/share/vulkan/icd.d/` 为空；lavapipe 库在 `evalsite/lib` 内 | `LD_LIBRARY_PATH` **追加** `.../evalsite/lib/usr/lib/x86_64-linux-gnu`。容器 `/etc/vulkan/icd.d/lvp_icd.json` 已指向该 lavapipe（队友写入，勿删） |
| `RuntimeError: Failed to load the backend extension: torch_musa` | `LD_LIBRARY_PATH` 被**覆盖**，丢了 `/usr/local/musa/lib` | 必须**追加**（`...:$LD_LIBRARY_PATH`），不能 `=` 覆盖 |
| 环境建起来了但 obs 没有 `rgb`（报 `KeyError: 'rgb'`） | 默认渲染后端是 `sapien_cuda`，MUSA 上回退 CPU 后渲染被禁用 | 显式 `--render-backend cpu`（强制 lavapipe 软渲染），此时 `obs/sensor_data/base_camera/rgb` = (1,128,128,3) uint8 |
| `eval_dp.py` 里 `ensure_render_icd()` 没生效 | 它找 `/usr/share/vulkan/icd.d/lvp_icd.json`，容器里是空的 | 实际靠 `/etc/vulkan/icd.d/lvp_icd.json` 生效，无需改代码 |

物理仍是 `physx_cpu`（与数据生成后端一致，保证公平），渲染走 CPU 软件渲染，**策略推理**才用卡。

### 7.2 用法（脚本已放在工程根 `run_eval_pickcube.sh`）

```bash
docker exec -it dp bash
cd /workspace/code/dp-manip

# 默认：test split（seeds 10000–10099）、num_envs=4、推理在卡 7、渲染 CPU
bash run_eval_pickcube.sh pickcube_rgb_unet_n100_s1

# 完全不占 GPU（训练期间用这个最安全；推理也在 CPU 上）
DEVICE=cpu bash run_eval_pickcube.sh pickcube_rgb_unet_n100_s1

# 只快速看 val（前 50 条）
bash run_eval_pickcube.sh pickcube_rgb_unet_n100_s1 val 2
```

结果落在 `runs/<run>/eval/<split>_final.json`，日志在 `logs/eval/`。

批量（顺序评估多个 run，避免同卡并发）：

```bash
mkdir -p logs/eval     # 注意：重定向目标目录必须先存在，否则 nohup 那条命令会静默失败
nohup bash eval_queue_pickcube.sh \
  pickcube_rgb_unet_n100_s1 pickcube_rgb_unet_n100_s2 pickcube_rgb_unet_n100_s3 \
  > logs/eval/eval_queue_unet.log 2>&1 &
```

队列会把每个 run 的 `rc / 耗时 / test: success...` 汇总在 `logs/eval/eval_queue_<时间戳>.log`。
单个 run 约 40 分钟，建议串行。（2026-10-06 15:20 已用此命令启动 unet 三个 run 的评估。）

### 7.3 实测速度与对训练的干扰

| 项 | 实测 |
| --- | --- |
| 启动开销 | 约 25 s（import + 建环境） |
| 2 回合 / num_envs=2 | 86 s（即约 30 s/回合） |
| 100 回合推算 | **约 30–50 分钟/checkpoint**（num_envs=4，机器有 128 核、1TB 内存，可再加并行） |
| 对训练的干扰 | 推理走卡 7 时会与训练抢卡；用 `DEVICE=cpu` 则卡占用为零 |

> 口径提醒（与 `report/handoff_lusen_n100.md` 对齐才可比）：**`success_once`**、测试种子 **10000–10099**、
> 100 回合、`physx_cpu`、只用 `final.pt`。别混用 `success_at_end`，否则数字系统性偏低、不能直接相减。

### 7.4 评估计划（2026-10-05 与本人确认）

1. **10-06 中午：先评 `pickcube_rgb_unet_n100_s1`**（约 00:40 训练完成）。
   目的：尽早验证"摩尔这条链路复现出的 UNet@N100 是否与集群 0.410 ± 0.029（5 seeds）同量级"，
   偏差大就要在投入剩余 30 小时前排查（数据/口径/渲染差异）。
   命令：`bash run_eval_pickcube.sh pickcube_rgb_unet_n100_s1`（占卡约 40 分钟，训练会略慢）。
2. **10-07 凌晨全部训练结束后：一次性评完其余 8 个 checkpoint**（卡 7 空闲，速度最快）。
3. 汇总时按种子取均值 ± SD 报告，单 run 的 100 回合精度只有约 ±0.08，不要用单 run 下结论。

## 8. 结果（摩尔 MUSA 复现）

### 8.1 训练

9 个 run 全部正常结束，无失败（`final.pt` + `summary.json` 齐全）：

| backbone | s1 完成 | s2 完成 | s3 完成 |
| --- | --- | --- | --- |
| unet | 10-06 00:38 | 10-06 04:56 | 10-06 09:14 |
| transformer | 10-06 12:30 | 10-06 15:52 | 10-06 19:20 |
| mlp | 10-06 21:22 | 10-06 23:24 | 10-07 01:26 |

### 8.2 闭环评估（test，seeds 10000–10099，100 回合，success_once，physx_cpu，final.pt）

| backbone | s1 | s2 | s3 | 摩尔均值 ± SD（3 seeds） | 集群对照（PickCube N=100，5 seeds） | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.790 | 0.800 | 0.810 | **0.800 ± 0.010** | 0.796 ± 0.019 | ✅ 复现 |
| unet | 0.130 | 0.120 | 0.210 | **0.153 ± 0.049** | 0.192 ± 0.018（s1–s3 = 0.180） | ✅ 同量级 |
| transformer | 0.910 | 0.900 | 0.900 | **0.903 ± 0.006** | 0.906 ± 0.041（s1–s3 = 0.930） | ✅ **复现**（10-07 重训后） |

> transformer 一列为**手动注意力补丁重训后**的结果（见 §8.5）。原 fp32 SDPA 版本三个种子均为 0.000，已作废归档到
> `runs_bad_transformer_sdpa_bug_20261007/`。

`suc_end` 对照（摩尔 vs 集群 s1–s3）：transformer 0.820/0.750/0.780 vs 0.85/0.76/0.84；
mlp 0.66/0.73/0.75 vs 0.74/0.68/0.69；unet 0.07/0.06/0.14 vs 0.13/0.13/0.12。

`return` 对照（摩尔 s1/s2/s3 vs 集群 s1–s3）：mlp 19.99/20.72/20.58 vs 19.83/19.62/20.06（集群均值 19.69）；
unet 2.79/3.11/5.01 vs 4.28/3.85/4.11；transformer **25.85/24.38/24.91 vs 21.98/21.61/24.53**（摩尔反而略高）。

结论：**三个主干全部复现成功**。mlp 逐点几乎重合；transformer 均值 0.903 vs 集群 0.906（5 seeds），
差异远小于种子波动；unet 差 −0.027。

### 8.2.1 4080 vs S4000 的系统性偏差有多大（2026-10-07）

把同一训练种子在两侧的 `success_once` 相减（S4000 − 4080，seeds 1–3）：

| 主干 | 4080 均值 | S4000 均值 | 差值 | 配对 t |
| --- | --- | --- | --- | --- |
| transformer | 0.930 | 0.903 | **−0.027** | −4.00 |
| unet | 0.180 | 0.153 | −0.027 | −1.11 |
| mlp | 0.807 | 0.800 | −0.007 | −0.76 |
| **9 对合并** | | | **−0.020**（7/9 为负） | −2.40（保守按 backbone 聚合 n=3：t=−3.00） |

判读：

1. **存在一个很小的系统性偏低，量级约 2 个百分点。** transformer 最能说明问题——它成功率高、种子间方差极小，
   三个种子的差值几乎完全一致（−0.04 / −0.02 / −0.02），配对 t = −4.0，不可能是纯噪声。
2. **但它不改结论。** 两次独立评估（各 100 回合）差值的 95% 噪声带约 **±0.11**，单 run Wilson 半宽约 ±0.078；
   2 个百分点远小于任何单个数字的不确定度，也和 unet 的种子波动（0.12–0.21）没法比。
3. **推论**：S4000 侧可能在渲染（lavapipe 软渲染）、物理计时精度或 MUSA 数值上有一点细微差异，
   导致 "抓到但不稳" 的回合略多——注意 `suc_end` 的差值（unet −0.037、transformer −0.033）系统性大于
   `suc_once`（−0.027），方向一致。
4. **本机无法验证到这个精度**：单 run 100 回合的统计精度决定了，要分辨 2 个百分点需要补 seed 4/5
   （集群有 s1–s5）凑 n=5 配对，或用两层 bootstrap（`~/analyze_stackcube.py` 口径）。

### 8.3 transformer 训练失败：MUSA 后端数值问题

**现象**：transformer 三个种子的训练 loss 从第 5k 步起就卡在 **1.47–1.55** 不再下降（unet 收敛到 0.0015、mlp 到 0.016），
即模型完全没学到东西（loss > 1.0 意味着比"直接预测零噪声"还差），闭环成功率必然为 0。

**根因（已定位）**：

1. `dp_manip/trainer.py:318` 是 `use_amp = cfg.train.amp and device.type == "cuda"`；
   在摩尔上 `torch.device("cuda").type` 经 torchada 重定向后是 **`"musa"`**（日志里那行 `...; musa` 就是 device.type），
   于是 **AMP 被静默关闭**，transformer 全程按 fp32 训练。
2. 训练日志里正好有 torch_musa 关于 `scaled_dot_product_attention` 的告警（`sdp_utils_cpp.h:91`）。
3. 组内另一位同学在 `plug5/attempts.md` 里已经独立定位过同一台机器上的同一问题：
   > MUSA 上 transformer 的 fp32 SDPA 反向数值错误 → 梯度偶发爆炸到 1e6~1e11 → grad_clip=1.0 把模型锁死（loss 卡在 1.0+ 且回升）。
   > 实测对比：fp32 训练 ep1=0.45→ep2=0.97→崩；**bf16 autocast 训练 ep1=0.17→ep2=0.09**（gmax=2，健康）。
   > unet/mlp 无注意力不受影响，可 fp32。

**这解释了为什么只有 transformer 崩**：它的三个 arm 里只有它用注意力（`nn.MultiheadAttention`），
unet/mlp 不含 SDPA，所以 fp32 下不受影响 —— 与"unet/mlp 复现成功、transformer 为 0"的观测完全一致。

**待定方案**（需与本人确认后再动手）：

- A. 把 `TransformerForDiffusion` 里的注意力改成显式实现（`softmax(q@kᵀ/√d)@v`，绕开 MUSA 的 SDPA 内核）后重训 transformer ×3。
- B. 让 trainer 在 MUSA 上也能启用 AMP（`device.type in ("cuda", "musa")`）并用 bf16 重训 transformer ×3。
  注意：本机快测里 `torch.amp.autocast("cuda", dtype=bfloat16)` 在 MUSA 上看起来是 no-op（fp32/bf16 数值完全一致），
  选 B 之前必须先验证 autocast 真的生效。
- C. transformer 一列直接用集群数字，摩尔侧只在报告里注明"MUSA 后端无法复现 transformer"。

无论选 A/B，重训前都要先处理 `runs/pickcube_rgb_transformer_n100_s*` 里的 `final.pt`（否则 trainer 判 CONFLICT 直接退出）。

### 8.4 评估跑在哪些卡上（2026-10-07）

| 卡 | 状态 | 谁在用 |
| --- | --- | --- |
| 0 | 有进程占显存、利用率低 | 队友 |
| 1 | ≈90% | 队友训练 |
| 2 / 3 / 4 / 5 / 6 | **空闲**（0%、0MiB） | — |
| 7 | ≈90% | 原定给我的卡；15:34 起队友也在上面跑 stackcube 评估 |

处理方式：**正在卡 7 上跑的评估不打断**（自然结束），把其余 4 个 run 改到 **卡 5** 上跑（`GPU_INDEX=5 PARALLEL=2`）。
切换前先 `kill` 掉旧队列的包装进程（只停自己的 `eval_queue_pickcube.sh`，不动别人的 `eval_dp.py`）。

> 选卡经验：`mthreads-gmi` 的进程表会把同一个 PID 列在多张卡下（非归属卡显示 `<1MiB`），
> **判断某卡是否空闲要看该卡的 "0MiB/49152MiB" 和利用率**，别被进程表里的 `<1MiB` 条目误导。

### 8.5 transformer 重训：改用手动注意力（方案 A，2026-10-07）

本人 2026-10-07 选定**方案 A（改注意力实现后重训，推荐）**，不用方案 B（本机快测显示 MUSA 上
`autocast("cuda", bf16)` 数值与 fp32 完全一致，疑似 no-op，无法确认 bf16 真的生效，风险高）。

#### 改动：`dp_manip/backbones/transformer.py`

在原文件基础上新增（**不改任何原有层/超参**）：

- `import importlib.util, math, os` + `import torch.nn.functional as F`
- `_manual_attention_enabled()`：默认 auto —— 检测到 torchada / torch_musa 即启用；可用环境变量
  `DP_TRANSFORMER_ATTN=manual|sdpa` 强制开关。
- `class ManualMultiheadAttention(nn.Module)`：显式 `softmax(q@kᵀ/√d)@v`，**与原 `nn.MultiheadAttention`
  共享同一组 Parameter 对象**（in_proj_weight / in_proj_bias / out_proj.*），因此 state_dict 完全兼容。
- `install_manual_attention_if_needed()`：在 `TransformerForDiffusion.__init__` 末尾调用，把
  `nn.TransformerEncoderLayer` / `TransformerDecoderLayer` 里的 `.self_attn` / `.multihead_attn` 换成手动实现。
- `_init_weights` 的 `ignore_types` 里加入 `ManualMultiheadAttention`（避免按其内部参数重复初始化）。

备份：`dp_manip/backbones/transformer.py.orig-preManualAttn`（回滚用）。

#### 换实现前后的数值等价性自检（全部通过）

| 检查项 | 结果 |
| --- | --- |
| `state_dict` 兼容性 | missing=0, unexpected=0 |
| 注意力层被替换 | 4 层 |
| forward 最大绝对误差（fp32，同一权重/输入） | 4.172e-07 |
| backward 梯度最大绝对误差 | 1.490e-07 |
| SMOKE（mlp 与 transformer，2000 步） | transformer loss 1.11 → 0.09（**修复前一直卡在 ~1.48**） |

#### 重训前清理

原来的 3 个失败 run 目录是容器内 `root` 创建的，宿主 `group11` 无权限 `mv`。
改用**容器内 root** 归档：`docker exec dp bash -lc 'mv runs/pickcube_rgb_transformer_n100_s{1,2,3} runs_bad_transformer_sdpa_bug_20261007/'`。
原 `logs/backbone/pickcube_rgb_transformer_n100_s*.log` 改名 `.skipped.log` 留档。
（脚本：`.workbuddy/archive_bad_transformer.sh`。）

> ⚠️ **关键坑**：`run_backbone_pickcube.sh` 调用的 `train_dp.py --resume auto` 一旦发现 run 目录里
> 有 `final.pt`，会打印 `final checkpoint already exists; nothing to do` **直接退出**（launcher 秒退，
> 日志里的"全部 run 正常结束"是假象）。所以**重训前必须先清掉 run 目录**，否则白等。

#### 启动（脚本 `.workbuddy/launch_transformer_retrain.sh`）

```bash
# 容器内，3 个 seed 分别跑卡 4 / 5 / 6，单卡 CONCURRENCY=1
GPU_INDEX=4 BACKBONES=transformer SEEDS=1 CONCURRENCY=1 TASK=pickcube NUM_DEMOS=100 \
  DATA_ROOT=/workspace/code/maniskill-demogen/data/dataset nohup bash run_backbone_pickcube.sh \
  > logs/backbone/launcher_transformer_s1_card4.log 2>&1 &
# s2→卡5、s3→卡6 同理
```

| 项 | 值 |
| --- | --- |
| 启动时刻（服务器时间） | 2026-10-07 16:52 |
| 卡位 | s1→卡 4、s2→卡 5、s3→卡 6（三张都 83–95% 利用率、各约 2.8GB） |
| 启动后实测 | s1 loss 1.139（step 1）→ 0.172（step 600，91s）≈ **6.6 步/秒** |
| 单 run 预计 | ≈ 4.2 h，三卡并行 ⇒ **全部约 4.2 h**（预计 10-07 21:00 前后完成） |

#### 结果（10-07 20:30 训练完成，21:52 评估完成）

| 项 | 结果 |
| --- | --- |
| 训练 | 三个 seed 全部 `finished`，耗时 **3.70 / 3.70 / 3.73 h**（16:48 启动 → 20:30 落盘 `final.pt`） |
| 训练 loss | 收敛到 0.020–0.035（修复前死卡 1.48） |
| 闭环评估 | **suc_once 0.910 / 0.900 / 0.900 → 0.903 ± 0.006**；`suc_end` 0.820/0.750/0.780；`return` 25.85/24.38/24.91 |
| 集群对照 | 0.906 ± 0.041（5 seeds），s1–s3 子集 = 0.930 |
| 评估耗时 | 1382 / 1420 / 1577 s（≈23–26 min，三卡并行，同时受队友 CPU 评估干扰） |
| 判定 | ✅ **复现成功**，且摩尔侧种子间方差比集群小得多 |

结论：手动注意力补丁彻底解决问题——transformer 从 0.000 恢复到 0.903，与集群 0.906 实质上一致。

#### 补丁的适用范围与遗留风险（**重要**）

1. **修的是工程代码，不是"环境"。** 容器全程未改动（未重建、未改依赖）。根因在 torch_musa 的融合
   `scaled_dot_product_attention` 内核里，我们无法修改后端本身，只能在**模型代码层绕开**它。
2. **自动探测，跨平台安全。** `_manual_attention_enabled()` 默认 `auto`：检测到 `torchada` / `torch_musa`
   才启用手动注意力；在 CUDA 机器上自动返回 False → 走原生 SDPA，**不会影响集群那套**。
   可用 `DP_TRANSFORMER_ATTN=manual|sdpa` 强制覆盖。
3. **训练与评估都受益**：评估同样从 `dp_manip` 构造 policy，所以 eval 侧走的是同一份手动注意力（已用 0.903 验证）。
4. ⚠️ **AMP 那个坑没有修，只是绕开了。** `trainer.py:318` 仍是 `use_amp = cfg.train.amp and device.type == "cuda"`，
   摩尔上 `device.type == "musa"` → AMP 依旧静默关闭。unet / mlp 在 fp32 下本来就正常，所以不受影响；
   但**以后若新增含注意力的主干，会撞上同一个 SDPA 数值 bug**，除非它的代码路径也走手动注意力。
   （不建议简单地把 `device.type in ("cuda","musa")` 改掉——本机快测显示 MUSA 上 `autocast` 疑似 no-op，
   开了也未必生效，反而增加不确定性。）
5. **补丁只存在于这一份共享拷贝**（宿主 `~/7606C/dp-manip` = 容器 `/workspace/code/dp-manip`）。
   用这份代码在摩尔上跑都自动生效；队友若有**另一份 checkout**（或集群那份），不会继承此补丁——集群不需要（CUDA 无此 bug）。
6. §8.2.1 里那约 2 个百分点的平台系统性偏差**与本补丁无关**，是另一件事（评估渲染/数值细节层面）。



## 9. 收尾事项

- SMOKE 产物 `runs_smoke/pickcube_rgb_mlp_n100_s1/` 是验证件，确认正式 run 正常后可删（可选）。
- 数据副本 `~/7606C/maniskill-demogen/data/dataset/{train,val}/PickCube-v1`（约 178MB）保留，后续评估/复跑还要用。
- 训练完成后，闭环评估仍需 ManiSkill 环境，且必须固定 `physx_cpu`、测试种子 10000–10099、指标 `success_once`，才能与已有结果比较。
- 本记录未改动容器环境、未重建容器、未触碰卡 0–3。
- 2026-10-07 16:47 起，卡 4 / 5 / 6 被本任务占用（transformer 重训 + 评估）；21:55 评估结束后已**释放**。
- transformer 手动注意力补丁是**唯一**改过的工程代码（+ 备份 `.orig-preManualAttn`）；unet / mlp 路径未受影响。
- 作废产物：`runs_bad_transformer_sdpa_bug_20261007/`（3 个 fp32 SDPA 失败 run，含各自的 eval），
  以及 `logs/backbone/pickcube_rgb_transformer_n100_s*.skipped.log`。确认无用后可删。

