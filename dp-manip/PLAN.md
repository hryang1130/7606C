# PLAN — Mac 作为唯一真理的多设备工作流

**目标**：把 `/Users/hollins/Documents/Coding/dp-manip`（Mac）变为项目唯一权威仓库；代码/配置/文档经 git 同步，**所有 git 与 rsync 操作都由 Mac 发起**；数据与训练结果经 rsync 传输、不入库，只把 SHA-256 清单入库；保留 ubuntu 与 wsl 现有环境与数据，使其继续可用。

## 决策（已确认）

| 议题            | 决定                                                                                          |
| --------------- | --------------------------------------------------------------------------------------------- |
| 同步方向        | Mac 单向发起：`git push` 到远端，`git fetch` 从远端取回；远端从不主动连 Mac                    |
| Mac 远程登录    | **不开启**。Mac 已能 `ssh ubuntu` / `ssh wsl`，无需反向访问，不增加暴露面                       |
| 远端仓库        | 非 bare，`receive.denyCurrentBranch=updateInstead`（远端工作区脏时 push 会被拒，作为防漂移阀） |
| 远端 URL（Mac） | 用 SSH 别名：`ubuntu:Coding/dp-manip`、`wsl:projects/dp-manip`，不写死 IP                      |
| 数据            | `data/`、`demos-*/` 全部 gitignore；经 rsync 传输；`manifests/*.sha256` 入库做一致性校验        |
| 结果/checkpoint | wsl 产出，rsync 回传 Mac 存档，gitignore                                                       |
| 依赖            | 本轮不改 `pyproject.toml` / `uv.lock`（它们是 wsl 训练环境）；ubuntu 继续用 freeze 文件；**只有 wsl 可以 `uv sync`**（ubuntu 与 Mac 都有各自不同的 `.venv`） |
| 编辑位置        | 默认只在 Mac 改；wsl 允许临时热修并本地提交，由 Mac `fetch` 后 fast-forward 合入；ubuntu 只跑不改 |
| wsl 现有 `.git` | 已核实 0 提交、全部未跟踪；`mv .git .git.bak` 保留，不直接删除                                 |
| 文档口径        | 统一为中文单一套；`progress.md` → `STATUS.md`                                                  |

## 执行记录

| 阶段 | 状态 | 说明 |
| ---- | ---- | ---- |
| 0 | ✅ 9.23 | 备份 ubuntu `~/dp-manip-pre-git-20260923.tgz`（262K，24 项，git 2.53.0）、wsl 同名（280K，54 项，git 2.43.0） |
| 1 | ✅ 9.23 | Mac `git init -b main`，remote `ubuntu` / `wsl` |
| 2 | ✅ 9.23 | 导入代码与文档；数据 rsync 到 Mac，12 个文件按清单校验通过；文档合并；首次提交 |
| 3 | ✅ 9.23 | 两端 `git init` + `updateInstead`；wsl 原 `.git` → `.git.bak`（写入 `.git/info/exclude`）；推到 `incoming` 比对：代码全部 SAME/NEW，仅 wsl 的 `.gitignore` 与三份文档 DIFF（预期）；`reset --hard` 后两端 HEAD `3f69b79`、工作区干净；ubuntu `~/.zshenv` 加 `UV_PROJECT_ENVIRONMENT`；两端清单校验通过，`.venv` 关键包版本不变 |
| 4 | ✅ 9.23 | `scripts/sync.sh`：`status` / `manifest`（重新生成与已提交清单一致）/ `pull-results`（wsl 暂无产出）/ `data`（两段传输后校验通过）均已实测 |
| 5 | ✅ 9.23 | 24 Mac push → 两端 `a7c1f08`；25 wsl 提交 `5e56296` → Mac `fetch` fast-forward → push 给 ubuntu；26 wsl `verify_cuda.py` 通过、ubuntu 关键包版本不变、三端清单 OK、工作区干净；另测 wsl 有未提交改动时 Mac push 被 `updateInstead` 拒绝、ubuntu 照常更新 |

## 现状（9.23 只读核实，执行前）

| 设备   | 路径                                       | git                     | 与本计划相关的事实                                                                 |
| ------ | ------------------------------------------ | ----------------------- | ---------------------------------------------------------------------------------- |
| Mac    | `/Users/hollins/Documents/Coding/dp-manip` | 非仓库                  | README / AGENT / progress / TODO / PLAN / `in.txt`；本地 `.venv`（1.4G，ManiSkill 仿真环境，9.23 00:44 创建）；IP `10.0.0.119`  |
| ubuntu | `~/Coding/dp-manip`                        | 非仓库                  | `run_cpu.py patches/ environment/ mplib-probe-overrides.txt demos-*`（约 770K）+ `.venv`；**无** pyproject / uv.lock / .gitignore |
| wsl    | `~/projects/dp-manip`                      | 有 `.git`，0 提交       | `scripts/ configs/ pyproject.toml uv.lock .python-version README/STATUS/TODO.md .gitignore`；`data/` 676K |

## 目标目录结构（Mac，扁平根）

```
dp-manip/
  README.md AGENT.md STATUS.md TODO.md PLAN.md
  docs/history.md                # 原 in.txt 归档
  pyproject.toml uv.lock .python-version   # wsl 训练环境；ubuntu 不得据此 uv sync
  .gitignore
  scripts/                       # 训练/校验代码（来自 wsl）+ sync.sh
  configs/
  run_cpu.py patches/ environment/ mplib-probe-overrides.txt   # 来自 ubuntu
  manifests/                     # 入库：数据 SHA-256 清单
    ubuntu-demos.sha256          #   相对 ubuntu 项目根，覆盖 demos-*/
    wsl-data.sha256              #   相对 wsl 项目根，覆盖 data/
  demos-batch/ demos-final-verify/ demos-rebuild-verify/       # gitignore；rsync 自 ubuntu
  data/                          # gitignore；rsync 自 wsl
  results/ checkpoints/ logs/    # gitignore；rsync 自 wsl
```

## 前置（用户手动完成）

1. 路由器：为 ubuntu（`.200`）与 wsl（`.248`）做 DHCP 保留（Mac 不再需要）。
2. 确认本计划后，由用户明确授权执行阶段 3（会改写两台远端的工作区）。

不再需要：开启 macOS 远程登录、向 Mac `authorized_keys` 追加远端公钥、Mac 固定 IP。

## 执行步骤

**阶段 0 — 冻结与备份（远端，非破坏）**

1. 通知：从此刻到阶段 3 结束，不在 ubuntu / wsl 上改代码或文档。
2. ubuntu：`tar czf ~/dp-manip-pre-git-$(date +%Y%m%d).tgz -C ~/Coding --exclude=dp-manip/.venv dp-manip`（含 `demos-*`，体积小）。
3. wsl：`tar czf ~/dp-manip-pre-git-$(date +%Y%m%d).tgz -C ~/projects --exclude=dp-manip/.venv --exclude=dp-manip/.python --exclude=dp-manip/.uv-cache dp-manip`。
4. 记录两端 git 版本（`updateInstead` 需 git ≥ 2.4）。

**阶段 1 — Mac 建仓**

5. 写 `.gitignore`：
   ```
   .venv/
   .python/
   .uv-cache/
   __pycache__/
   *.py[cod]
   .pytest_cache/
   .DS_Store
   data/
   demos-*/
   results/
   checkpoints/
   logs/
   ```
6. `git init -b main`；配置仓库级 `user.name` / `user.email`。
7. `git remote add ubuntu ubuntu:Coding/dp-manip`；`git remote add wsl wsl:projects/dp-manip`。

**阶段 2 — 导入并合并**

8. 从 wsl 取代码：`scripts/ configs/ pyproject.toml uv.lock .python-version`；wsl 的 `README.md STATUS.md TODO.md` 取到临时目录供合并。
9. 从 ubuntu 取：`run_cpu.py patches/ environment/ mplib-probe-overrides.txt`。
10. 数据 rsync 到 Mac（不入库）：ubuntu `demos-*/` → Mac；wsl `data/` → Mac。
11. 生成清单：在各远端项目根执行 `sha256sum` 生成 `manifests/ubuntu-demos.sha256`、`manifests/wsl-data.sha256`，拉回 Mac；在 Mac 上用 `shasum -a 256 -c` 核对 Mac 副本，并确认 PickCube 两个文件的哈希与 README 记录一致。
12. 合并文档为中文单一套：
    - `progress.md` 改名 `STATUS.md`，并入 wsl `STATUS.md` 的 M0 状态；
    - `README.md`、`TODO.md` 合并 wsl 版本内容；README / TODO / AGENT 中所有指向 `progress.md` 的链接改为 `STATUS.md`；
    - `in.txt` 整理为 Markdown，归档为 `docs/history.md`，删除原文件；
    - AGENT.md 更新：第 6 条补充 `data/`、`demos-*` 由 rsync + 清单管理；第 7 条改为「按 `sync.sh` 流程提交，push 前需用户确认」；新增硬规则「只有 wsl 可以 `uv sync` / `uv lock`；ubuntu 上也禁止 `uv pip install`，环境只按 `environment/ubuntu-expert-freeze.txt` 与补丁重建；Mac 上执行会替换本地 ManiSkill `.venv`」；新增「默认只在 Mac 编辑」。
13. `git status` 检查无数据文件、无 `.venv` 被跟踪；首次提交。

**阶段 3 — 挂接远端（需用户授权）**

对 ubuntu 与 wsl 分别执行（以 wsl 为例）：

14. 远端准备：
    ```bash
    ssh wsl 'cd ~/projects/dp-manip && mv .git .git.bak && git init -b main \
      && git config receive.denyCurrentBranch updateInstead'
    ```
    （ubuntu 无 `.git`，跳过 `mv`。）
15. Mac 推送到临时分支（避开 `updateInstead` 对未跟踪文件的冲突）：`git push wsl main:refs/heads/incoming`。
16. 覆盖前比对：在远端 `git diff --no-index --stat` 或逐文件 `sha256sum`，确认 `incoming` 中来自该设备的文件与当前工作区一致（仅文档合并产生的差异属预期）。有意外差异则停下。
17. 远端落地：
    ```bash
    ssh wsl 'cd ~/projects/dp-manip && git reset --hard incoming && git branch -D incoming && git status --short'
    ```
    `data/`、`demos-*`、`.venv/` 均被忽略，不会被改写。
18. ubuntu 额外防护：ubuntu 登录 shell 是 zsh，在 `~/.zshenv`（非交互 ssh 也会读取）中 `export UV_PROJECT_ENVIRONMENT="$HOME/Coding/dp-manip/.venv-uv-guard"`，确保误执行 `uv sync` 也不会动到 `.venv`；并把 `.venv-uv-guard/` 加入 `.gitignore`。

**阶段 4 — 同步脚本 `scripts/sync.sh`（仅在 Mac 执行）**

19. `push`：要求 Mac 工作区干净；`git push ubuntu main` 与 `git push wsl main`（远端脏或非 fast-forward 时被拒，脚本如实报错，不强推）。
20. `fetch`：`git fetch wsl`（及 ubuntu）；若远端有新提交，`git merge --ff-only`，不能 fast-forward 则停下提示人工处理。
21. `pull-results`：`rsync -a`（不带 `--delete`）wsl `results/ checkpoints/ logs/` → Mac。
22. `data`：按需 rsync 数据（ubuntu → Mac → wsl），传输后在目标端用 `manifests/*.sha256` 校验；新数据生成后先更新清单再提交。
23. `status`：三端 `git status --short` + `git rev-parse HEAD` 对照 + 各端清单校验结果。

**阶段 5 — 验证**

24. Mac 改一行文档 → `sync.sh push` → 两端 `git log -1` 一致。
25. wsl 本地提交一次 → Mac `sync.sh fetch` → Mac fast-forward 成功，再 `push` 回 ubuntu。
26. 确认两端 `.venv` 未变（ubuntu 抽查 `mani-skill` / `mplib` 版本；wsl 跑一次 `scripts/verify_cuda.py`）、清单校验全通过、三端 `git status` 干净。

**阶段 6 — 延后（不在本轮）**

- 依赖拆分：把 `pyproject.toml` 改为 `train` / `expert` 两个依赖组并重新 lock（会改动 wsl 环境，需单独验证）。
- 旧备份目录与 TODO E 清理；`.git.bak` 与 `dp-manip-pre-git-*.tgz` 在验证稳定一周后再由用户决定删除。

## 风险与对策

- **ubuntu 误执行 `uv sync` 会按 wsl 的 lock 精确同步，卸掉 mani-skill / mplib**（Mac 同理会替换本地 ManiSkill 环境）→ AGENT.md 硬规则 + ubuntu `UV_PROJECT_ENVIRONMENT` 防护 + 阶段 6 依赖拆分。
- `updateInstead` 在远端有未提交改动时拒绝 push → 这是预期的防漂移阀；先在远端提交或丢弃改动，再由 Mac `fetch` / `push`。
- `reset --hard` 会改写远端被跟踪的文件 → 阶段 0 备份 + 阶段 3 覆盖前比对。
- Mac 休眠或离开局域网 → 同步由 Mac 发起，不在线时只是暂不同步，不会出错；离网时 ubuntu 可走 `ubuntu-frp`。
- 数据只靠清单校验、不入库 → 以后数据量增长不影响仓库体积；清单变动随提交留痕。

## 回滚

- 阶段 1–2：只在 Mac 新增文件，删除 `.git`、`manifests/` 等新增内容即可。原 `progress.md` 与 `in.txt` 已删除（未提交前删除，无原件备份）：`progress.md` 全文并入 `STATUS.md`；`in.txt` 内容完整保留在 `docs/history.md`，仅表格排版改为 Markdown。
- 阶段 3：远端 `rm -rf .git && mv .git.bak .git`（wsl），并从 `~/dp-manip-pre-git-*.tgz` 恢复被改写的文件；ubuntu 删除 `.git` 与 `.gitignore` 后按备份恢复。
- 全程不修改任何 `.venv/`；数据目录被忽略，不会被 git 改写。
