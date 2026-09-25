# 学院 GPU 装环境：uv 教程

> 适用：学院 GPU 机器（Linux, RTX 4080）。整套环境只有两步：**① 装 uv → ② 装库**，装完就能跑 `train/` 下的训练与评测。
>
> 本项目只依赖 6 个第三方库：`torch` / `numpy` / `h5py` / `gymnasium` / `sapien` / `mani_skill`
> （外加可选的 `matplotlib`，仅用来画 loss 曲线）。
> **不需要** diffusers / wandb / tensorboard / Vulkan / CUDA Toolkit / mplib —— 这些统统不用装。

---

## 第 1 步：装 uv（一次性）

装到个人目录，不需要 sudo：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env      # 让 PATH 生效；或者直接重开一个 shell
uv --version
```

如果 curl 被墙或机器没有外网，就用 pip 装到用户目录：

```bash
python3 -m pip install --user uv -i https://pypi.tuna.tsinghua.edu.cn/simple
export PATH="$HOME/.local/bin:$PATH"     # 建议把这一行写进 ~/.bashrc
uv --version
```

---

## 第 2 步：装库

仓库根目录里已经有一份 `pyproject.toml`，依赖版本和 torch 的下载源（PyTorch 官方 cu126）都写好了。
把仓库放到机器上，在**仓库根目录**跑一条命令：

```bash
cd ~/VariDP      # 换成你实际放仓库的目录
uv sync
```

`uv sync` 会自动做三件事：装一个符合 `requires-python` 的 Python → 在仓库根建 `.venv` →
装齐所有依赖并生成 `uv.lock`（全组共用这份 lock，环境可复现）。

> 仓库里的 `.python-version` 把解释器钉在 **3.12**，别往上改：mani_skill 3.0.1 在 Linux 上
> 依赖 `mplib==0.1.1`，而 mplib 只发了 cp310–cp312 的 manylinux 轮子、没有源码包，用
> 3.13 / 3.14 会直接报 `can't be installed because it doesn't have a source distribution or
> wheel for the current platform`（我们不用 mplib 做规划，但它装不上整个 sync 就失败）。

> 之后跑本项目的命令都加 `uv run` 前缀，比如 `uv run python train/train.py ...`。
> 它会自动使用这个 `.venv`，不用手动 activate。

机器上 `nvidia-smi` 显示 Driver 580 / CUDA 13.0，那只是**驱动**支持的版本上限；
装 cu126 的 PyTorch 轮子即可，**不用另外装 CUDA Toolkit**。

装库时常见的两种情况：

| 情况 | 处理 |
|---|---|
| 下载慢、超时 | `export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"` 再 `uv sync`（旧版 uv 用 `UV_INDEX_URL`）。torch 的 cu126 源不在镜像里，保持 `pyproject.toml` 里那个 URL 不动 |
| 提示 `torch==2.14.0+cu126` 在 Linux 没有对应轮子 | 把 `pyproject.toml` 里 torch 那行改成 `"torch"`（去掉版本号）再 `uv sync`，uv 会自动选最新的 cu126 版；实际装到哪个版本以 `uv.lock` 为准 |

集群用 SLURM 的话，先 `srun` / `salloc` 申请到交互节点，确认 `nvidia-smi` 能看到卡再装。
网络特别差、或家目录在 NFS 上导致装得特别慢，见附录 A。

---

## 第 3 步：验证

```bash
cd ~/VariDP

# ① GPU 可用？
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.14.0+cu126 True NVIDIA GeForce RTX 4080 ...

# ② 依赖齐了？
uv run python -c "import mani_skill, sapien, gymnasium, h5py; print(mani_skill.__version__, sapien.__version__, gymnasium.__version__, h5py.__version__)"
# 期望：3.0.1 3.0.3 1.3.0 3.16.0

# ③ dp 库能导入？
uv run python -c "import dp; print('dp 库可导入')"

# ④ 真正的冒烟：能开始训练、loss 在降、结束后 train/runs/ 里出现 loss_curve.png
uv run python train/train.py --env-id PickCube-v1 --backbone unet --total-iters 3000 --max-episodes 50
```

训练脚本会自动把仓库根挂到 `sys.path`，所以 `dp` 不必 `pip install -e .` 也能导入。

---

## 附录 A：装得慢 / 网络差怎么办

uv 默认就是在项目目录下建 `.venv`，慢通常不是 uv 的问题，而是两件事：
① torch 那个 ~2.5 GB 的大轮子；② **家目录在 NFS 网络盘上**（集群标配，缓存和解压 I/O 都很慢）。

先判断慢在哪：

```bash
df -hT ~        # 家目录是 nfs/fuse 类型 → 磁盘 I/O 拖慢的
uv sync -v      # 卡在 Downloading = 网络问题；卡在 Installing / Linking = 磁盘问题
```

**招 1：把缓存和 venv 挪到本地盘**（家目录是 NFS 时收益最大）

```bash
export UV_CACHE_DIR=/tmp/$USER/uv-cache              # 有 /scratch、/data 就用那些，更持久
export UV_PROJECT_ENVIRONMENT=/tmp/$USER/VariDP/.venv
uv sync
```

注意 `/tmp` 重启会清空；venv 挪走后激活路径变成 `/tmp/$USER/VariDP/.venv/bin/activate`。

**招 2：PyPI 换国内镜像**

```bash
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
```

**招 3：torch 大文件单独处理**

- uv 有缓存，断了重跑 `uv sync` 会续传，不用从头来；
- 或者在带宽好的机器上先把 wheel 下下来，再 scp 过去离线装：

```bash
pip download torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126 -d wheels/
# 传到集群后：
uv pip install --find-links wheels/ torch==2.14.0+cu126
```

**招 4：看看集群有没有现成模块**：`module avail | grep -i torch`。
有的话能省掉整个下载，但版本未必和本机对齐，报告里要写清楚实际用的版本。

## 附录 B：不用 uv——原生 venv + pip

等价可行，只是没有锁文件，用 `requirements.txt` 承担「全组统一 / 报告可复现」的角色：

```bash
cd ~/VariDP
python3 --version            # 必须是 3.12（mplib 只有 cp310–cp312 轮子）；不对就先 uv python install 3.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 顺序重要：先钉死 torch，再装 mani_skill，免得它拉一个别的 torch 版本
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install mani_skill==3.0.1 numpy==2.5.3 h5py==3.16.0 gymnasium==1.3.0 matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

pip freeze > requirements.txt
```

两点说明：

1. pip 不会更快——依赖解析和下载通常比 uv 还慢，快慢主要取决于网络和磁盘，不是工具。
2. 同样别把 `.venv` 建在 NFS 家目录上（理由见附录 A）。

## 附录 C：和本机 Windows 环境的差异

| 项 | 本机 Windows | 学院 GPU（Linux） | 说明 |
|---|---|---|---|
| pinocchio | 手动移植（conda-forge 拷包） | **什么都不用做** | sapien 的 Linux 轮子自带；`replay_trajectory` / `pd_ee_delta_pos` 直接能用 |
| Vulkan / 渲染 | 没有 → 不出视频 | 也**不需要装** | 本项目训练 + 数值评测不渲染；`train.py` 画图已强制 `Agg` 后端 |
| `physx_cuda` 并行仿真 | 不可用（缺 CUDA Toolkit） | **可能可用** | Linux 上 sapien 自带 PhysX CUDA 库，通常只要驱动。收益：评测可开 `--num-envs 32`，50 ep 从约 1 分钟降到几秒 |
| 评测后端一致性 | 全部 `physx_cpu` | 混用会毁掉公平性 | **同一组实验必须同一后端**。农场用 `train/eval.py --num-envs 32 --sim-backend gpu`（本机 `train_local/eval.py` 固定 `physx_cpu` 单环境），且所有对比都要重跑成同一后端 |

> 结论：训练随便用 GPU 农场（这是它最大的价值）；**评测仍建议用 `physx_cpu`**，单环境串行也就几十秒，
> 和已有结果直接可比。要么全 cpu、要么全 gpu，别混。

## 附录 D：常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `uv: command not found` | 装完 PATH 没刷新 | `source ~/.local/bin/env`，或重开 shell |
| 下载 torch 超时 / 反复 retry | 到 `download.pytorch.org` 网络差 | 重试；或在本机下好 wheel 再 `uv pip install <whl文件>`（附录 A 招 3） |
| `import sapien` 报 libvulkan 缺失 | 少系统库（少见，不渲染一般碰不到） | `sudo apt install libvulkan1`；无 sudo 就 `mamba install -c conda-forge vulkan` 或找管理员 |
| `ModuleNotFoundError: mani_skill` | 没在 `.venv` 里跑 | 用 `uv run python ...`（推荐），或先 `source .venv/bin/activate` |
| `没找到 xxx 的 state 数据集` | 数据没传 / 路径不对 | 确认 `~/.maniskill/demos/<Task>-v1/**/*.state.pd_ee_delta_pos.physx_cpu.h5` 存在 |
| 家目录配额爆了 | torch + CUDA 轮子约 7 GB | `df -h ~` 看配额；`uv cache clean` 清缓存；或按附录 A 招 1 把 `.venv` 建到大盘上 |
| `uv sync` 报 `mplib==0.1.1 ... doesn't have a source distribution or wheel for the current platform` | uv 自动挑的解释器比 3.12 新（3.13/3.14），mplib 没有对应轮子 | 仓库里 `.python-version` 已是 3.12，确认没被改掉；然后 `uv python install 3.12` + `uv sync` |
| 改了项目目录名后 activate / pinocchio 报错 | venv 里存了创建时的绝对路径 | 跑 `python tools/fix_venv_paths.py --apply` 一键修（脚本在仓库里，只用标准库） |
| 训练比本机还慢 | 被分到共享节点 / CPU 核被限 | `nvidia-smi` 确认卡空闲；数据加载用 `--num-workers`（如支持） |

## 附录 E：版本对齐表（本机实测可用的一套，直接照抄）

| 包 | 版本 | 备注 |
|---|---|---|
| Python | **3.12（本机 / 农场都用这个）** | 不能更高：mani_skill 3.0.1 在 Linux 依赖的 `mplib==0.1.1` 只有 cp310–cp312 轮子（sapien 3.0.3 本身覆盖 cp310–cp314，mplib 才是瓶颈）。`uv python install 3.12`，仓库里的 `.python-version` 已钉好 |
| torch | 2.14.0+cu126 | cu126 在 4080 上跑满，驱动 580 兼容 |
| numpy | 2.5.3 | |
| h5py | 3.16.0 | |
| gymnasium | 1.3.0 | |
| sapien | 3.0.3 | |
| mani_skill | 3.0.1 | |
| matplotlib | 任意最新 | 仅画 loss 曲线，已用 Agg 后端 |

> 报告里「computing resources」一节可直接写：训练在 RTX 4080（16 GB，driver 580.178.04）上完成，
> 本机 RTX 4060 Ti（8 GB）做冒烟；软件版本如上表，环境用 uv 锁定（`uv.lock`）可复现。
