"""fetch_ms_baseline.py — 经镜像站抓取 ManiSkill 官方 DP 基线代码（绕过 github 不通）。

官方基线（`examples/baselines/diffusion_policy/`）**不在 pip 包里**，只能从 GitHub 取；
本机 github.com / raw.githubusercontent.com 不通，所以走 jsdelivr / gh-proxy 两个镜像。

用法（在仓库根目录，需要联网）：
  python scripts/fetch_ms_baseline.py                 # 默认下到 third_party/ms_dp_baseline
  python scripts/fetch_ms_baseline.py --out <目录>

说明：本项目自己的实现是 `dp/` + `train_local/`（结构对比实验用），
这个脚本只是为了"需要时能取到官方基线做对照"（它自带 train.py 与 baselines.sh）。
它的 diffusion_policy/conditional_unet1d.py 正是本项目 UNet 的对照来源。
"""
import argparse
import os
import os.path as osp
import urllib.request

MIRRORS = [
    "https://cdn.jsdelivr.net/gh/haosulab/ManiSkill@main/",
    "https://gh-proxy.com/https://raw.githubusercontent.com/haosulab/ManiSkill/main/",
]
PREFIX = "examples/baselines/diffusion_policy"
FILES = [
    "setup.py", "README.md", "baselines.sh", "train.py", "train_rgbd.py",
    "diffusion_policy/conditional_unet1d.py", "diffusion_policy/evaluate.py",
    "diffusion_policy/make_env.py", "diffusion_policy/plain_conv.py",
    "diffusion_policy/utils.py",
]
OUT = "third_party/ms_dp_baseline"

HDR = {"User-Agent": "Mozilla/5.0"}


def grab(rel, dst):
    last = None
    for m in MIRRORS:
        try:
            req = urllib.request.Request(m + f"{PREFIX}/{rel}", headers=HDR)
            data = urllib.request.urlopen(req, timeout=120).read()
            os.makedirs(osp.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
            print(f"  ok  {rel}  ({len(data)} B)")
            return
        except Exception as e:
            last = e
    raise RuntimeError(f"{rel} 全部镜像失败: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 ManiSkill 官方 DP 基线（镜像站）")
    ap.add_argument("--out", default=OUT, help=f"输出目录（默认 {OUT}）")
    a = ap.parse_args()

    for f in FILES:
        grab(f, osp.join(a.out, f))
    # 包内需要 __init__.py
    init = osp.join(a.out, "diffusion_policy", "__init__.py")
    os.makedirs(osp.dirname(init), exist_ok=True)
    open(init, "a").close()
    print(f"\n完成 → {osp.abspath(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
