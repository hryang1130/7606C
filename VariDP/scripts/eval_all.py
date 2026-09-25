"""eval_all.py — 批量评测多个 checkpoint（逐个调用本仓库的 eval 入口脚本）。

它会扫 `<runs-dir>/<pattern>/best.pt`，对每个 checkpoint 调一次
`train_local/eval.py`（或 `train/eval.py`），评测结果 JSON 由 eval 脚本写在 checkpoint 同目录；
**汇总报告交给 `tools/summarize_runs.py`**（自动出 5 节 Markdown，不用手写）。

用法（在仓库根目录）：
  python scripts/eval_all.py                                     # 扫 train_local/runs/*/best.pt，50 episode
  python scripts/eval_all.py -n 20 --pattern "*_unet_*"          # 只评 unet 那组，少跑几个 episode
  python scripts/eval_all.py --runs-dir train/runs --eval-script train/eval.py --num-envs 32 --sim-backend gpu
  python scripts/eval_all.py --dry-run                           # 只打印命令

评完汇总：python tools/summarize_runs.py --runs-dir <runs-dir> --csv
"""
import argparse
import glob
import os
import os.path as osp
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def supports_flag(script: str, flag: str) -> bool:
    """eval 脚本是否支持某个参数（train_local/eval.py 没有 --num-envs / --sim-backend）。"""
    try:
        return flag in open(script, encoding="utf-8").read()
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="批量评测 runs/ 下的 checkpoint")
    ap.add_argument("--runs-dir", default=osp.join(ROOT, "train_local", "runs"))
    ap.add_argument("--pattern", default="*", help="实验目录名通配，如 '*_unet_*'")
    ap.add_argument("-n", "--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=2000, help="held-out 评测起始 seed")
    ap.add_argument("--eval-script", default=osp.join("train_local", "eval.py"),
                    help="评测入口脚本（相对仓库根）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--num-envs", type=int, default=1, help="仅 train/eval.py 支持")
    ap.add_argument("--sim-backend", default="auto", help="仅 train/eval.py 支持")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    runs_dir = osp.abspath(a.runs_dir)
    eval_script = osp.join(ROOT, a.eval_script)
    if not osp.exists(eval_script):
        print(f"[error] 找不到评测脚本 {eval_script}")
        return 1

    ckpts = sorted(glob.glob(osp.join(runs_dir, a.pattern, "best.pt")))
    if not ckpts:
        print(f"[error] {runs_dir}/{a.pattern}/best.pt 一个都没找到")
        return 1

    print(f"[eval-all] {len(ckpts)} 个 checkpoint | {a.episodes} episode | seeds "
          f"{a.seed0}..{a.seed0 + a.episodes - 1} | 入口 {osp.relpath(eval_script, ROOT)}")
    for ck in ckpts:
        cmd = [sys.executable, eval_script, "--ckpt", ck, "-n", str(a.episodes),
               "--seed0", str(a.seed0), "--device", a.device, "--quiet"]
        if supports_flag(eval_script, "--num-envs"):
            cmd += ["--num-envs", str(a.num_envs), "--sim-backend", a.sim_backend]
        print("[run]", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=False, cwd=ROOT)

    print(f"\n[done] 汇总：python tools/summarize_runs.py --runs-dir "
          f"{osp.relpath(runs_dir, ROOT)} --csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
