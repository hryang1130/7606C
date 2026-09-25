"""
run_local.py — 一键流程：训练 -> 评测 -> 出汇总（把 train.py / eval.py 当子进程调用）

结果写成 `<train-dir>/runs/SUMMARY_<env>.md`，可直接贴进报告。

用法（在仓库任意位置都能跑）：
  python tools/run_local.py                                    # 默认 PickCube-v1 + mlp
  python tools/run_local.py --backbone unet --epochs 300
  python tools/run_local.py --backbone transformer --data-efficiency
  python tools/run_local.py --train-dir train --device cuda     # 换成学院 GPU 那套入口脚本

说明：`--train-dir` 默认指向 `train_local/`（本机 Windows 那套脚本）；脚本本身在 `tools/`，
所以两套机器共用同一份流程代码。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys

import subprocess

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, ".."))
PY = sys.executable


def run(cmd, cwd=HERE):
    print("\n" + "=" * 78)
    print("[cmd]", " ".join(str(c) for c in cmd))
    print("=" * 78, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        print(f"[warn] 子进程返回码 {r.returncode}（继续）", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="本地一键训练+评测流程")
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-frac", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--eval-seed0", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backbone", default="mlp", choices=["mlp", "unet", "transformer"],
                    help="噪声预测主干（结构对比实验的唯一变量）")
    ap.add_argument("--train-dir", default=osp.join(ROOT, "train_local"),
                    help="放 train.py / eval.py 的目录（默认 train_local，可指向 train）")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--data-efficiency", action="store_true",
                    help="跑 1.0/0.5/0.25/0.1 四档数据量，出数据效率曲线")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    a = ap.parse_args()

    train_dir = osp.abspath(a.train_dir)
    if not osp.isdir(train_dir):
        print(f"[error] --train-dir 不存在：{train_dir}")
        return
    for fn in ("train.py", "eval.py"):
        if not osp.exists(osp.join(train_dir, fn)):
            print(f"[error] {osp.join(train_dir, fn)} 不存在")
            return

    stages = []

    if a.data_efficiency:
        fracs = [1.0, 0.5, 0.25, 0.1]
    else:
        fracs = [a.demo_frac]

    all_rows = []
    for frac in fracs:
        # 目录名必须和 train.py 的默认命名一致（带主干），否则后面汇总找不到产物
        exp = f"{a.env_id}_frac{frac}_{a.backbone}_seed{a.seed}"
        out_dir = osp.join(train_dir, "runs", exp)

        cmd = [PY, osp.join(train_dir, "train.py"), "--env-id", a.env_id,
               "--backbone", a.backbone, "--epochs", a.epochs,
               "--batch", a.batch, "--seed", a.seed, "--demo-frac", frac,
               "--device", a.device]
        if a.dry_run:
            print("[dry-run]", " ".join(str(c) for c in cmd)); continue
        run(cmd, cwd=train_dir)

        ckpt = osp.join(out_dir, "best.pt")
        if a.skip_eval or not osp.exists(ckpt):
            continue
        run([PY, osp.join(train_dir, "eval.py"), "--ckpt", ckpt, "-n", a.eval_episodes,
             "--seed0", a.eval_seed0, "--device", a.device, "--quiet"], cwd=train_dir)

        # 收集结果
        ev = glob.glob(osp.join(out_dir, f"eval_seed{a.eval_seed0}_n{a.eval_episodes}*.json"))
        if ev:
            d = json.load(open(ev[0], encoding="utf-8"))
            s = json.load(open(osp.join(out_dir, "train_summary.json"), encoding="utf-8"))
            all_rows.append(dict(demo_frac=frac, n_train_episodes=s["cfg"]["n_train_episodes"],
                                 success_rate=d["success_rate"], n_success=d["n_success"],
                                 n_episodes=d["n_episodes"],
                                 avg_steps=round(sum(r["steps"] for r in d["results"])
                                                 / len(d["results"]), 1),
                                 best_val=s.get("best_score"),
                                 train_seconds=s.get("seconds")))

    if a.dry_run or not all_rows:
        print("\n没有可汇总的结果。")
        return

    # ---------------- 汇总 ----------------
    lines = [f"# {a.env_id} 训练结果（主干 {a.backbone}）", "",
             f"- 入口脚本目录：{train_dir}（评测后端 physx_cpu 单环境）",
             f"- 主干：{a.backbone}，epochs：{a.epochs}，batch：{a.batch}，训练 seed：{a.seed}",
             f"- 评测：{a.eval_episodes} episode，seeds {a.eval_seed0}~{a.eval_seed0+a.eval_episodes-1}（held-out）",
             "",
             "| 演示条数 | demo_frac | 成功率 | 成功/总数 | 平均步数 | best val loss | 训练用时(s) |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(all_rows, key=lambda x: -x["demo_frac"]):
        lines.append(f"| {r['n_train_episodes']} | {r['demo_frac']} | "
                     f"{r['success_rate']*100:.1f}% | {r['n_success']}/{r['n_episodes']} | "
                     f"{r['avg_steps']} | {r['best_val']:.5f} | {r['train_seconds']} |")

    summary = "\n".join(lines) + "\n"
    out_name = ("SUMMARY_data_efficiency.md" if a.data_efficiency
                else f"SUMMARY_{a.env_id}.md")
    path = osp.join(train_dir, "runs", out_name)
    os.makedirs(osp.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(summary)
    print("\n" + summary)
    print(f"[done] 汇总已保存 {path}")


if __name__ == "__main__":
    main()
