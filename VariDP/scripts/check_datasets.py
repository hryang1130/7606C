"""check_datasets.py — 扫描演示数据集做质检，输出 `report.md` + `report.csv`。

检查项（报告里要用，见 `实验三_Track3_完整教程.md` §5.4）：
条数 / obs 维度 / act 维度 / 成功率 / 轨迹长度分布 / 动作光滑性 / 动作取值范围。

用法（在仓库根目录）：
  python scripts/check_datasets.py
  python scripts/check_datasets.py --demo-dir D:\\demos --out-dir runs/_reports
  python scripts/check_datasets.py --patterns "*/*/*.state.pd_ee_delta_pos.physx_cpu.h5"
"""
import argparse
import csv
import glob
import json
import os
import os.path as osp

import h5py
import numpy as np

try:
    from mani_skill import DEMO_DIR
except Exception:
    DEMO_DIR = os.path.expanduser("~/.maniskill/demos")

ROWS = []


def check(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        lengths, succ, gaps, a_lo, a_hi = [], [], [], [], []
        obs_dim, act_dim = None, None
        for k in keys:
            g = f[k]
            a = g["actions"][:]
            o = g["obs"][:]
            obs_dim, act_dim = o.shape[-1], a.shape[-1]
            lengths.append(len(a))
            succ.append(bool(g["success"][-1]) if g["success"].shape else False)
            if len(a) > 1:
                gaps.append(float(np.abs(np.diff(a, axis=0)).mean()))
            a_lo.append(a.min(0)); a_hi.append(a.max(0))
        a_lo, a_hi = np.stack(a_lo).min(0), np.stack(a_hi).max(0)
        meta_path = h5_path.replace(".h5", ".json")
        if osp.exists(meta_path):
            ei = json.load(open(meta_path, encoding="utf-8"))["env_info"]
        else:                                  # 缺 meta 也不影响质检
            stem = osp.basename(h5_path).split(".")
            ei = {"env_id": stem[0], "env_kwargs": {}}
    return dict(
        file=osp.basename(h5_path), env_id=ei["env_id"],
        control_mode=ei["env_kwargs"].get("control_mode"),
        obs_mode=ei["env_kwargs"].get("obs_mode"),
        n_episodes=len(keys), obs_dim=obs_dim, act_dim=act_dim,
        success_rate=float(np.mean(succ)),
        len_mean=float(np.mean(lengths)), len_min=int(np.min(lengths)), len_max=int(np.max(lengths)),
        action_smoothness=float(np.mean(gaps)) if gaps else float("nan"),
        act_min=float(a_lo.min()), act_max=float(a_hi.max()),
        size_mb=round(osp.getsize(h5_path) / 1e6, 1),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="演示数据集质检")
    ap.add_argument("--demo-dir", default=str(DEMO_DIR), help="演示根目录（默认 mani_skill.DEMO_DIR）")
    ap.add_argument("--out-dir",
                    default=osp.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "train_local", "runs", "_reports"),
                    help="report.md / report.csv 写到哪里（默认 train_local/runs/_reports）")
    ap.add_argument("--patterns", nargs="+",
                    default=["*/*/*.state.pd_ee_delta_pos.physx_cpu.h5",
                             "*/*/*.state.pd_ee_delta_pose.physx_cpu.h5"],
                    help="相对 --demo-dir 的 glob（默认同时扫 4 维与 7 维任务）")
    a = ap.parse_args()

    for pat in a.patterns:
        for p in glob.glob(osp.join(a.demo_dir, pat)):
            ROWS.append(check(p))

    ROWS.sort(key=lambda r: (str(r["env_id"]), r["file"]))
    os.makedirs(a.out_dir, exist_ok=True)
    cols = list(ROWS[0].keys()) if ROWS else []

    csv_path = osp.join(a.out_dir, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        w.writerows(ROWS)

    md_path = osp.join(a.out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("| " + " | ".join(cols) + " |\n")
        fp.write("|" + "---|" * len(cols) + "\n")
        for r in ROWS:
            fp.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

    print(f"扫描到 {len(ROWS)} 个数据集 -> {md_path} / {csv_path}")
    if not ROWS:
        print("（没扫到文件：先用 scripts/fetch_demos.py + scripts/convert_all.py 生成）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
