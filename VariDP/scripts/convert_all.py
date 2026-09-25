"""convert_all.py — 把官方 `pd_joint_pos` 演示批量重放转换成 `state` + `pd_ee_delta_pos/pose`。

原理：用 `mani_skill.trajectory.replay_trajectory` 在仿真里重放原始演示，
按 `-c <control_mode> -o state` 重新录制动作与观测，输出
`<Task>/motionplanning/trajectory.<obs_mode>.<control_mode>.physx_cpu.h5`（训练直接吃这个文件）。

注意：
  * `PegInsertionSide-v1` / `PlugCharger-v1` / `PushT-v1` **必须**用 7 维的 `pd_ee_delta_pose`
    （插 peg / 插充电器需要旋转末端），其余任务用 4 维 `pd_ee_delta_pos`。
  * `-n/--num-envs` 是并行重放的环境数：Windows 建议 1，Linux 可以 4~8。
  * 转换依赖 pinocchio（`pd_joint_pos → pd_ee_delta_pos` 的 IK/动力学）——本机 venv 已移植。

用法（在仓库根目录）：
  python scripts/convert_all.py                                  # 默认 6 个任务
  python scripts/convert_all.py --tasks PickCube-v1 StackCube-v1
  python scripts/convert_all.py --num-envs 4                     # Linux 上加速
"""
import argparse
import os
import os.path as osp
import subprocess
import sys

try:
    from mani_skill import DEMO_DIR
except Exception:
    DEMO_DIR = os.path.expanduser("~/.maniskill/demos")

DEFAULT_TASKS = ["PickCube-v1", "StackCube-v1", "PushCube-v1",
                 "PullCube-v1", "PegInsertionSide-v1", "PlugCharger-v1"]
# 需要旋转末端的任务 -> 7 维动作空间
POSE_TASKS = {"PegInsertionSide-v1", "PlugCharger-v1", "PushT-v1"}


def main() -> int:
    ap = argparse.ArgumentParser(description="批量重放转换 ManiSkill 演示")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="要转换的任务 id")
    ap.add_argument("--demo-dir", default=str(DEMO_DIR), help="演示根目录（默认 mani_skill.DEMO_DIR）")
    ap.add_argument("--num-envs", type=int, default=1, help="并行重放环境数（Windows 建议 1）")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令")
    a = ap.parse_args()

    n_ok = 0
    for t in a.tasks:
        src = osp.join(a.demo_dir, t, "motionplanning", "trajectory.h5")
        if not osp.exists(src):
            print(f"[skip] {t}: 缺 {src}（先跑 scripts/fetch_demos.py）")
            continue
        control = "pd_ee_delta_pose" if t in POSE_TASKS else "pd_ee_delta_pos"
        cmd = [sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
               "--traj-path", src, "--use-first-env-state",
               "-c", control, "-o", "state",
               "--save-traj", "-n", str(a.num_envs), "-b", "cpu"]
        print("[run]", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=False, cwd=os.getcwd())
        n_ok += 1
    print(f"\n完成 {n_ok}/{len(a.tasks)} 个任务（动作空间：{POSE_TASKS} 用 7 维，其余 4 维）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
