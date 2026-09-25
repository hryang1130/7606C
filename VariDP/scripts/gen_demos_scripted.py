"""gen_demos_scripted.py — 用自写脚本控制器（不依赖 mplib）从零生成演示。

为什么需要它：官方运动规划演示依赖 `mplib`（Windows 装不上），而课程允许
`teleoperation / motion planning / scripted controllers / another justified method` ——
这个脚本就是「自己生成演示」的那条路：纯 `pd_ee_delta_pos` 端点闭环控制 + 手写 waypoint 程序。

要点（踩过的坑都在代码里）：
  * 动作是**基座坐标系**下的位移增量（POS_SCALE = 0.1 m），机器人根节点不在世界原点
    （`world_to_base` 做的就是这个变换）；
  * `max_episode_steps` 默认 50 太短，这里放宽到 300；
  * `save_video=False`（本机无 Vulkan，开渲染会崩）；
  * 只保存成功轨迹（`env.flush_trajectory(save=success)`），并在报告里记录成功率。

用法（在仓库根目录，需要 GPU/CPU 仿真实例；不需要 mplib）：
  python scripts/gen_demos_scripted.py --env-id PickCube-v1 -n 200
  python scripts/gen_demos_scripted.py --env-id PushCube-v1 -n 200
  python scripts/gen_demos_scripted.py --env-id StackCube-v1 -n 100 --out-dir <自定义目录>

产物：`<demo_dir>/<env-id>/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`
（可直接被 `dp.dp_lib.find_dataset` / `train_local/train.py` 找到）
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import time

import numpy as np
import gymnasium as gym

import mani_skill.envs  # noqa: F401  必须导入才会注册环境
from mani_skill import DEMO_DIR
from mani_skill.utils.wrappers.record import RecordEpisode

POS_SCALE = 0.1        # 归一化动作 1.0 == 0.1 m
OPEN, CLOSE = 1.0, -1.0
MAX_EPISODE_STEPS = 300   # 默认 50 太短，必须放宽


# ---------- 四元数工具（纯 numpy，避免依赖 Pose API 细节） ----------
def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate(q, v):
    """用 wxyz 四元数旋转向量 v"""
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=np.float64)
    return v + 2.0 * w * np.cross(qv, v) + 2.0 * np.cross(qv, np.cross(qv, v))


class ScriptedController:
    """pd_ee_delta_pos 动作空间下的末端点到点控制器"""

    def __init__(self, env, kp=2.0):
        self.env = env
        self.u = env.unwrapped
        self.agent = self.u.agent
        self.kp = kp
        self.n_steps = 0

    # ---- 坐标 ----
    def root_pose(self):
        pose = self.agent.robot.pose
        p = pose.p[0].detach().cpu().numpy().astype(np.float64)
        q = pose.q[0].detach().cpu().numpy().astype(np.float64)
        return p, q

    def world_to_base(self, p_world):
        rp, rq = self.root_pose()
        return quat_rotate(quat_conj(rq), np.asarray(p_world, dtype=np.float64) - rp)

    def tcp_world(self):
        return self.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float64)

    def tcp_base(self):
        return self.world_to_base(self.tcp_world())

    def obj_world(self, attr):
        return getattr(self.u, attr).pose.p[0].detach().cpu().numpy().astype(np.float64)

    # ---- 动作 ----
    def action_to(self, target_world, gripper):
        d = self.world_to_base(target_world) - self.tcp_base()
        a = np.clip(self.kp * d / POS_SCALE, -1.0, 1.0)
        return np.array([a[0], a[1], a[2], gripper], dtype=np.float32)

    def _step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.n_steps += 1
        return info, bool(np.asarray(terminated).any()), bool(np.asarray(truncated).any())

    def goto(self, target_world, gripper, tol=0.01, max_steps=60):
        """闭环移动到目标点，误差 < tol 认为到位"""
        for _ in range(max_steps):
            _, term, trunc = self._step(self.action_to(target_world, gripper))
            if term or trunc:
                return False
            if np.linalg.norm(self.world_to_base(target_world) - self.tcp_base()) < tol:
                return True
        return False

    def hold(self, target_world, gripper, n=18):
        """保持目标点不动，持续施加夹爪指令"""
        for _ in range(n):
            _, term, trunc = self._step(self.action_to(target_world, gripper))
            if term or trunc:
                return False
        return True

    # ---- 收尾 ----
    def success(self):
        return bool(np.asarray(self.u.evaluate()["success"]).any())

    def settle(self, gripper, n=60, check_every=2):
        """原地不动（目标点=当前 TCP），等待 is_robot_static 之类的判据满足"""
        for i in range(n):
            _, term, trunc = self._step(self.action_to(self.tcp_world(), gripper))
            if term or trunc:
                break
            if i % check_every == 0 and self.success():
                return True
        return self.success()

    def servo(self, make_target, gripper, n=150):
        """闭环伺服：每步用 make_target() 重算目标，一旦判成功立即停"""
        for _ in range(n):
            _, term, trunc = self._step(self.action_to(make_target(), gripper))
            if term or trunc:
                break
            if self.success():
                return True
        return self.success()


def prog_pick_cube(ctl):
    """抓取并稳定举在 goal_site 处（不松手）"""
    cube = ctl.obj_world("cube")
    goal = ctl.obj_world("goal_site")
    ctl.goto(cube + np.array([0, 0, 0.08]), OPEN, tol=0.012, max_steps=40)
    ctl.goto(cube + np.array([0, 0, 0.005]), OPEN, tol=0.005, max_steps=40)
    ctl.hold(cube + np.array([0, 0, 0.005]), CLOSE, n=18)
    ctl.goto(cube + np.array([0, 0, 0.10]), CLOSE, tol=0.012, max_steps=30)
    ctl.goto(goal, CLOSE, tol=0.008, max_steps=60)
    return ctl.settle(CLOSE, n=60)


def prog_push_cube(ctl):
    """绕到背面，把方块推（非抓取）进 goal_region"""
    cube = ctl.obj_world("obj")
    goal = ctl.obj_world("goal_region")
    d = goal[:2] - cube[:2]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return False
    d = d / n
    behind = np.array([cube[0] - d[0] * 0.10, cube[1] - d[1] * 0.10, cube[2]])

    ctl.goto(behind + np.array([0, 0, 0.06]), OPEN, tol=0.012, max_steps=40)
    ctl.goto(behind, OPEN, tol=0.008, max_steps=30)
    ctl.goto(np.array([cube[0] - d[0] * 0.03, cube[1] - d[1] * 0.03, cube[2]]),
             CLOSE, tol=0.006, max_steps=20)

    def tgt():                       # 始终保持在方块"背面 3.5 cm"，边推边纠偏
        c = ctl.obj_world("obj")
        v = goal[:2] - c[:2]
        v = v / max(np.linalg.norm(v), 1e-6)
        return np.array([c[0] - v[0] * 0.035, c[1] - v[1] * 0.035, c[2]])

    return ctl.servo(tgt, CLOSE, n=150)


def prog_stack_cube(ctl):
    """把 cubeA 抓起来叠到 cubeB 上（松手 + 撤回 + 静止）"""
    a = ctl.obj_world("cubeA")
    b = ctl.obj_world("cubeB")
    ctl.goto(a + np.array([0, 0, 0.08]), OPEN, tol=0.012, max_steps=40)
    ctl.goto(a + np.array([0, 0, 0.005]), OPEN, tol=0.005, max_steps=40)
    ctl.hold(a + np.array([0, 0, 0.005]), CLOSE, n=18)
    ctl.goto(a + np.array([0, 0, 0.12]), CLOSE, tol=0.012, max_steps=30)

    place = np.array([b[0], b[1], b[2] + 0.045])   # 方块半边长 0.02，略高一点释放
    ctl.goto(place + np.array([0, 0, 0.10]), CLOSE, tol=0.015, max_steps=45)
    ctl.goto(place, CLOSE, tol=0.005, max_steps=30)
    ctl.hold(place, OPEN, n=18)
    ctl.goto(place + np.array([0, 0, 0.12]), OPEN, tol=0.02, max_steps=30)
    return ctl.settle(OPEN, n=40)


def prog_pull_cube(ctl):
    """抓住方块后拖进 goal_region"""
    cube = ctl.obj_world("obj")
    goal = ctl.obj_world("goal_region")
    ctl.goto(cube + np.array([0, 0, 0.08]), OPEN, tol=0.012, max_steps=40)
    ctl.goto(cube + np.array([0, 0, 0.005]), OPEN, tol=0.005, max_steps=40)
    ctl.hold(cube + np.array([0, 0, 0.005]), CLOSE, n=18)
    ctl.goto(cube + np.array([0, 0, 0.06]), CLOSE, tol=0.012, max_steps=30)

    def tgt():
        c = ctl.obj_world("obj")
        v = goal[:2] - c[:2]
        v = v / max(np.linalg.norm(v), 1e-6)
        return np.array([c[0] + v[0] * 0.06, c[1] + v[1] * 0.06, c[2] + 0.04])

    return ctl.servo(tgt, CLOSE, n=150)


PROGRAMS = {
    "PickCube-v1": prog_pick_cube,
    "PushCube-v1": prog_push_cube,
    "StackCube-v1": prog_stack_cube,
    "PullCube-v1": prog_pull_cube,
}


def make_env(env_id, obs_mode, control_mode, backend, record_dir, traj_name):
    env = gym.make(env_id, obs_mode=obs_mode, control_mode=control_mode,
                   sim_backend=backend, render_mode=None,
                   max_episode_steps=MAX_EPISODE_STEPS)   # 关键！
    env = RecordEpisode(env, output_dir=record_dir, trajectory_name=traj_name,
                        save_video=False,                 # 本机无 Vulkan，必须关
                        source_type="scripted",
                        source_desc="self-written scripted waypoint controller (no mplib)",
                        record_reward=False, save_on_reset=False)
    return env


def run(env_id, n_traj, seed0, out_dir, traj_name="trajectory"):
    env = make_env(env_id, "state", "pd_ee_delta_pos", "cpu", out_dir, traj_name)
    h5_path = env._h5_file.filename
    prog, ctl = PROGRAMS[env_id], None
    seed, n_ok, n_try = seed0, 0, 0
    ctl = ScriptedController(env)
    t0 = time.time()
    while n_ok < n_traj:
        env.reset(seed=seed)
        ctl.n_steps = 0
        try:
            prog(ctl)
        except Exception as e:
            print(f"[warn] seed={seed}: {type(e).__name__}: {e}", flush=True)
        success = bool(np.asarray(env.unwrapped.evaluate()["success"]).any())
        env.flush_trajectory(save=success)      # 只保留成功轨迹
        n_try += 1
        n_ok += int(success)
        print(f"[{env_id}] try={n_try} ok={n_ok} seed={seed} "
              f"steps={ctl.n_steps} success={success} ({time.time()-t0:.1f}s)", flush=True)
        seed += 1
        if n_try > n_traj * 30:
            print("连续失败过多，提前结束", flush=True)
            break
    env.close()
    return h5_path, n_ok, n_try


def main() -> int:
    ap = argparse.ArgumentParser(description="自写脚本控制器生成演示（不依赖 mplib）")
    ap.add_argument("--env-id", default="PickCube-v1", choices=sorted(PROGRAMS))
    ap.add_argument("-n", "--n-traj", type=int, default=200, help="目标成功轨迹条数")
    ap.add_argument("--seed0", type=int, default=0, help="起始种子（逐条递增）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录，默认 <demo_dir>/<env-id>/motionplanning")
    ap.add_argument("--traj-name", default="trajectory")
    a = ap.parse_args()

    out_dir = a.out_dir or osp.join(str(DEMO_DIR), a.env_id, "motionplanning")
    os.makedirs(out_dir, exist_ok=True)
    h5_path, n_ok, n_try = run(a.env_id, a.n_traj, a.seed0, out_dir, a.traj_name)
    print(f"\n[done] {a.env_id}: 成功 {n_ok}/{n_try} 条（目标 {a.n_traj}）")
    print(f"       h5 文件: {h5_path}")
    print("       质检: python scripts/check_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
