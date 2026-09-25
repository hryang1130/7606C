"""
eval.py — 在本地 CPU 单环境仿真上评测训练好的 DP checkpoint

为什么用 CPU 仿真：
  本机 `physx_cuda` 需要 CUDA Toolkit 的 cuda.dll（未安装），
  且无 Vulkan 无法渲染。所以评测统一走 physx_cpu + num_envs=1。
  实测 ~60~140 步/秒，50 个 episode × 100 步 ≈ 40~80 秒。

用法：
  python eval.py --ckpt runs/PickCube-v1_frac1.0_seed0/best.pt -n 50 --seed0 2000
  python eval.py --ckpt ... --use-raw           # 用未 EMA 的权重做对比
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
import time

import gymnasium as gym
import numpy as np
import torch

# ---- 把仓库根目录挂到 sys.path，直接 import dp（装不装包都能跑）----
HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mani_skill.envs  # noqa: F401  必须导入才会注册环境
from dp.dp_lib import DiffusionPolicy  # noqa: E402


def load_policy(ckpt_path: str, device: str, use_raw: bool = False):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    pol = DiffusionPolicy(**cfg["model_kwargs"]).to(device).eval()
    state = ck["model_state_dict"] if use_raw else ck["ema_state_dict"]
    pol.load_state_dict(state)
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in ck["stats"].items()}
    return pol, cfg, stats, ck


def as_flat(obs) -> np.ndarray:
    """obs_mode='state' 返回扁平张量；'state_dict' 返回 dict。两种都兼容。"""
    if isinstance(obs, dict):
        obs = obs.get("state", obs)
        if isinstance(obs, dict):
            parts = []
            for k in sorted(obs.keys()):
                v = obs[k]
                if isinstance(v, dict):
                    parts += [np.asarray(v[kk]).reshape(-1) for kk in sorted(v.keys())]
                else:
                    parts.append(np.asarray(v).reshape(-1))
            obs = np.concatenate(parts)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


@torch.no_grad()
def evaluate(ckpt_path: str, n_episodes: int = 50, seed0: int = 2000,
             device: str = "auto", verbose: bool = True,
             save_json: str | None = None, use_raw: bool = False,
             inference_steps: int | None = None):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pol, cfg, st, _ = load_policy(ckpt_path, device, use_raw=use_raw)
    env_id = cfg["env_id"]
    To = cfg["model_kwargs"]["obs_horizon"]
    Ta = cfg["model_kwargs"]["action_horizon"]
    n_steps = cfg["max_episode_steps"]
    n_inf = int(inference_steps or cfg["num_inference_timesteps"])   # 可覆盖：DP 采样步数消融

    env = gym.make(
        env_id,
        obs_mode=cfg.get("obs_mode", "state"),
        control_mode=cfg["control_mode"],
        sim_backend=cfg.get("sim_backend", "cpu"),   # 本机只能是 cpu
        render_mode=None,
        max_episode_steps=n_steps,                   # 必须显式传，默认 50 太短
    )
    print(f"[eval] {env_id} | {osp.basename(osp.dirname(ckpt_path))} | "
          f"device={device} backend={cfg.get('sim_backend','cpu')} "
          f"max_steps={n_steps} | {'raw' if use_raw else 'ema'}")

    results, n_ok = [], 0
    t0 = time.time()
    for ep in range(n_episodes):
        seed = seed0 + ep
        obs, info = env.reset(seed=seed)
        o = as_flat(obs)
        hist = [o.copy() for _ in range(To)]
        chunk, t, k, success = None, 0, 0, False
        while t < n_steps:
            if k % Ta == 0:                              # receding horizon
                oseq = np.stack(hist[-To:], 0)
                oseq = np.clip((oseq - st["obs_mean"]) / st["obs_std"], -10, 10)
                with torch.no_grad():
                    a_norm = pol.sample(torch.from_numpy(oseq)[None].to(device),
                                        n_inf)[0].cpu().numpy()
                chunk = ((a_norm + 1) / 2 * (st["act_max"] - st["act_min"])
                         + st["act_min"])
            a = np.clip(chunk[k % Ta], -1, 1).astype(np.float32)
            obs, reward, term, trunc, info = env.step(a)
            o = as_flat(obs)
            hist.append(o.copy())
            t += 1
            k += 1
            if bool(np.asarray(info.get("success", [False])).any()):
                success = True
            if bool(np.asarray(term).any()) or bool(np.asarray(trunc).any()):
                break
        n_ok += int(success)
        results.append(dict(seed=seed, success=success, steps=t))
        if verbose:
            print(f"  ep {ep+1:3d}/{n_episodes}  seed={seed}  success={str(success):5s}  "
                  f"steps={t:3d}  ({time.time()-t0:.0f}s)", flush=True)

    env.close()
    sr = n_ok / max(1, n_episodes)
    print(f"\n[eval] 成功率 = {sr*100:.1f}%   ({n_ok}/{n_episodes})   "
          f"平均步数 {np.mean([r['steps'] for r in results]):.1f}   "
          f"用时 {time.time()-t0:.0f}s")

    # 失败案例分类（报告里要用）
    fails = [r for r in results if not r["success"]]
    if fails:
        timeout = sum(1 for r in fails if r["steps"] >= n_steps)
        print(f"[eval] 失败 {len(fails)} 个：超时 {timeout} 个 / 提前终止 "
              f"{len(fails)-timeout} 个（早停通常是物体掉落触发 terminated）")

    if save_json:
        json.dump(dict(ckpt=osp.abspath(ckpt_path), env_id=env_id,
                       n_episodes=n_episodes, seed0=seed0, success_rate=sr,
                       n_success=n_ok, use_raw=use_raw, results=results),
                  open(save_json, "w", encoding="utf-8"), indent=2)
        print(f"[eval] 结果已保存 {save_json}")
    return sr, results


def main():
    ap = argparse.ArgumentParser(description="本地 CPU 仿真评测 DP")
    ap.add_argument("--ckpt", required=True, help="checkpoint 路径（best.pt / last.pt）")
    ap.add_argument("-n", "--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=2000, help="评测起始 seed（held-out）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--use-raw", action="store_true", help="不用 EMA 权重")
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="覆盖 DDIM 采样步数（默认用 checkpoint 里的 10）。"
                         "这是『改进 DP：diffusion sampling』消融的开关")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    tag = "" if a.inference_steps is None else f"_steps{a.inference_steps}"
    ev = osp.join(osp.dirname(a.ckpt),
                  f"eval_seed{a.seed0}_n{a.episodes}{'_raw' if a.use_raw else ''}{tag}.json")
    evaluate(a.ckpt, a.episodes, a.seed0, a.device,
             verbose=not a.quiet, save_json=None if a.no_save else ev,
             use_raw=a.use_raw, inference_steps=a.inference_steps)


if __name__ == "__main__":
    main()
