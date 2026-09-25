"""
eval.py — 学院 GPU 机器上的批量评测（支持 physx GPU 并行仿真）

与 train_local/eval.py 的核心差别：
  * --num-envs N：一次并行跑 N 个环境。GPU 仿真下 50 个 episode 从 ~1 分钟降到几秒；
    CPU 后端也会用多进程并行（Linux 上没问题，Windows 别开 >1）
  * --sim-backend auto|gpu|cpu：auto = 有 CUDA 就用 gpu（ManiSkill 3 里 "gpu" 即 physx_cuda）
  * 输出 JSON 与 train_local/eval.py 完全同格式，默认存到 checkpoint 同目录
    → summarize_runs.py 直接能扫（注意 --runs-dir 要指向对应的 runs 目录）

用法：
  python eval.py --ckpt runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --seed0 2000
  python eval.py --ckpt ... -n 200 --num-envs 64            # GPU 并行
  python eval.py --ckpt ... -n 50 --sim-backend cpu         # 与本机结果严格可比
"""
from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
import time

import numpy as np
import torch

import gymnasium as gym

import mani_skill.envs  # noqa: F401  必须导入才会注册环境

# ---- 复用 dp 包里的模型代码（单一真源：dp/dp_lib.py）----
HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dp.dp_lib import DiffusionPolicy  # noqa: E402


def to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_state_obs(obs, n_envs: int) -> np.ndarray:
    """ManiSkill 各种观测容器 -> (n_envs, obs_dim) 的 numpy。"""
    if isinstance(obs, dict):
        if "state" in obs:                       # obs_mode="state" 的标准形态
            s = obs["state"]
            if isinstance(s, dict):
                parts = [to_np(s[k]).reshape(n_envs, -1) for k in sorted(s)]
                return np.concatenate(parts, axis=1)
            return to_np(s).reshape(n_envs, -1)
        parts = []                               # 兜底：dict of sensor dicts
        for k in sorted(obs):
            v = obs[k]
            if isinstance(v, dict):
                parts += [to_np(v[kk]).reshape(n_envs, -1) for kk in sorted(v)]
            else:
                parts.append(to_np(v).reshape(n_envs, -1))
        return np.concatenate(parts, axis=1)
    return to_np(obs).reshape(n_envs, -1)


def load_policy(ckpt_path: str, device: str, use_raw: bool = False):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    pol = DiffusionPolicy(**cfg["model_kwargs"]).to(device).eval()
    pol.load_state_dict(ck["model_state_dict"] if use_raw else ck["ema_state_dict"])
    stats = {k: np.asarray(v, dtype=np.float32) for k, v in ck["stats"].items()}
    return pol, cfg, stats


def main():
    ap = argparse.ArgumentParser(description="GPU 机器批量评测（支持并行仿真）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-n", "--episodes", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=2000, help="held-out 评测起始 seed")
    ap.add_argument("--num-envs", type=int, default=16,
                    help="并行环境数；GPU 仿真建议 32~64，cpu 后端建议 1（Windows 必须 1）")
    ap.add_argument("--sim-backend", default="auto", choices=["auto", "gpu", "cpu"],
                    help="auto = 有 CUDA 用 gpu。**同一组对比实验必须用同一后端**")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--inference-steps", type=int, default=None,
                    help="覆盖 DDIM 采样步数（默认用 checkpoint 里存的）")
    ap.add_argument("--use-raw", action="store_true", help="不用 EMA 权重")
    ap.add_argument("--out-dir", default=None, help="默认存到 checkpoint 同目录")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pol, cfg, stats = load_policy(a.ckpt, device, use_raw=a.use_raw)

    env_id = cfg["env_id"]
    To = cfg["model_kwargs"]["obs_horizon"]
    Ta = cfg["model_kwargs"]["action_horizon"]
    max_steps = cfg["max_episode_steps"]
    n_inf = int(a.inference_steps or cfg["num_inference_timesteps"])
    obs_mean, obs_std = stats["obs_mean"], stats["obs_std"]
    act_lo, act_hi = stats["act_min"], stats["act_max"]

    backend = a.sim_backend
    if backend == "auto":
        backend = "gpu" if (device == "cuda") else "cpu"
    n_envs = max(1, min(a.num_envs, a.episodes))
    if backend == "cpu" and n_envs > 1 and sys.platform.startswith("win"):
        print("[warn] Windows 上 cpu 后端并行容易出问题，改为 --num-envs 1")
        n_envs = 1

    try:
        env = gym.make(env_id, obs_mode=cfg.get("obs_mode", "state"),
                       control_mode=cfg["control_mode"], sim_backend=backend,
                       num_envs=n_envs, render_mode=None, max_episode_steps=max_steps)
    except Exception as e:
        raise RuntimeError(
            f"创建并行仿真失败（backend={backend}, num_envs={n_envs}）：{e}\n"
            f"  GPU 仿真需要驱动支持的显卡且 sapien 自带 PhysX CUDA；"
            f"不行就用 --sim-backend cpu --num-envs 1（结果照样有效，只是慢）") from e

    print(f"[eval] {env_id} | {osp.basename(osp.dirname(a.ckpt))} | device={device} "
          f"backend={backend} num_envs={n_envs} max_steps={max_steps} "
          f"| {'raw' if a.use_raw else 'ema'} | ddim {n_inf}")

    obs_mean_b = obs_mean[None]
    obs_std_b = obs_std[None]
    act_span = np.maximum(act_hi - act_lo, 1e-6)
    results, n_ok, t0 = [], 0, time.time()

    for wave0 in range(0, a.episodes, n_envs):
        k = min(n_envs, a.episodes - wave0)          # 本波要记录的前 k 个环境
        seeds = np.arange(a.seed0 + wave0, a.seed0 + wave0 + n_envs)
        obs, _ = env.reset(seed=seeds)
        st = get_state_obs(obs, n_envs)
        hist = np.repeat(st[:, None, :], To, axis=1)          # (N, To, D)
        chunk = np.zeros((n_envs, Ta, pol.act_dim), np.float32)
        k_in_chunk = np.full(n_envs, Ta, np.int64)   # 设为 Ta：第一步就触发推理（同 train_local 的 k%Ta==0）
        done = np.zeros(n_envs, bool)
        active = ~done
        steps = np.zeros(n_envs, np.int64)
        success = np.zeros(n_envs, bool)

        while not done.all():
            need = active & (k_in_chunk >= Ta)
            if need.any():
                idx = np.where(need)[0]
                o_n = np.clip((hist[idx] - obs_mean_b) / obs_std_b, -10, 10)
                with torch.no_grad():
                    a_n = pol.sample(torch.from_numpy(o_n).to(device), n_inf)
                act_un = (a_n.detach().cpu().numpy() + 1) / 2 * act_span + act_lo
                chunk[idx] = act_un[:, :Ta]
                k_in_chunk[idx] = 0

            act = np.clip(chunk[np.arange(n_envs), np.minimum(k_in_chunk, Ta - 1)], -1, 1)
            act_t = torch.from_numpy(act.astype(np.float32))
            obs, _, term, trunc, info = env.step(act_t.to(device) if backend == "gpu" else act)

            st = get_state_obs(obs, n_envs)
            hist = np.concatenate([hist[:, 1:], st[:, None, :]], axis=1)
            steps += active
            succ = to_np(info.get("success", np.zeros(n_envs, bool))).reshape(-1).astype(bool)
            success |= succ & active                       # done 那一步的 success 仍有效
            term_a = to_np(term).reshape(-1).astype(bool)
            trunc_a = to_np(trunc).reshape(-1).astype(bool)
            newly = active & (term_a | trunc_a)
            done |= newly
            active &= ~done
            k_in_chunk = np.where(active, k_in_chunk + 1, 0)
            if steps.max() > 3 * max_steps:
                print("[warn] 超过 3×max_episode_steps 仍未全部结束，强制结束本波", flush=True)
                break

        for j in range(k):
            results.append(dict(seed=int(seeds[j]), success=bool(success[j]),
                                steps=int(steps[j])))
            n_ok += int(success[j])
        if not a.quiet:
            print(f"  wave {wave0//n_envs + 1}: ep {wave0+1}~{wave0+k}  "
                  f"本波 {int(success[:k].sum())}/{k}  累计 {n_ok}/{len(results)}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    env.close()
    sr = n_ok / max(1, len(results))
    fails = [r for r in results if not r["success"]]
    timeouts = sum(1 for r in fails if r["steps"] >= max_steps)
    print(f"\n[eval] 成功率 = {sr*100:.1f}%  ({n_ok}/{len(results)})  "
          f"平均步数 {np.mean([r['steps'] for r in results]):.1f}  用时 {time.time()-t0:.0f}s")
    if fails:
        print(f"[eval] 失败 {len(fails)} 个：超时 {timeouts} / 提前终止 {len(fails)-timeouts}")

    tag = ""
    if a.use_raw:
        tag += "_raw"
    if a.inference_steps is not None:
        tag += f"_steps{a.inference_steps}"
    if backend != "cpu" or n_envs > 1:
        tag += f"_{backend}{n_envs}"          # 避免覆盖本机跑的同名 cpu 结果
    out_dir = a.out_dir or osp.dirname(a.ckpt)
    out = osp.join(out_dir, f"eval_seed{a.seed0}_n{a.episodes}{tag}.json")
    json.dump(dict(ckpt=osp.abspath(a.ckpt), env_id=env_id,
                   n_episodes=len(results), seed0=a.seed0, success_rate=sr,
                   n_success=n_ok, use_raw=a.use_raw, results=results,
                   sim_backend=backend, num_envs=n_envs,
                   inference_steps=n_inf, seconds=round(time.time() - t0, 1)),
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"[eval] 结果已保存 {out}")


if __name__ == "__main__":
    main()
