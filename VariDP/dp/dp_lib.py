"""
dp_lib.py — 本地训练用的 Diffusion Policy（state 观测）与数据集

噪声预测主干可选 mlp / unet / transformer（见 backbones.py），
其余部分（观测编码、加噪、DDIM 采样、损失）三种主干完全共用。
只依赖 torch / numpy / h5py，不需要 diffusers、wandb、tensorboard。
"""
from __future__ import annotations
from .backbones import build_noise_pred

import math
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

H5_SUFFIX = ".state.pd_ee_delta_pos.physx_cpu.h5"


# --------------------------------------------------------------------------- #
# 噪声调度：squaredcos_cap_v2（Diffusion Policy 论文默认）
# --------------------------------------------------------------------------- #
def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    ac = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999).float()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device, dtype=torch.float32) /
                          max(half - 1, 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLP(nn.Module):
    def __init__(self, dims, act=nn.Mish):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [act(), nn.LayerNorm(dims[i + 1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DiffusionPolicy(nn.Module):
    """To 帧 state 观测 -> Tp 步动作块（DDPM 训练 / DDIM 推理）"""

    def __init__(self, obs_dim: int, act_dim: int, obs_horizon: int = 2,
                 pred_horizon: int = 16, action_horizon: int = 8,
                 num_train_timesteps: int = 100, obs_feat_dim: int = 256,
                 hidden: int = 256, n_layers: int = 3, backbone="mlp",
                 unet_down_dims=(256, 512, 1024), unet_kernel_size: int = 5,
                 unet_n_groups: int = 8, unet_step_embed_dim: int = 256,
                 unet_cond_predict_scale: bool = True,
                 tf_n_layer: int = 8, tf_n_head: int = 4, tf_n_emb: int = 256,
                 tf_p_drop_emb: float = 0.0, tf_p_drop_attn: float = 0.3,
                 tf_causal_attn: bool = True, tf_n_cond_layers: int = 0):
        super().__init__()
        assert action_horizon <= pred_horizon
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.action_horizon = action_horizon
        self.num_train_timesteps = num_train_timesteps

        self.backbone = backbone
        # A1（对齐论文 state/lowdim 配置）：
        #   unet        -> 无 obs encoder，global_cond = 归一化 obs[:, :To] 展平，cond_dim = obs_dim * To
        #   transformer -> 无 obs encoder，cond = 归一化 obs[:, :To]（逐步，(B,To,obs_dim)）
        #   mlp         -> 保留 obs encoder（MLP 主干是本项目自加的 baseline，论文未给对应网络）
        if backbone == "unet":
            self.cond_dim = obs_dim * obs_horizon
            self.obs_encoder = None
        elif backbone == "transformer":
            self.cond_dim = obs_dim
            self.obs_encoder = None
        else:
            self.cond_dim = obs_feat_dim
            self.obs_encoder = MLP([obs_dim * obs_horizon, obs_feat_dim, obs_feat_dim])
        self.t_emb = SinusoidalPosEmb(128)              # 仅 mlp / transformer 用
        # 噪声预测主干：mlp / unet / transformer（其余部分完全复用）
        self.noise_pred = build_noise_pred(
            backbone, pred_horizon, act_dim, self.cond_dim,
            hidden=hidden, n_layers=n_layers,
            unet_down_dims=unet_down_dims, unet_kernel_size=unet_kernel_size,
            unet_n_groups=unet_n_groups, unet_step_embed_dim=unet_step_embed_dim,
            unet_cond_predict_scale=unet_cond_predict_scale,
            n_obs_steps=obs_horizon,
            tf_n_layer=tf_n_layer, tf_n_head=tf_n_head, tf_n_emb=tf_n_emb,
            tf_p_drop_emb=tf_p_drop_emb, tf_p_drop_attn=tf_p_drop_attn,
            tf_causal_attn=tf_causal_attn, tf_n_cond_layers=tf_n_cond_layers)

        betas = cosine_beta_schedule(num_train_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("ac", torch.cumprod(1.0 - betas, dim=0))

    # ---- 条件特征 ----
    def cond(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """unet：归一化 obs 展平；transformer：保留 (B, To, D) 逐步条件；mlp：MLP 编码"""
        if self.backbone == "transformer":
            return obs_seq
        if self.backbone == "unet":
            return obs_seq.reshape(obs_seq.shape[0], -1)
        return self.obs_encoder(obs_seq.reshape(obs_seq.shape[0], -1))

    def eps(self, x, t, c):
        """统一契约：主干自己决定怎么用 x / t_emb / c

        unet / transformer 走官方签名（原始 timestep + 条件），mlp 仍用预计算的 t_emb。
        """
        if self.backbone == "unet":
            return self.noise_pred(x, t, global_cond=c)
        if self.backbone == "transformer":
            return self.noise_pred(x, t, cond=c)
        return self.noise_pred(x, self.t_emb(t), c)

    # ---- 训练损失：预测噪声 ----
    def compute_loss(self, obs_seq, act_seq):
        b = act_seq.shape[0]
        c = self.cond(obs_seq)
        t = torch.randint(0, self.num_train_timesteps, (b,), device=act_seq.device)
        noise = torch.randn_like(act_seq)
        ac = self.ac[t].reshape(b, 1, 1)
        xt = ac.sqrt() * act_seq + (1 - ac).sqrt() * noise
        return F.mse_loss(self.eps(xt, t, c), noise)

    # ---- DDIM 采样 ----
    @torch.no_grad()
    def sample(self, obs_seq: torch.Tensor, num_inference_timesteps: int = 10) -> torch.Tensor:
        c = self.cond(obs_seq)
        b = c.shape[0]
        x = torch.randn(b, self.pred_horizon, self.act_dim, device=c.device)
        ts = torch.linspace(self.num_train_timesteps - 1, 0,
                            num_inference_timesteps, dtype=torch.long, device=c.device)
        for i, t in enumerate(ts):
            e = self.eps(x, t.expand(b), c)
            ac_t = self.ac[t].reshape(1, 1, 1)
            x0 = ((x - (1 - ac_t).sqrt() * e) / ac_t.sqrt()).clamp(-1, 1)
            if i + 1 < len(ts):
                ac_p = self.ac[ts[i + 1]].reshape(1, 1, 1)
                x = ac_p.sqrt() * x0 + (1 - ac_p).sqrt() * e
            else:
                x = x0
        return x


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_trajectories(h5_path: str, success_only: bool = True,
                      max_episodes: int = -1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """读 h5，返回 [(obs[T,D], act[T,A]), ...]（obs 已裁到与 action 等长）"""
    trajs: List[Tuple[np.ndarray, np.ndarray]] = []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        if max_episodes > 0:
            keys = keys[:max_episodes]
        for k in keys:
            g = f[k]
            if success_only and "success" in g and len(g["success"]) > 0:
                if not bool(np.asarray(g["success"][-1]).any()):
                    continue
            a = np.asarray(g["actions"][:], dtype=np.float32)
            o = np.asarray(g["obs"][:], dtype=np.float32)[:len(a)]
            if len(a) == 0:
                continue
            trajs.append((o, a))
    if not trajs:
        raise RuntimeError(f"{h5_path} 里没有可用轨迹")
    return trajs


def compute_stats(trajs) -> Dict[str, np.ndarray]:
    """观测 z-score + 动作 min/max；低方差观测维度不做缩放（防止除 0 爆炸）"""
    obs = np.concatenate([o for o, _ in trajs], axis=0)
    act = np.concatenate([a for _, a in trajs], axis=0)
    std = obs.std(0)
    std = np.where(std < 1e-3, 1.0, std)
    return dict(
        obs_mean=obs.mean(0).astype(np.float32),
        obs_std=std.astype(np.float32),
        act_min=act.min(0).astype(np.float32),
        act_max=act.max(0).astype(np.float32),
    )


class DPDataset(Dataset):
    """按 episode 切窗，两端用「重复边界帧」补齐（DP 官方做法）"""

    def __init__(self, trajs, stats: Dict[str, np.ndarray], To: int = 2, Tp: int = 16):
        self.trajs = trajs
        self.stats = stats
        self.To, self.Tp = To, Tp
        self.index = [(i, t) for i, (_, a) in enumerate(trajs) for t in range(len(a))]

    def __len__(self) -> int:
        return len(self.index)

    # ---- 归一化 ----
    def norm_obs(self, o: np.ndarray) -> np.ndarray:
        return np.clip((o - self.stats["obs_mean"]) / self.stats["obs_std"], -10.0, 10.0)

    def norm_act(self, a: np.ndarray) -> np.ndarray:
        lo, hi = self.stats["act_min"], self.stats["act_max"]
        return 2.0 * (a - lo) / np.maximum(hi - lo, 1e-6) - 1.0

    @staticmethod
    def unnorm_act(a: np.ndarray, stats) -> np.ndarray:
        lo, hi = stats["act_min"], stats["act_max"]
        return (a + 1.0) / 2.0 * (hi - lo) + lo

    def __getitem__(self, idx):
        i, t = self.index[idx]
        o, a = self.trajs[i]
        length = len(a)
        o_ids = [max(0, t - j) for j in range(self.To - 1, -1, -1)]
        a_ids = [min(length - 1, t + j) for j in range(self.Tp)]
        return (torch.from_numpy(self.norm_obs(o[o_ids])),
                torch.from_numpy(self.norm_act(a[a_ids])))


def split_episodes(n: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """按 episode 划分训练/验证（绝不按单步随机划分）"""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac))) if n > 4 else 0
    val = sorted(perm[:n_val].tolist())
    train = sorted(perm[n_val:].tolist())
    return train, val


def find_dataset(env_id: str, demo_dir: str) -> str:
    """在 ~/.maniskill/demos/<env>/ 下找最新的 state 数据集"""
    import glob
    import os.path as osp
    pats = [osp.join(demo_dir, env_id, "*", "*" + H5_SUFFIX),
            osp.join(demo_dir, env_id, "*" + H5_SUFFIX)]
    cands = [p for pat in pats for p in glob.glob(pat)]
    if not cands:
        raise FileNotFoundError(
            f"没找到 {env_id} 的 state 数据集（在 {demo_dir} 下）。\n"
            f"先跑: python convert_all.py {env_id}  或  python gen_demos_scripted.py --env-id {env_id} -n 200")
    return max(cands, key=osp.getmtime)
