"""
train.py — 学院 GPU 机器（Linux / RTX 4080）上的训练脚本

与 train_local/train.py 的差别（训练循环与 checkpoint 格式完全一致，产物可互相评测）：
  * 默认 --device cuda、--num-workers 8、pin_memory + persistent_workers（Linux 上安全）
  * 结果默认存到 train/runs/（与本机结果分开，报告里好区分算力来源）
  * 模型 / 数据代码复用 train_local（单一真源，避免两份实现漂移）

用法：
  python train.py --env-id PickCube-v1 --backbone unet
  python train.py --env-id PickCube-v1 --total-iters 30000 --seed 0
  python train.py --env-id StackCube-v1 --max-episode-steps 200 --backbone unet
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- 复用 dp 包里的模型 / 数据 / 训练工具（单一真源：dp/）----
HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dp.dp_lib import (DiffusionPolicy, DPDataset, compute_stats,  # noqa: E402
                       find_dataset, load_trajectories, split_episodes)
from dp.utils import EMA, plot_curve  # noqa: E402

try:
    from mani_skill import DEMO_DIR as DEFAULT_DEMO_DIR
except Exception:                                    # 没装 mani_skill 也能训
    DEFAULT_DEMO_DIR = osp.expanduser("~/.maniskill/demos")


def build_args():
    ap = argparse.ArgumentParser(description="GPU 机器训练 Diffusion Policy（单任务）")
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--h5", default=None, help="数据集路径；默认自动查找")
    ap.add_argument("--demo-dir", default=str(DEFAULT_DEMO_DIR))
    ap.add_argument("--out", default=osp.join(HERE, "runs"),
                    help="输出根目录（默认 train/runs，与本机结果分开）")
    ap.add_argument("--exp-name", default=None)

    # 数据
    ap.add_argument("--demo-frac", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-episodes", type=int, default=-1)

    # 模型 / DP 超参（与 train_local 完全一致）
    ap.add_argument("--backbone", default="mlp", choices=["mlp", "unet", "transformer"])
    ap.add_argument("--obs-horizon", type=int, default=2)
    ap.add_argument("--pred-horizon", type=int, default=16)
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--num-train-timesteps", type=int, default=100)
    ap.add_argument("--obs-feat-dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--unet-down-dims", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--unet-kernel-size", type=int, default=5)
    ap.add_argument("--unet-n-groups", type=int, default=8)
    ap.add_argument("--unet-step-embed-dim", type=int, default=256)
    ap.add_argument("--unet-cond-predict-scale", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--tf-n-layer", type=int, default=8)
    ap.add_argument("--tf-n-head", type=int, default=4)
    ap.add_argument("--tf-n-emb", type=int, default=256)
    ap.add_argument("--tf-p-drop-emb", type=float, default=0.0)
    ap.add_argument("--tf-p-drop-attn", type=float, default=0.3)
    ap.add_argument("--tf-causal-attn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tf-n-cond-layers", type=int, default=0)

    # 训练
    ap.add_argument("--total-iters", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=256,
                    help="为了和本机结果可比，默认仍 256；想加速可开 512/1024，"
                         "但换了 batch 的结果不要和 256 的混在同一张表里")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-6)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8, help="Linux 上可以开大；报错就降回 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])

    # 评测用（写进 checkpoint，eval.py 会读）
    ap.add_argument("--control-mode", default="pd_ee_delta_pos")
    ap.add_argument("--obs-mode", default="state")
    ap.add_argument("--sim-backend", default="cpu",
                    help="写进 checkpoint；评测时可用 eval.py --sim-backend 覆盖。"
                         "为了和已有结果可比，默认保持 cpu")
    ap.add_argument("--max-episode-steps", type=int, default=100,
                    help="PickCube 100 / StackCube 200 / PegInsertionSide 300")
    ap.add_argument("--num-inference-steps", type=int, default=10)
    return ap.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}"
          + (f"  ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB)"
             if device == "cuda" else ""))

    # ---------------- 数据（与 train_local 完全同一套逻辑） ----------------
    h5 = args.h5 or find_dataset(args.env_id, args.demo_dir)
    print(f"[data] {h5}")
    trajs = load_trajectories(h5, success_only=True, max_episodes=args.max_episodes)
    n_all = len(trajs)

    train_ids, val_ids = split_episodes(n_all, args.val_frac, args.seed)
    if args.demo_frac < 1.0:
        keep = max(1, int(round(len(train_ids) * args.demo_frac)))
        rng = np.random.default_rng(args.seed + 12345)
        train_ids = sorted(rng.permutation(train_ids)[:keep].tolist())

    train_trajs = [trajs[i] for i in train_ids]
    val_trajs = [trajs[i] for i in val_ids]
    stats = compute_stats(train_trajs)                 # 统计量只用训练集，防泄漏
    To, Tp, Ta = args.obs_horizon, args.pred_horizon, args.action_horizon
    train_ds = DPDataset(train_trajs, stats, To=To, Tp=Tp)
    val_ds = DPDataset(val_trajs, stats, To=To, Tp=Tp) if val_trajs else None
    obs_dim = trajs[0][0].shape[-1]
    act_dim = trajs[0][1].shape[-1]
    print(f"[data] 演示总数 {n_all} | 训练 {len(train_ids)} 条/{len(train_ds)} 步 "
          f"| 验证 {len(val_ids)} 条/{len(val_ds) if val_ds else 0} 步 "
          f"| obs {obs_dim} 维 / act {act_dim} 维")

    # ---------------- 模型 ----------------
    steps_per_epoch = max(1, len(train_ds) // args.batch)
    if args.epochs is None:
        args.epochs = max(1, int(round(args.total_iters / steps_per_epoch)))
    total_iters = steps_per_epoch * args.epochs
    print(f"[plan] {steps_per_epoch} step/epoch × {args.epochs} epoch = {total_iters} iter "
          f"(batch {args.batch})")

    model = DiffusionPolicy(
        obs_dim=obs_dim, act_dim=act_dim, obs_horizon=To, pred_horizon=Tp,
        action_horizon=Ta, num_train_timesteps=args.num_train_timesteps,
        obs_feat_dim=args.obs_feat_dim, hidden=args.hidden, n_layers=args.n_layers,
        backbone=args.backbone, unet_down_dims=list(args.unet_down_dims),
        unet_kernel_size=args.unet_kernel_size, unet_n_groups=args.unet_n_groups,
        unet_step_embed_dim=args.unet_step_embed_dim,
        unet_cond_predict_scale=args.unet_cond_predict_scale,
        tf_n_layer=args.tf_n_layer, tf_n_head=args.tf_n_head, tf_n_emb=args.tf_n_emb,
        tf_p_drop_emb=args.tf_p_drop_emb, tf_p_drop_attn=args.tf_p_drop_attn,
        tf_causal_attn=args.tf_causal_attn, tf_n_cond_layers=args.tf_n_cond_layers,
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] backbone={args.backbone}  参数量 {n_par/1e6:.3f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    ema = EMA(model, args.ema_decay)

    nw = args.num_workers
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=nw, drop_last=True, pin_memory=(device == "cuda"),
                              persistent_workers=(nw > 0))
    val_loader = (DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=nw, pin_memory=(device == "cuda"),
                             persistent_workers=(nw > 0)) if val_ds else None)

    # ---------------- 输出 ----------------
    name = args.exp_name or (f"{args.env_id}_frac{args.demo_frac}"
                             f"_{args.backbone}_seed{args.seed}")
    out_dir = osp.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = osp.join(out_dir, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "val_loss", "lr", "sec"])

    cfg = dict(
        env_id=args.env_id, h5=h5, demo_frac=args.demo_frac, seed=args.seed,
        backbone=args.backbone,
        control_mode=args.control_mode, obs_mode=args.obs_mode,
        sim_backend=args.sim_backend, max_episode_steps=args.max_episode_steps,
        num_inference_timesteps=args.num_inference_steps,
        n_train_episodes=len(train_ids), n_val_episodes=len(val_ids),
        n_params=n_par,
        model_kwargs=dict(obs_dim=obs_dim, act_dim=act_dim, obs_horizon=To,
                          pred_horizon=Tp, action_horizon=Ta,
                          num_train_timesteps=args.num_train_timesteps,
                          obs_feat_dim=args.obs_feat_dim,
                          hidden=args.hidden, n_layers=args.n_layers,
                          backbone=args.backbone,
                          unet_down_dims=list(args.unet_down_dims),
                          unet_kernel_size=args.unet_kernel_size,
                          unet_n_groups=args.unet_n_groups,
                          unet_step_embed_dim=args.unet_step_embed_dim,
                          unet_cond_predict_scale=args.unet_cond_predict_scale,
                          tf_n_layer=args.tf_n_layer, tf_n_head=args.tf_n_head,
                          tf_n_emb=args.tf_n_emb,
                          tf_p_drop_emb=args.tf_p_drop_emb,
                          tf_p_drop_attn=args.tf_p_drop_attn,
                          tf_causal_attn=args.tf_causal_attn,
                          tf_n_cond_layers=args.tf_n_cond_layers),
    )
    stats_json = {k: v.tolist() for k, v in stats.items()}

    # ---------------- 训练循环（与 train_local/train.py 一致） ----------------
    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nseen = 0.0, 0
        for obs_seq, act_seq in train_loader:
            obs_seq = obs_seq.to(device, non_blocking=True)
            act_seq = act_seq.to(device, non_blocking=True)
            loss = model.compute_loss(obs_seq, act_seq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)
            tot += loss.item() * obs_seq.shape[0]
            nseen += obs_seq.shape[0]
        sched.step()
        train_loss = tot / max(1, nseen)

        val_loss = ""
        if val_loader is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            model.eval()
            vtot, vseen = 0.0, 0
            with torch.no_grad():
                for obs_seq, act_seq in val_loader:
                    obs_seq = obs_seq.to(device)
                    act_seq = act_seq.to(device)
                    vtot += model.compute_loss(obs_seq, act_seq).item() * obs_seq.shape[0]
                    vseen += obs_seq.shape[0]
            val_loss = vtot / max(1, vseen)
            print(f"[{args.env_id}] ep {epoch:4d}/{args.epochs}  train {train_loss:.5f}  "
                  f"val {val_loss:.5f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{time.time()-t0:.0f}s", flush=True)

        with open(log_path, "a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([epoch, f"{train_loss:.6f}",
                                     f"{val_loss:.6f}" if val_loss != "" else "",
                                     f"{sched.get_last_lr()[0]:.3e}", f"{time.time()-t0:.1f}"])

        if val_loss != "" or epoch == args.epochs:
            ck = dict(config=cfg, stats=stats_json,
                      model_state_dict=model.state_dict(),
                      ema_state_dict=ema.ema.state_dict(),
                      epoch=epoch, val_loss=(val_loss if val_loss != "" else None),
                      train_loss=train_loss)
            torch.save(ck, osp.join(out_dir, "last.pt"))
            score = val_loss if val_loss != "" else train_loss
            if score < best:
                best = score
                torch.save(ck, osp.join(out_dir, "best.pt"))
        if args.save_every and epoch % args.save_every == 0:
            torch.save(model.state_dict(), osp.join(out_dir, f"epoch{epoch}.pt"))

    # ---------------- 收尾 ----------------
    json.dump(dict(cfg=cfg, best_score=best, epochs=args.epochs,
                   total_iters=total_iters, batch=args.batch,
                   seconds=round(time.time() - t0, 1),
                   final_train_loss=train_loss),
              open(osp.join(out_dir, "train_summary.json"), "w", encoding="utf-8"),
              indent=2)
    plot_curve(log_path, osp.join(out_dir, "loss_curve.png"), name)

    print(f"\n[done] {name}")
    print(f"  best score : {best:.5f}")
    print(f"  用时       : {time.time()-t0:.0f} s")
    print(f"  产物目录   : {osp.abspath(out_dir)}")
    print(f"  下一步     : python eval.py --ckpt \"{osp.join(out_dir,'best.pt')}\" "
          f"-n 50 --seed0 2000")


if __name__ == "__main__":
    main()
