#!/usr/bin/env python3
"""Offline checks of the DP pipeline (no simulator needed).

1. demos load with JSON episode ids and match the config's task;
2. training windows equal a direct index computation, including padding;
3. action normalization maps data into [-1, 1] and round-trips;
4. one training step gives a finite loss, finite grads and changes parameters;
5. sampling returns (B, act_horizon, act_dim) inside the demo action range;
6. checkpoint save/load reproduces the EMA weights and normalizer.

Example:
  .venv/bin/python scripts/check_dp_offline.py --config configs/pickcube_state_jointpos.toml --device cuda
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.data import ActionNormalizer, WindowSampler, build_windows, is_delta_control, load_demos  # noqa: E402
from dp_manip.policy import DiffusionPolicy, load_checkpoint, num_params, save_checkpoint  # noqa: E402


def expected_window(ep, t: int, oh: int, ph: int, delta: bool):
    """Direct (unvectorized) definition of the window whose current step is t."""
    T = ep.actions.shape[0]
    obs = np.stack([ep.obs[max(i, 0)] for i in range(t - oh + 1, t + 1)])
    still = ep.actions[-1].copy()
    if delta:
        still[:-1] = 0
    acts = []
    for i in range(t - oh + 1, t - oh + 1 + ph):
        acts.append(ep.actions[0] if i < 0 else ep.actions[i] if i < T else still)
    return obs, np.stack(acts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device(args.device)
    root = Path(__file__).resolve().parents[1]
    cfg = config_lib.load(args.config, args.overrides)
    p = cfg.policy

    demos = load_demos(root / cfg.data.demo_path, cfg.data.num_demos)
    assert (demos.env_id, demos.control_mode, demos.obs_mode) == \
        (cfg.task.env_id, cfg.task.control_mode, cfg.task.obs_mode), "dataset/config mismatch"
    assert not set(demos.seeds) & (set(cfg.val_seeds()) | set(cfg.test_seeds())), "eval seeds overlap demos"
    lengths = [e.actions.shape[0] for e in demos.episodes]
    print(f"[1] {len(demos.episodes)} demos, seeds {demos.seeds}, lengths {lengths} "
          f"(total {sum(lengths)}), obs_dim {demos.obs_dim}, act_dim {demos.act_dim}")

    obs_w, act_w = build_windows(demos, p.obs_horizon, p.pred_horizon)
    assert obs_w.shape == (sum(lengths), p.obs_horizon, demos.obs_dim)
    assert act_w.shape == (sum(lengths), p.pred_horizon, demos.act_dim)
    delta = is_delta_control(demos.control_mode)
    k = 0
    for ep in demos.episodes:
        for t in range(ep.actions.shape[0]):
            o, a = expected_window(ep, t, p.obs_horizon, p.pred_horizon, delta)
            assert np.array_equal(obs_w[k], o) and np.array_equal(act_w[k], a), (ep.episode_id, t)
            k += 1
    print(f"[2] {k} windows match direct indexing (obs {obs_w.shape[1:]}, actions {act_w.shape[1:]}, "
          f"end padding {'zero-delta' if delta else 'repeat last'})")

    norm = ActionNormalizer.fit(demos)
    n = norm.normalize(act_w)
    assert n.min() >= -1 - 1e-6 and n.max() <= 1 + 1e-6, (n.min(), n.max())
    back = norm.unnormalize(n)
    err = float(np.abs(back - act_w).max())
    assert err < 1e-5, err
    t_err = float((norm.unnormalize(torch.from_numpy(n)) - torch.from_numpy(act_w)).abs().max())
    assert t_err < 1e-5, t_err
    raw_range = (float(act_w.min()), float(act_w.max()))
    print(f"[3] raw action range [{raw_range[0]:.3f}, {raw_range[1]:.3f}] -> normalized "
          f"[{n.min():.3f}, {n.max():.3f}], round-trip error {max(err, t_err):.1e}")

    sampler = WindowSampler(demos, norm, p.obs_horizon, p.pred_horizon, device)
    policy = DiffusionPolicy(p, demos.obs_dim, demos.act_dim, norm).to(device)
    before = copy.deepcopy(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    obs_b, act_b = sampler.sample(32, torch.Generator().manual_seed(0))
    loss = policy.compute_loss(obs_b, act_b)
    opt.zero_grad()
    loss.backward()
    grads_finite = all(torch.isfinite(q.grad).all() for q in policy.parameters() if q.grad is not None)
    opt.step()
    changed = sum(not torch.equal(before[name], v) for name, v in policy.state_dict().items())
    assert torch.isfinite(loss) and grads_finite and changed > 0
    print(f"[4] {num_params(policy.noise_pred_net) / 1e6:.2f}M params, loss {loss.item():.4f}, "
          f"grads finite, {changed} tensors updated")

    policy.eval()
    actions = policy.get_action(obs_b[:4])
    assert actions.shape == (4, p.act_horizon, demos.act_dim), actions.shape
    lo, hi = torch.as_tensor(norm.low, device=device), torch.as_tensor(norm.high, device=device)
    assert torch.isfinite(actions).all() and (actions >= lo - 1e-4).all() and (actions <= hi + 1e-4).all()
    print(f"[5] sampled actions {tuple(actions.shape)} within demo range")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check.pt"
        save_checkpoint(path, policy=policy, ema_policy=policy, config=cfg.to_dict(), iteration=1)
        loaded, loaded_cfg, _ = load_checkpoint(path, device)
    same = all(torch.equal(a, b) for a, b in zip(policy.state_dict().values(), loaded.state_dict().values()))
    assert same and np.array_equal(loaded.normalizer.low, norm.low) and np.array_equal(loaded.normalizer.high, norm.high)
    assert loaded_cfg.to_dict() == cfg.to_dict()
    print("[6] checkpoint round-trip: weights, normalizer and config identical")
    print("all offline checks passed")


if __name__ == "__main__":
    main()
