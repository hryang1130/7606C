#!/usr/bin/env python3
"""Open-loop replay of demo actions in the *evaluation* environment.

Checks that the eval env built from a config (env id, control mode, obs mode,
episode length, rendering off) reproduces the dataset: for each demo, reset
with its seed, compare the first observation with the stored one, execute the
stored actions and record success. If this does not reach (near) 100%, the
policy cannot be evaluated fairly in this env and the mismatch must be fixed first.

Example (wsl):
  .venv/bin/python scripts/replay_check.py --config configs/pickcube_state_jointpos.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dp_manip import config as config_lib  # noqa: E402
from dp_manip.data import load_demos  # noqa: E402
from dp_manip.envs import env_kwargs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument("--obs-atol", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, help="write per-episode results as JSON")
    args = parser.parse_args()

    import gymnasium as gym
    import mani_skill
    import mani_skill.envs  # noqa: F401

    cfg = config_lib.load(args.config, args.overrides)
    root = Path(__file__).resolve().parents[1]
    demos = load_demos(root / cfg.data.demo_path, cfg.data.num_demos)
    env = gym.make(cfg.task.env_id, **env_kwargs(cfg))

    rows = []
    for ep in demos.episodes:
        obs, _ = env.reset(seed=ep.seed)
        obs0 = np.asarray(obs.cpu()).reshape(-1)
        first_diff = float(np.abs(obs0 - ep.obs[0]).max())
        success_once, success, max_diff = False, False, first_diff
        for t, action in enumerate(ep.actions):
            obs, _, _, _, info = env.step(action)
            success = bool(np.asarray(info["success"].cpu()).item())
            success_once |= success
            max_diff = max(max_diff, float(np.abs(np.asarray(obs.cpu()).reshape(-1) - ep.obs[t + 1]).max()))
        rows.append({
            "episode_id": ep.episode_id, "seed": ep.seed, "steps": int(ep.actions.shape[0]),
            "first_obs_max_abs_diff": first_diff, "max_obs_abs_diff": max_diff,
            "success_once": success_once,
            "success_at_end": success,
        })
        r = rows[-1]
        print(f"traj_{ep.episode_id} seed={ep.seed} steps={r['steps']} obs0_diff={first_diff:.2e} "
              f"max_diff={max_diff:.2e} success_once={success_once} success_at_end={r['success_at_end']}")
    env.close()

    n = len(rows)
    ok_obs = sum(r["first_obs_max_abs_diff"] <= args.obs_atol for r in rows)
    ok_succ = sum(r["success_once"] for r in rows)
    print(f"mani_skill {mani_skill.__version__}: first obs match {ok_obs}/{n}, "
          f"open-loop success {ok_succ}/{n}, worst obs diff {max(r['max_obs_abs_diff'] for r in rows):.2e}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"mani_skill": mani_skill.__version__, "episodes": rows}, indent=2))
    sys.exit(0 if ok_obs == n and ok_succ == n else 1)


if __name__ == "__main__":
    main()
