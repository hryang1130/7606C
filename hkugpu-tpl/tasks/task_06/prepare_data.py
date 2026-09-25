from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def prepare(data_dir: str | Path, conversion_workers: int = 8):
    data_dir = Path(data_dir).resolve()
    converted_datasets = sorted(data_dir.rglob("trajectory.state.pd_ee_delta_pose.physx_cpu.h5"))
    if converted_datasets:
        print(f"Converted dataset already exists: {converted_datasets[0]}")
        return
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "mani_skill.utils.download_demo", "PlugCharger-v1",
        "--output_dir", str(raw_dir),
    ], check=True)

    candidates = sorted(raw_dir.rglob("trajectory.h5"))
    if not candidates:
        raise FileNotFoundError(f"Could not find downloaded trajectory.h5 under {raw_dir}")
    source = candidates[0]
    expected = source.with_name("trajectory.state.pd_ee_delta_pose.physx_cpu.h5")
    if expected.exists():
        print(f"Converted dataset already exists: {expected}")
        return
    metadata_path = source.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["env_info"]["env_kwargs"]["reward_mode"] = "none"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", str(source),
        "--sim-backend", "physx_cpu",
        "--obs-mode", "state",
        "--target-control-mode", "pd_ee_delta_pose",
        "--reward-mode", "none",
        "--save-traj", "--use-first-env-state",
        "--num-envs", str(conversion_workers),
    ], check=True)
    print(f"Training dataset: {expected}")


def main():
    parser = argparse.ArgumentParser(description="Download and convert official PlugCharger expert demonstrations")
    parser.add_argument("--data-dir", type=Path, default=Path("data/task_06"))
    parser.add_argument("--conversion-workers", type=int, default=8,
                        help="CPU replay workers for state/action conversion; not GPU simulation")
    args = parser.parse_args()
    prepare(args.data_dir, args.conversion_workers)


if __name__ == "__main__":
    main()
