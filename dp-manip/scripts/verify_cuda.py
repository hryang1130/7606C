#!/usr/bin/env python3
"""Print the training runtime and fail when the requested CUDA device is unusable."""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true", help="environment-only check on a login node")
    args = parser.parse_args()
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        if args.allow_cpu:
            print("CUDA is not visible here; run training only inside a GPU allocation")
            return
        raise SystemExit("CUDA is required for RGB training")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"cuda:{index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
    left = torch.randn(64, 64, device="cuda")
    right = torch.randn(64, 64, device="cuda")
    result = left @ right
    if not torch.isfinite(result).all():
        raise SystemExit("CUDA smoke calculation produced non-finite values")


if __name__ == "__main__":
    main()
