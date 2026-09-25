#!/usr/bin/env python3
"""Verify that this Python environment can execute a small CUDA computation."""

import torch


print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

device = torch.device("cuda:0")
print(f"GPU: {torch.cuda.get_device_name(device)}")
left = torch.arange(16, device=device, dtype=torch.float32).reshape(4, 4)
right = torch.eye(4, device=device)
result = left @ right
torch.cuda.synchronize(device)
assert torch.equal(result, left)
print(f"4x4 matrix multiplication on {device}: passed (sum={result.sum().item():.0f})")
