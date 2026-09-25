#!/usr/bin/env python3
"""Run a few real CUDA optimizer steps; optionally repeat with CUDA AMP."""

import argparse
from contextlib import nullcontext

import torch


def run(amp: bool) -> None:
    torch.manual_seed(7606)
    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
    ).to(device)
    inputs = torch.randn(128, 32, device=device)
    targets = torch.randn(128, 16, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    losses = []

    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            prediction = model(inputs)
            loss = torch.nn.functional.mse_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is not finite")
        if amp:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        ):
            raise RuntimeError("A gradient is missing or non-finite")
        if amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.append(float(loss.detach()))

    torch.cuda.synchronize(device)
    parameters_changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial_parameters, model.parameters())
    )
    if not parameters_changed:
        raise RuntimeError("Optimizer steps did not change model parameters")
    print(f"Mode: {'AMP fp16' if amp else 'FP32'}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(device)}")
    print(f"Losses: {[round(value, 6) for value in losses]}")
    print("Finite loss and gradients: PASS")
    print("Parameters changed: PASS")
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast and gradient scaling")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    run(args.amp)


if __name__ == "__main__":
    main()
