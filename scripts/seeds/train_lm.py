#!/usr/bin/env python3
"""Train one seeded instance of the 2L2H attention-only LM from scratch.

Recipe (from filename hint in HF checkpoint; full details not recorded in the
original training script, so we pick a defensible default):
  optimizer   : AdamW, betas=(0.9, 0.95), weight_decay=1e-2
  learning rate: 5e-4, cosine decay with 1,000-step linear warmup
  batch size  : 256
  steps       : 200,000 (configurable via --steps)
  loss        : cross-entropy at Q position only (labels produced by collate_fn)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from minimal_sae_train import (
    IGNORE_INDEX,
    TrainStream,
    build_model,
    collate_fn,
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_schedule(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--stream-size", type=int, default=100_000)
    ap.add_argument("--stream-offset", type=int, default=500_000,
                    help="Start index for TrainStream, chosen to avoid overlap "
                         "with SAE train/eval index ranges used elsewhere")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd,
    )

    train_stream = TrainStream(size=args.stream_size, offset=args.stream_offset)
    loader = iter(DataLoader(
        train_stream,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    ))

    def _next_batch():
        nonlocal loader
        try:
            return next(loader)
        except StopIteration:
            loader = iter(DataLoader(
                train_stream,
                batch_size=args.batch_size,
                collate_fn=collate_fn,
            ))
            return next(loader)

    model.train()
    t0 = time.time()
    log = []
    for step in range(args.steps):
        lr = lr_schedule(step, args.steps, args.warmup, args.lr)
        for g in optim.param_groups:
            g["lr"] = lr

        tokens, labels = _next_batch()
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(tokens)  # [B, S, V_out]
        # Only the Q position has a non-IGNORE label.
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                # quick on-the-fly accuracy at Q positions
                q_mask = labels != IGNORE_INDEX
                preds = logits.argmax(dim=-1)
                acc = (preds[q_mask] == labels[q_mask]).float().mean().item()
            elapsed = time.time() - t0
            log.append({"step": step, "loss": float(loss.item()), "acc": acc, "lr": lr, "elapsed": elapsed})
            print(f"step={step:7d}  loss={loss.item():.4f}  acc={acc:.3f}  lr={lr:.2e}  t={elapsed:.0f}s")

    ckpt = {"model": model.state_dict(), "seed": args.seed, "args": vars(args), "log": log}
    path = os.path.join(args.out_dir, "model.pt")
    torch.save(ckpt, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
