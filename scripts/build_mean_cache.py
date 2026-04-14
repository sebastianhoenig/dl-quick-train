#!/usr/bin/env python3
"""Cache per-site reference means over a fixed eval distribution.

Writes ``experiments/results/ablation/mean_cache.pt`` with a manifest that
includes the example-index range, a hash of the model state-dict, and the
resulting per-site means.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time

import torch

from dl_quick_train.analysis.ablation import compute_reference_means
from dl_quick_train.analysis.data import build_eval_batch
from minimal_sae_train import build_model, load_weights


DEFAULT_OUT = "experiments/results/ablation/mean_cache.pt"


def state_dict_sha256(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--offset", type=int, default=100_000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    batch = build_eval_batch(args.n, offset=args.offset)
    means = compute_reference_means(model, batch, device=device, chunk=args.chunk)

    manifest = {
        "means": means,
        "indices": {"offset": args.offset, "n": args.n},
        "model_sha256": state_dict_sha256(model.state_dict()),
        "timestamp": time.time(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(manifest, args.out)
    print(f"n_positions used in mean: {int(means['n_positions'].item())}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
