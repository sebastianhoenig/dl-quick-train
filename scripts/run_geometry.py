#!/usr/bin/env python3
"""P0.1: geometry of isolated vs composed activations at the target-fact SEP."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dl_quick_train.analysis.data import build_eval_batch
from dl_quick_train.analysis.geometry import (
    extract_site_activations,
    pairwise_cosine_stats,
    pca_2d,
)
from minimal_sae_train import build_model, load_weights


DEFAULT_OUT = "experiments/results/geometry"
DEFAULT_ROLES = "experiments/results/verification/head_roles.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--offset", type=int, default=300_000)
    ap.add_argument("--roles", type=str, default=DEFAULT_ROLES)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.roles) as f:
        roles = json.load(f)
    address_head = int(roles["address_head"])
    payload_head = int(roles["payload_head"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    batch = build_eval_batch(args.n, offset=args.offset)
    acts = extract_site_activations(model, batch, address_head, payload_head, device)

    os.makedirs(args.out, exist_ok=True)
    summary = {}
    for key, tensor in acts.items():
        stats = pairwise_cosine_stats(tensor, max_pairs=4_000_000)
        summary[key] = {"shape": list(tensor.shape), **stats.to_dict()}
        pca = pca_2d(tensor)
        np.save(os.path.join(args.out, f"pca_{key}.npy"), pca)

    # Metadata for per-point coloring in plots.
    meta = {
        "Eq": batch.eq_token.tolist(),
        "Tq": batch.tq_token.tolist(),
        "E2q": batch.labels.tolist(),
        "address_head": address_head,
        "payload_head": payload_head,
        "n": args.n,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(args.out, "cosine_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Pairwise cosine (off-diagonal) at target-fact SEP:")
    for key, s in summary.items():
        print(f"  {key:10s}  mean={s['mean']:+.3f}  p5={s['p5']:+.3f}  "
              f"p95={s['p95']:+.3f}  std={s['std']:.3f}  shape={s['shape']}")
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
