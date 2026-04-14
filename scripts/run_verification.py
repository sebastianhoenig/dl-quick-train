#!/usr/bin/env python3
"""P0 pre-step: identify which Layer-0 head is address vs payload.

Writes ``experiments/results/verification/head_roles.json`` for downstream
scripts. All subsequent analyses read this file rather than hardcoding head
indices.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from dl_quick_train.analysis.data import build_eval_batch
from dl_quick_train.analysis.patching import head_role_scan
from minimal_sae_train import build_model, load_weights


DEFAULT_OUT = "experiments/results/verification/head_roles.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--offset", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--model-path", type=str, default=None,
                    help="Optional local .pt to load instead of HF checkpoint")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    if args.model_path is not None:
        sd = torch.load(args.model_path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd)
    else:
        load_weights(model, device)

    batch = build_eval_batch(args.n, offset=args.offset)
    roles = head_role_scan(model, batch, device=device, seed=args.seed)

    print(f"Clean accuracy: {roles.clean_accuracy:.4f} (n={roles.n_examples})")
    for h in range(len(roles.address_scores)):
        print(f"  head {h}: address_score={roles.address_scores[h]:.3f}"
              f"  payload_score={roles.payload_scores[h]:.3f}")
    print(f"address_head = L0H{roles.address_head}")
    print(f"payload_head = L0H{roles.payload_head}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(roles.to_dict(), f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
