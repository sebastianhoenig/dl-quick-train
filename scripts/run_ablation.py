#!/usr/bin/env python3
"""P0.3: mean-ablation of Layer-0 heads at SEP positions."""
from __future__ import annotations

import argparse
import json
import os

import torch

from dl_quick_train.analysis.ablation import run_condition
from dl_quick_train.analysis.data import build_eval_batch
from minimal_sae_train import build_model, load_weights


DEFAULT_ROLES = "experiments/results/verification/head_roles.json"
DEFAULT_MEAN = "experiments/results/ablation/mean_cache.pt"
DEFAULT_OUT = "experiments/results/ablation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--offset", type=int, default=400_000)
    ap.add_argument("--roles", type=str, default=DEFAULT_ROLES)
    ap.add_argument("--mean-cache", type=str, default=DEFAULT_MEAN)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.roles) as f:
        roles = json.load(f)
    addr = int(roles["address_head"])
    pay = int(roles["payload_head"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    manifest = torch.load(args.mean_cache, map_location=device, weights_only=False)
    means = manifest["means"]

    batch = build_eval_batch(args.n, offset=args.offset)

    conditions = [
        ("clean", dict()),
        (f"ablate_L0H{addr}_address_SEP_all",
            dict(ablate_head_z=[addr])),
        (f"ablate_L0H{pay}_payload_SEP_all",
            dict(ablate_head_z=[pay])),
        ("ablate_both_SEP_all",
            dict(ablate_head_z=[addr, pay])),
        (f"ablate_L0H{addr}_address_SEP_non_target",
            dict(ablate_head_z=[addr], sep_subset="non_target")),
        ("ablate_resid_post_SEP_all",
            dict(ablate_resid=True)),
    ]

    results = {}
    for name, kwargs in conditions:
        metrics = run_condition(model, batch, means, device=device, **kwargs)
        results[name] = metrics.to_dict()
        print(f"{name:50s}  acc={metrics.accuracy:.3f}  "
              f"routing={metrics.routing_correct:.3f}  logit_diff={metrics.logit_diff:+.2f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({
            "roles": roles,
            "mean_cache_model_sha256": manifest.get("model_sha256"),
            "n": args.n,
            "results": results,
        }, f, indent=2)

    lines = [
        "| condition | accuracy | routing_correct | logit_diff |",
        "|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['accuracy']:.3f} | {m['routing_correct']:.3f} | "
            f"{m['logit_diff']:+.2f} |"
        )
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}/results.{{json,md}}")


if __name__ == "__main__":
    main()
