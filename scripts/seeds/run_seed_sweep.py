#!/usr/bin/env python3
"""Train N seeded instances of the 2L2H LM and run the verification pre-step
on each. Produces a markdown table summarizing per-seed role assignments.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

from dl_quick_train.analysis.data import build_eval_batch
from dl_quick_train.analysis.patching import head_role_scan
from minimal_sae_train import build_model


DEFAULT_OUT = "experiments/results/seeds"


def train_seed(seed: int, out_dir: str, steps: int, extra_args):
    cmd = [sys.executable, "scripts/seeds/train_lm.py",
           "--seed", str(seed),
           "--steps", str(steps),
           "--out-dir", out_dir] + list(extra_args)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})


def verify_checkpoint(ckpt_path, n=1024) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)["model"]
    model.load_state_dict(sd)
    batch = build_eval_batch(n, offset=200_000)
    roles = head_role_scan(model, batch, device=device)
    return roles.to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--skip-train", action="store_true",
                    help="Only run verification on existing checkpoints under --out")
    ap.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args forwarded to train_lm.py after --")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for s in args.seeds:
        seed_dir = os.path.join(args.out, f"seed_{s}")
        os.makedirs(seed_dir, exist_ok=True)
        if not args.skip_train:
            extra = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
            train_seed(s, seed_dir, args.steps, extra)
        ckpt = os.path.join(seed_dir, "model.pt")
        if not os.path.exists(ckpt):
            print(f"[seed {s}] no checkpoint at {ckpt}, skipping verification")
            continue
        roles = verify_checkpoint(ckpt)
        with open(os.path.join(seed_dir, "roles.json"), "w") as f:
            json.dump(roles, f, indent=2)
        emerged = (
            abs(roles["address_scores"][roles["address_head"]]) > 1.0
            and abs(roles["payload_scores"][roles["payload_head"]]) > 1.0
            and roles["clean_accuracy"] > 0.95
        )
        rows.append((s, roles, emerged))

    # Markdown table.
    lines = [
        "| seed | acc | address_head | payload_head | addr_score | pay_score | circuit_emerged |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, r, e in rows:
        lines.append(
            f"| {s} | {r['clean_accuracy']:.3f} | L0H{r['address_head']} | L0H{r['payload_head']} | "
            f"{r['address_scores'][r['address_head']]:+.2f} | "
            f"{r['payload_scores'][r['payload_head']]:+.2f} | "
            f"{'yes' if e else 'no'} |"
        )
    with open(os.path.join(args.out, "table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
