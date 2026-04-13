#!/usr/bin/env python3
"""Evaluate existing SAE checkpoints on the paper's core thesis: can Top-k SAEs
recover the composed (E1+T) state as well as they recover each isolated
variable?

Iterates the full sweep directories under ``sae_ckpts/`` and, per trained
SAE (dict_size x k), reports:
  - reconstruction: normalized MSE, fraction variance explained, mean L0
  - recovery F1: logistic probe on SAE features predicting {Eq, Tq, E2q}
  - raw-baseline F1: the same probe trained on the raw activation vectors

A large (f1_raw − f1_sae) gap on the **composed** site (resid_post) despite
small gaps on the **isolated** site (hook_z per-head) is exactly the
"dark matter" failure mode the paper argues for.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from dl_quick_train.analysis.data import build_eval_batch, gather_at_positions
from dl_quick_train.analysis.sae_loader import BatchTopKSAE
from minimal_sae_train import build_model, load_weights


DEFAULT_OUT = "experiments/results/sae_eval"


SITE_MAP = {
    "b0_residpost_sep_sweep_d1024-4096_k8-16": {
        "hook_name": "blocks.0.hook_resid_post",
        "head_index": None,
        "activation_dim": 256,
        "label": "composed_E1_plus_T",
    },
    "b0_hookz_h0_sep_sweep_d1024-4096_k8-16": {
        "hook_name": "blocks.0.attn.hook_z",
        "head_index": 0,
        "activation_dim": 128,
        "label": "isolated_H0",
    },
    "b0_hookz_h1_sep_sweep_d1024-4096_k8-16": {
        "hook_name": "blocks.0.attn.hook_z",
        "head_index": 1,
        "activation_dim": 128,
        "label": "isolated_H1",
    },
}


def _probe_f1(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    # Guard against degenerate classes.
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if counts.min() >= 2 else None,
    )
    clf = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, ytr)
    return float(f1_score(yte, clf.predict(Xte), average="macro"))


@dataclass
class SaeResult:
    sweep: str
    site_label: str
    trainer_dir: str
    step: int
    dict_size: int
    k: int
    activation_dim: int
    n_eval: int
    l0_mean: float
    nmse: float
    fvu: float   # fraction of variance unexplained
    f1_sae_Eq: float
    f1_sae_Tq: float
    f1_sae_E2q: float
    f1_raw_Eq: float
    f1_raw_Tq: float
    f1_raw_E2q: float

    def to_dict(self) -> dict:
        return self.__dict__


@torch.no_grad()
def _extract_activations(model, batch, device) -> Dict[str, torch.Tensor]:
    """Return activations at target-fact SEP for the two required sites."""
    batch = batch.to(device)
    names = ["blocks.0.hook_resid_post", "blocks.0.attn.hook_z"]
    _, cache = model.run_with_cache(batch.tokens, names_filter=names)
    resid = gather_at_positions(cache["blocks.0.hook_resid_post"], batch.target_fact_sep_position)
    z = gather_at_positions(cache["blocks.0.attn.hook_z"], batch.target_fact_sep_position)
    return {
        "blocks.0.hook_resid_post": resid,          # [N, 256]
        "blocks.0.attn.hook_z": z,                  # [N, n_heads, d_head]
    }


def _parse_cfg(trainer_dir: str) -> Tuple[int, int, int]:
    with open(os.path.join(trainer_dir, "config.json")) as f:
        cfg = json.load(f)["trainer"]
    return int(cfg["activation_dim"]), int(cfg["dict_size"]), int(cfg["k"])


def _latest_ckpt(trainer_dir: str) -> Optional[Tuple[str, int]]:
    paths = glob.glob(os.path.join(trainer_dir, "checkpoints", "ae_*.pt"))
    if not paths:
        return None
    best = max(paths, key=lambda p: int(re.search(r"ae_(\d+)\.pt$", p).group(1)))
    step = int(re.search(r"ae_(\d+)\.pt$", best).group(1))
    return best, step


def _load_sae(ckpt_path: str, activation_dim: int, dict_size: int, k: int, device):
    sae = BatchTopKSAE(activation_dim=activation_dim, dict_size=dict_size, k=k).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    sae.load_state_dict(sd)
    sae.eval()
    return sae


@torch.no_grad()
def _evaluate_sae(
    sae,
    raw: torch.Tensor,
    labels: Dict[str, np.ndarray],
) -> Dict[str, float]:
    x = raw.to(next(sae.parameters()).device, dtype=torch.float32)
    recon = sae(x)
    if isinstance(recon, tuple):
        recon = recon[0]
    f = sae.encode(x)
    if isinstance(f, tuple):
        f = f[0]

    # Reconstruction stats.
    diff = x - recon
    nmse = float((diff.pow(2).sum(dim=-1) / (x.pow(2).sum(dim=-1) + 1e-9)).mean().item())
    var_total = x.var(dim=0).sum()
    var_resid = diff.var(dim=0).sum()
    fvu = float((var_resid / (var_total + 1e-9)).item())
    l0 = float((f != 0).float().sum(dim=-1).mean().item())

    feat_np = f.float().cpu().numpy()
    raw_np = x.float().cpu().numpy()
    out = {"nmse": nmse, "fvu": fvu, "l0_mean": l0}
    for key, y in labels.items():
        out[f"f1_sae_{key}"] = _probe_f1(feat_np, y)
        out[f"f1_raw_{key}"] = _probe_f1(raw_np, y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="sae_ckpts")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--offset", type=int, default=600_000)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--sweeps", nargs="*", default=None,
                    help="Optional subset of sweep directory names")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    batch = build_eval_batch(args.n, offset=args.offset)
    acts = _extract_activations(model, batch, device)

    labels = {
        "Eq": batch.eq_token.cpu().numpy(),
        "Tq": (batch.tq_token.cpu().numpy() - 100),   # shift relation ids to 0..T-1
        "E2q": batch.labels.cpu().numpy(),
    }

    sweeps = args.sweeps or sorted(os.listdir(args.ckpt_root))
    results: List[SaeResult] = []
    for sweep in sweeps:
        info = SITE_MAP.get(sweep)
        if info is None:
            print(f"skipping unknown sweep dir {sweep}")
            continue
        raw_act = acts[info["hook_name"]]
        if info["head_index"] is not None:
            raw_act = raw_act[:, info["head_index"], :]

        sweep_dir = os.path.join(args.ckpt_root, sweep)
        trainer_dirs = sorted(
            d for d in glob.glob(os.path.join(sweep_dir, "trainer_*"))
            if os.path.isdir(d)
        )
        for tdir in trainer_dirs:
            act_dim, dict_size, k = _parse_cfg(tdir)
            if act_dim != info["activation_dim"]:
                print(f"skip {tdir}: activation_dim mismatch")
                continue
            ck = _latest_ckpt(tdir)
            if ck is None:
                print(f"skip {tdir}: no checkpoints")
                continue
            ckpt_path, step = ck
            sae = _load_sae(ckpt_path, act_dim, dict_size, k, device)
            metrics = _evaluate_sae(sae, raw_act, labels)

            r = SaeResult(
                sweep=sweep,
                site_label=info["label"],
                trainer_dir=os.path.relpath(tdir),
                step=step,
                dict_size=dict_size,
                k=k,
                activation_dim=act_dim,
                n_eval=args.n,
                l0_mean=metrics["l0_mean"],
                nmse=metrics["nmse"],
                fvu=metrics["fvu"],
                f1_sae_Eq=metrics["f1_sae_Eq"],
                f1_sae_Tq=metrics["f1_sae_Tq"],
                f1_sae_E2q=metrics["f1_sae_E2q"],
                f1_raw_Eq=metrics["f1_raw_Eq"],
                f1_raw_Tq=metrics["f1_raw_Tq"],
                f1_raw_E2q=metrics["f1_raw_E2q"],
            )
            results.append(r)
            print(f"[{info['label']:20s}  d={dict_size:<4d} k={k:<2d} step={step:>8d}]"
                  f"  fvu={r.fvu:.3f}  L0={r.l0_mean:5.1f}"
                  f"  Eq gap={r.f1_raw_Eq - r.f1_sae_Eq:+.3f}"
                  f"  Tq gap={r.f1_raw_Tq - r.f1_sae_Tq:+.3f}"
                  f"  E2q gap={r.f1_raw_E2q - r.f1_sae_E2q:+.3f}")

    # JSON dump
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)

    # Markdown table
    lines = [
        "| site | dict | k | step | L0 | FVU | F1(raw Eq) | F1(sae Eq) | gap Eq | F1(raw Tq) | F1(sae Tq) | gap Tq | F1(raw E2q) | F1(sae E2q) | gap E2q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.site_label} | {r.dict_size} | {r.k} | {r.step} | "
            f"{r.l0_mean:.1f} | {r.fvu:.3f} | "
            f"{r.f1_raw_Eq:.3f} | {r.f1_sae_Eq:.3f} | {r.f1_raw_Eq - r.f1_sae_Eq:+.3f} | "
            f"{r.f1_raw_Tq:.3f} | {r.f1_sae_Tq:.3f} | {r.f1_raw_Tq - r.f1_sae_Tq:+.3f} | "
            f"{r.f1_raw_E2q:.3f} | {r.f1_sae_E2q:.3f} | {r.f1_raw_E2q - r.f1_sae_E2q:+.3f} |"
        )
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nwrote {args.out}/results.{{json,md}}  ({len(results)} SAEs evaluated)")


if __name__ == "__main__":
    main()
