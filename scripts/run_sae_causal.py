#!/usr/bin/env python3
"""Causal SAE-recovery test at the composed target-fact SEP.

For a trained SAE at ``blocks.0.hook_resid_post``, measure how ablating
individual SAE features at the target-fact SEP shifts the Layer-1
retrieval-head attention from the Q token back to that SEP.

A well-factored ("address-aligned") SAE should show a *sparse* set of
features whose ablation drops the L1-to-target attention; a superposed
SAE should spread the effect densely.

Reports per run (see ``--out/results.json``):
  - clean_attn_L1_to_target: L1 attention Q→target_fact_sep (clean forward).
  - sae_recon_attn_L1_to_target: same, with resid_post at target SEP
    replaced by full SAE reconstruction. Quantifies the reconstruction
    overhead before any ablation.
  - per_feature_shift: for each of the top-``K`` features (ranked by
    activation count on the batch), the attention shift from baseline
    when that feature is zeroed at the target SEP. Sorted by max-head
    |delta|.
  - random_control: same ablation on ``K`` random unrelated features.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable

import numpy as np
import torch

from dl_quick_train.analysis.data import build_eval_batch, gather_at_positions
from dl_quick_train.analysis.sae_loader import BatchTopKSAE
from minimal_sae_train import build_model, load_weights


def _attn_q_to_target(pat: torch.Tensor, q_pos: torch.Tensor,
                      tgt_pos: torch.Tensor) -> torch.Tensor:
    """[B, H, S, S] → [B, H] attention weight from q_pos to tgt_pos."""
    B, H = pat.shape[0], pat.shape[1]
    rows = torch.arange(B, device=pat.device)
    at_q = pat[rows, :, q_pos, :]  # [B, H, S]
    return at_q.gather(
        -1, tgt_pos[:, None, None].expand(B, H, 1)
    ).squeeze(-1)  # [B, H]


def _make_patch_hook(new_at_sep: torch.Tensor, tgt_pos: torch.Tensor) -> Callable:
    """Returns a TL forward hook that overwrites resid_post at tgt_pos with
    ``new_at_sep`` ([B, D])."""
    def hook(resid, hook=None):
        resid = resid.clone()
        B = resid.shape[0]
        rows = torch.arange(B, device=resid.device)
        resid[rows, tgt_pos, :] = new_at_sep.to(resid.dtype)
        return resid
    return hook


@torch.no_grad()
def _run_patched(model, site: str, tokens: torch.Tensor,
                 new_at_sep: torch.Tensor, tgt_pos: torch.Tensor) -> torch.Tensor:
    """Run with resid_post at target SEP replaced; return L1 attention pattern."""
    captured = {}

    def cap_hook(p, hook=None):
        captured["pat"] = p.detach()
        return p

    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (site, _make_patch_hook(new_at_sep, tgt_pos)),
            ("blocks.1.attn.hook_pattern", cap_hook),
        ],
    )
    return captured["pat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--offset", type=int, default=600_000)
    ap.add_argument("--site", default="blocks.0.hook_resid_post")
    ap.add_argument("--sae-ckpt", required=True)
    ap.add_argument("--activation-dim", type=int, default=256)
    ap.add_argument("--dict-size", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--top-features", type=int, default=64,
                    help="Number of most-active features to probe individually.")
    ap.add_argument("--out", default="experiments/results/sae_causal")
    ap.add_argument("--model-ckpt", default=None,
                    help="Local LM checkpoint; defaults to released 2L2H_Final.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    load_weights(model, device, ckpt_path=args.model_ckpt)
    model.eval()

    sae = BatchTopKSAE(
        activation_dim=args.activation_dim, dict_size=args.dict_size, k=args.k
    ).to(device)
    sae.load_state_dict(
        torch.load(args.sae_ckpt, map_location=device, weights_only=True)
    )
    sae.eval()

    batch = build_eval_batch(args.n, offset=args.offset).to(device)
    tgt_pos = batch.target_fact_sep_position
    q_pos = batch.q_position

    # --- Clean forward: cache resid at target SEP + L1 attention.
    with torch.no_grad():
        _, cache = model.run_with_cache(
            batch.tokens,
            names_filter=[args.site, "blocks.1.attn.hook_pattern"],
        )
    resid_at_sep = gather_at_positions(cache[args.site], tgt_pos)  # [B, D]
    clean_pat = cache["blocks.1.attn.hook_pattern"]
    clean_attn = _attn_q_to_target(clean_pat, q_pos, tgt_pos).mean(0).cpu().numpy()

    # --- SAE encode/decode baseline.
    with torch.no_grad():
        features = sae.encode(resid_at_sep)
        if isinstance(features, tuple):
            features = features[0]
        recon_full = sae.decode(features)
        if isinstance(recon_full, tuple):
            recon_full = recon_full[0]

    base_pat = _run_patched(model, args.site, batch.tokens, recon_full, tgt_pos)
    base_attn = _attn_q_to_target(base_pat, q_pos, tgt_pos).mean(0).cpu().numpy()

    # --- Identify top-K features by batch-wide activation frequency.
    active_counts = (features != 0).float().sum(0)  # [dict_size]
    top_feats = torch.argsort(active_counts, descending=True)[: args.top_features]
    top_feats = [int(x) for x in top_feats.cpu().tolist()]

    def ablate_and_measure(feat_idx: int) -> np.ndarray:
        f_ab = features.clone()
        f_ab[:, feat_idx] = 0.0
        with torch.no_grad():
            recon = sae.decode(f_ab)
            if isinstance(recon, tuple):
                recon = recon[0]
        pat = _run_patched(model, args.site, batch.tokens, recon, tgt_pos)
        return _attn_q_to_target(pat, q_pos, tgt_pos).mean(0).cpu().numpy()

    per_feat = []
    for f_idx in top_feats:
        attn_ab = ablate_and_measure(f_idx)
        delta = (base_attn - attn_ab).tolist()  # per head
        per_feat.append({
            "feature": f_idx,
            "active_count": int(active_counts[f_idx].item()),
            "delta_attn_L1": delta,
        })
    per_feat.sort(key=lambda d: -max(abs(x) for x in d["delta_attn_L1"]))

    # --- Random control: K features that are unused on this batch.
    rng = np.random.default_rng(0)
    unused = [i for i in range(args.dict_size) if active_counts[i].item() == 0]
    if len(unused) >= args.top_features:
        rand_feats = rng.choice(unused, size=args.top_features, replace=False).tolist()
    else:
        rand_feats = rng.choice(args.dict_size, size=args.top_features, replace=False).tolist()
    rand_shifts = []
    for f_idx in rand_feats:
        attn_ab = ablate_and_measure(int(f_idx))
        rand_shifts.append({
            "feature": int(f_idx),
            "delta_attn_L1": (base_attn - attn_ab).tolist(),
        })

    # --- Summary stats.
    def _max_abs(shifts):
        return [max(abs(x) for x in s["delta_attn_L1"]) for s in shifts]
    top_mags = _max_abs(per_feat)
    rand_mags = _max_abs(rand_shifts)
    summary = {
        "clean_attn_L1_to_target": clean_attn.tolist(),
        "sae_recon_attn_L1_to_target": base_attn.tolist(),
        "reconstruction_overhead": (clean_attn - base_attn).tolist(),
        "top_feature_shift_mean": float(np.mean(top_mags)),
        "top_feature_shift_max": float(np.max(top_mags)),
        "random_control_shift_mean": float(np.mean(rand_mags)),
        "random_control_shift_max": float(np.max(rand_mags)),
        "n_top_features_above_random_max": int(
            sum(1 for m in top_mags if m > max(rand_mags, default=0.0))
        ),
    }

    out = {
        "sae_ckpt": args.sae_ckpt,
        "site": args.site,
        "n": args.n,
        "top_features_probed": top_feats,
        "summary": summary,
        "per_feature_shift": per_feat,
        "random_control": rand_shifts,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("top-5 address-like features (by |delta_attn| on L1):")
    for s in per_feat[:5]:
        print(f"  feat {s['feature']:5d}  active={s['active_count']:4d}"
              f"  delta={['%+.3f' % x for x in s['delta_attn_L1']]}")
    print(f"\nwrote {args.out}/results.json")


if __name__ == "__main__":
    main()
