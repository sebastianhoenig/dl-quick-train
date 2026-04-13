#!/usr/bin/env python3
"""Apply a Gemma Scope residual SAE at the address-router's layer and score
whether the composed address state is recoverable from SAE features.

Metrics:
  - reconstruction MSE (normalized) at the retrieval token
  - F1 gap: logistic probe on SAE features vs raw residual, predicting the
    retrieved tail entity among in-prompt candidates.

NOTE: requires ``sae_lens`` for Gemma Scope loading.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch

try:
    from transformer_lens import HookedTransformer
except Exception as e:
    raise SystemExit("transformer_lens required") from e

try:
    from sae_lens import SAE
except Exception as e:
    raise SystemExit("sae_lens required: pip install sae_lens") from e

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scripts.gemma.gemma_prompts import (
    DEFAULT_HEADS,
    DEFAULT_TAILS,
    build_prompt_set,
)


DEFAULT_OUT = "experiments/results/gemma/summary.json"
DEFAULT_ROLES = "experiments/results/gemma/gemma_head_roles.json"


@torch.no_grad()
def _collect_residuals(model, prompts, tok, layer, device, batch_size=8):
    pad_id = tok.pad_token_id or 0
    name = f"blocks.{layer}.hook_resid_post"
    residuals, labels_out = [], []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        ids_list = [tok(p.text, add_special_tokens=False)["input_ids"] for p in chunk]
        max_len = max(len(x) for x in ids_list)
        padded = [x + [pad_id] * (max_len - len(x)) for x in ids_list]
        last = [len(x) - 1 for x in ids_list]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        _, cache = model.run_with_cache(ids, names_filter=[name])
        resid = cache[name]  # [b, S, d_model]
        for b, p in enumerate(chunk):
            residuals.append(resid[b, last[b], :].float().cpu().numpy())
            labels_out.append(
                tok(p.target_token, add_special_tokens=False)["input_ids"][0]
            )
    return np.stack(residuals), np.array(labels_out)


def _probe_f1(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, n_jobs=1).fit(Xtr, ytr)
    return float(f1_score(yte, clf.predict(Xte), average="macro"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gemma-2-2b")
    ap.add_argument("--roles", type=str, default=DEFAULT_ROLES)
    ap.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res")
    ap.add_argument("--sae-id", type=str, default=None,
                    help="Gemma Scope sae_id; if None, pick the 'width-16k, "
                         "average_l0 closest to 100' variant for the router's layer.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.roles) as f:
        roles = json.load(f)
    layer = int(roles["address_router"]["layer"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    tok = model.tokenizer

    sae_id = args.sae_id or f"layer_{layer}/width_16k/average_l0_100"
    sae, _, _ = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id, device=device)
    sae.eval()

    prompts = build_prompt_set(DEFAULT_HEADS, DEFAULT_TAILS, n_prompts=args.n * 2)
    # Filter to single-token tails.
    prompts = [
        p for p in prompts
        if len(tok(p.target_token, add_special_tokens=False)["input_ids"]) == 1
    ][: args.n]
    print(f"{len(prompts)} prompts after filtering")

    X_resid, y = _collect_residuals(model, prompts, tok, layer, device)
    with torch.no_grad():
        X_resid_t = torch.from_numpy(X_resid).to(device, dtype=sae.W_enc.dtype)
        recon = sae(X_resid_t)
        if isinstance(recon, tuple):
            recon = recon[0]
        features = sae.encode(X_resid_t).float().cpu().numpy()
        recon_np = recon.float().cpu().numpy()

    mse = float(np.mean((X_resid - recon_np) ** 2) / (np.mean(X_resid ** 2) + 1e-9))

    f1_raw = _probe_f1(X_resid, y)
    f1_sae = _probe_f1(features, y)

    out = {
        "model": args.model,
        "sae_release": args.sae_release,
        "sae_id": sae_id,
        "layer": layer,
        "n": len(prompts),
        "mse_normalized": mse,
        "f1_raw": f1_raw,
        "f1_sae": f1_sae,
        "f1_gap": f1_raw - f1_sae,
    }
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
