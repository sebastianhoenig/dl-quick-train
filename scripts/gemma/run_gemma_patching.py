#!/usr/bin/env python3
"""Find Gemma-2B's retrieval-head for a natural-language relational task.

Algorithm (per-head denoising):
  1. Run clean prompts → cache ``blocks.{L}.attn.hook_z`` for all L.
  2. Run corrupted prompts (same facts, different query head) and at each
     (layer, head) patch in the clean ``hook_z`` slice at the retrieval token.
  3. Score: recovery of the clean target token logit relative to corrupted
     baseline. The head with the largest average recovery is the retrieval
     ("address-router") head.

Writes ``experiments/results/gemma/gemma_head_roles.json``.

NOTE: downloads Gemma-2B and runs on GPU (bf16). Budget ~5 GB VRAM.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

try:
    from transformer_lens import HookedTransformer
except Exception as e:  # pragma: no cover
    raise SystemExit("transformer_lens required: pip install transformer_lens") from e

from scripts.gemma.gemma_prompts import (
    DEFAULT_HEADS,
    DEFAULT_TAILS,
    RelationalPrompt,
    build_prompt_set,
)


DEFAULT_OUT = "experiments/results/gemma/gemma_head_roles.json"


def _tokenize_single(tok, s: str) -> int:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"not single-token: {s!r} → {ids}")
    return ids[0]


def _filter_prompts(prompts, tok):
    out = []
    for p in prompts:
        try:
            _tokenize_single(tok, p.target_token)
            _tokenize_single(tok, p.corrupted_target_token)
        except ValueError:
            continue
        out.append(p)
    return out


def _pad_ids(ids_list, pad_id):
    max_len = max(len(x) for x in ids_list)
    out = [x + [pad_id] * (max_len - len(x)) for x in ids_list]
    last = [len(x) - 1 for x in ids_list]
    return out, last


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gemma-2-2b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device, dtype=torch.bfloat16)
    tok = model.tokenizer

    prompts = build_prompt_set(DEFAULT_HEADS, DEFAULT_TAILS, n_prompts=args.n * 2)
    prompts = _filter_prompts(prompts, tok)[: args.n]
    if not prompts:
        raise SystemExit("No prompts survived single-token filtering.")
    print(f"Using {len(prompts)} prompts after single-token filtering")

    clean_ids = [tok(p.text, add_special_tokens=False)["input_ids"] for p in prompts]
    corr_ids = [tok(p.corrupted_text, add_special_tokens=False)["input_ids"] for p in prompts]
    targets = [_tokenize_single(tok, p.target_token) for p in prompts]
    corr_targets = [_tokenize_single(tok, p.corrupted_target_token) for p in prompts]

    pad_id = tok.pad_token_id or 0
    clean_ids, clean_last = _pad_ids(clean_ids, pad_id)
    corr_ids, corr_last = _pad_ids(corr_ids, pad_id)

    clean_ids_t = torch.tensor(clean_ids, dtype=torch.long, device=device)
    corr_ids_t = torch.tensor(corr_ids, dtype=torch.long, device=device)
    clean_last_t = torch.tensor(clean_last, dtype=torch.long, device=device)
    corr_last_t = torch.tensor(corr_last, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    names = [f"blocks.{L}.attn.hook_z" for L in range(n_layers)]

    # Clean forward with cache.
    _, clean_cache = model.run_with_cache(clean_ids_t, names_filter=names)
    # Corrupted baseline.
    corr_logits = model(corr_ids_t)

    B = clean_ids_t.shape[0]
    rows = torch.arange(B, device=device)

    baseline_clean = model(clean_ids_t)[rows, clean_last_t, targets_t].float().mean().item()
    baseline_corr = corr_logits[rows, corr_last_t, targets_t].float().mean().item()
    print(f"baseline_clean_logit={baseline_clean:+.2f}  baseline_corr_logit={baseline_corr:+.2f}")

    recovery = torch.zeros(n_layers, n_heads)
    for L in range(n_layers):
        name = f"blocks.{L}.attn.hook_z"
        clean_z = clean_cache[name]  # [B, S_clean, H, dh]
        for h in range(n_heads):
            def hook(z, hook=None, L=L, h=h):
                z = z.clone()
                # Copy from clean's retrieval token into corrupted's retrieval token.
                z[rows, corr_last_t, h, :] = clean_z[rows, clean_last_t, h, :].to(
                    z.device, z.dtype
                )
                return z

            patched = model.run_with_hooks(corr_ids_t, fwd_hooks=[(name, hook)])
            patched_target = patched[rows, corr_last_t, targets_t].float().mean().item()
            recovery[L, h] = patched_target - baseline_corr

    max_flat = int(torch.argmax(recovery).item())
    best_L = max_flat // n_heads
    best_H = max_flat % n_heads
    print(f"best head = L{best_L} H{best_H}  recovery={recovery[best_L, best_H].item():+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "model": args.model,
            "address_router": {"layer": best_L, "head": best_H},
            "recovery_matrix": recovery.tolist(),
            "baseline_clean_logit": baseline_clean,
            "baseline_corr_logit": baseline_corr,
            "n_prompts": len(prompts),
        }, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
