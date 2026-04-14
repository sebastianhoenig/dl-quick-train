"""Geometric analysis of isolated vs composed activations."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .data import EvalBatch, gather_at_positions


@dataclass
class CosineStats:
    mean: float
    p5: float
    p50: float
    p95: float
    std: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


def pairwise_cosine_stats(x: torch.Tensor, max_pairs: int = 4_000_000) -> CosineStats:
    """Off-diagonal pairwise cosine similarity stats.

    x: [N, D]. If N*(N-1)/2 > max_pairs, subsamples pairs uniformly.
    """
    x = x.float()
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
    N = x.shape[0]
    if N * (N - 1) // 2 > max_pairs:
        # subsample rows to cap pair count
        keep = int(np.floor((1 + np.sqrt(1 + 8 * max_pairs)) / 2))
        perm = torch.randperm(N)[:keep]
        x = x[perm]
        N = keep
    cos = x @ x.T
    idx = torch.triu_indices(N, N, offset=1)
    vals = cos[idx[0], idx[1]].cpu().numpy()
    return CosineStats(
        mean=float(vals.mean()),
        p5=float(np.percentile(vals, 5)),
        p50=float(np.percentile(vals, 50)),
        p95=float(np.percentile(vals, 95)),
        std=float(vals.std()),
        n=int(vals.size),
    )


def pca_2d(x: torch.Tensor) -> np.ndarray:
    """Project x: [N, D] onto top-2 PCs (centered, unscaled). Returns [N, 2]."""
    x = x.float().cpu().numpy()
    x = x - x.mean(axis=0, keepdims=True)
    # SVD on centered data
    U, S, Vt = np.linalg.svd(x, full_matrices=False)
    return (U[:, :2] * S[:2])


@torch.no_grad()
def extract_site_activations(
    model: HookedTransformer,
    batch: EvalBatch,
    address_head: int,
    payload_head: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Run the model and return:

    - E1_z: L0H{address} hook_z at target-fact SEP, [N, d_head]
    - T_z:  L0H{payload} hook_z at target-fact SEP, [N, d_head]
    - E1_plus_T: blocks.0.hook_resid_post at target-fact SEP, [N, d_model]
    """
    model.eval()
    batch = batch.to(device)
    names = ["blocks.0.attn.hook_z", "blocks.0.hook_resid_post"]
    _, cache = model.run_with_cache(batch.tokens, names_filter=names)
    z = cache["blocks.0.attn.hook_z"]               # [B, S, H, d_head]
    resid = cache["blocks.0.hook_resid_post"]       # [B, S, d_model]

    # Gather per-row at target_fact_sep_position.
    z_at_sep = gather_at_positions(z, batch.target_fact_sep_position)
    resid_at_sep = gather_at_positions(resid, batch.target_fact_sep_position)
    return {
        "E1_z": z_at_sep[:, address_head, :].contiguous(),
        "T_z": z_at_sep[:, payload_head, :].contiguous(),
        "E1_plus_T": resid_at_sep.contiguous(),
    }


__all__ = [
    "CosineStats",
    "pairwise_cosine_stats",
    "pca_2d",
    "extract_site_activations",
]
