"""QK / OV weight-matrix analyses for a HookedTransformer.

Shapes:
  W_Q, W_K, W_V: [n_heads, d_model, d_head]
  W_O:           [n_heads, d_head, d_model]

Effective circuits per head ``h`` at layer ``L``:
  QK[h] = W_Q[L, h] @ W_K[L, h].T           → [d_model, d_model]
  OV[h] = W_V[L, h] @ W_O[L, h]             → [d_model, d_model]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformer_lens import HookedTransformer


def qk_circuit(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    W_Q = model.W_Q[layer, head]  # [d_model, d_head]
    W_K = model.W_K[layer, head]
    return (W_Q @ W_K.T).detach().float().cpu()


def ov_circuit(model: HookedTransformer, layer: int, head: int) -> torch.Tensor:
    W_V = model.W_V[layer, head]  # [d_model, d_head]
    W_O = model.W_O[layer, head]  # [d_head, d_model]
    return (W_V @ W_O).detach().float().cpu()


def svd(M: torch.Tensor):
    """Full SVD of a 2D matrix. Returns (U, S, Vh) as numpy arrays."""
    M = M.float().cpu()
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return U.numpy(), S.numpy(), Vh.numpy()


def sym_eig(M: torch.Tensor):
    """Eigendecomposition of the symmetric part (M + M.T)/2. Returns (eigvals, eigvecs).

    Eigenvalues are returned sorted descending.
    """
    M = M.float().cpu()
    sym = 0.5 * (M + M.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    # eigh returns ascending; flip.
    order = torch.argsort(eigvals, descending=True)
    return eigvals[order].numpy(), eigvecs[:, order].numpy()


def projection_energy(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """For each column of ``vectors`` [D, k], return the fraction of its norm
    that lies in the column span of ``basis`` [D, r]. basis columns need not
    be orthogonal; we orthonormalize via QR.
    """
    Q, _ = np.linalg.qr(basis)
    proj = Q @ (Q.T @ vectors)
    num = np.linalg.norm(proj, axis=0) ** 2
    den = np.linalg.norm(vectors, axis=0) ** 2 + 1e-12
    return num / den


def random_basis_like(basis: np.ndarray, seed: int = 0) -> np.ndarray:
    D, r = basis.shape
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((D, r)).astype(np.float32)
    Q, _ = np.linalg.qr(R)
    return Q


__all__ = [
    "qk_circuit",
    "ov_circuit",
    "svd",
    "sym_eig",
    "projection_energy",
    "random_basis_like",
]
