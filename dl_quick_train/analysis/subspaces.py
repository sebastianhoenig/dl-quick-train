"""Address / payload subspaces from Layer-0 OV circuits.

The address subspace is the span of the top-``rank`` right singular vectors of
the address head's effective OV (``W_V @ W_O``), which is the set of directions
in the residual stream that the address head can *write into*. Payload
subspace is analogous for the payload head.
"""
from __future__ import annotations

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .weights import ov_circuit, svd


def address_basis(model: HookedTransformer, address_head: int, rank: int = 32, layer: int = 0) -> np.ndarray:
    """Returns [d_model, rank]: top-``rank`` right singular vectors of
    ``OV[layer, address_head]``.
    """
    ov = ov_circuit(model, layer=layer, head=address_head)
    _, _, Vh = svd(ov)
    return Vh[:rank].T  # Vh: [d_model, d_model]; rows are right sing. vectors.


def payload_basis(model: HookedTransformer, payload_head: int, rank: int = 32, layer: int = 0) -> np.ndarray:
    ov = ov_circuit(model, layer=layer, head=payload_head)
    _, _, Vh = svd(ov)
    return Vh[:rank].T


__all__ = ["address_basis", "payload_basis"]
