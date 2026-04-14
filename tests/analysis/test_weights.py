import numpy as np
import torch

from dl_quick_train.analysis.weights import (
    projection_energy,
    qk_circuit,
    ov_circuit,
    random_basis_like,
    sym_eig,
    svd,
)
from minimal_sae_train import build_model


def test_qk_ov_shapes():
    model = build_model()
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            qk = qk_circuit(model, L, h)
            ov = ov_circuit(model, L, h)
            assert qk.shape == (model.cfg.d_model, model.cfg.d_model)
            assert ov.shape == (model.cfg.d_model, model.cfg.d_model)


def test_sym_eig_is_real_descending():
    M = torch.randn(8, 8)
    ev, _ = sym_eig(M)
    assert np.all(ev[:-1] >= ev[1:])  # descending
    assert np.isrealobj(ev)


def test_projection_energy_bounds():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((16, 4)).astype(np.float32)
    vecs = rng.standard_normal((16, 3)).astype(np.float32)
    e = projection_energy(vecs, basis)
    assert e.shape == (3,)
    assert np.all(e >= 0) and np.all(e <= 1 + 1e-6)


def test_random_basis_orthonormal():
    rb = random_basis_like(np.zeros((32, 8), dtype=np.float32), seed=0)
    G = rb.T @ rb
    assert np.allclose(G, np.eye(8), atol=1e-5)
