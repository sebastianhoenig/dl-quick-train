import torch

from dl_quick_train.analysis.data import (
    _produce_example_with_meta,
    build_eval_batch,
)
from minimal_sae_train import Q, SEP, produce_example_by_index


def test_meta_matches_original_generator():
    for idx in [0, 1, 42, 1000, 99_999]:
        seq_o, lbl_o = produce_example_by_index(idx)
        meta = _produce_example_with_meta(idx)
        assert meta["seq"] == seq_o
        assert meta["label"] == lbl_o


def test_target_fact_sep_positions_are_SEP():
    for idx in range(0, 200):
        meta = _produce_example_with_meta(idx)
        p = meta["target_fact_sep"]
        assert meta["seq"][p] == SEP
        assert meta["seq"][p - 1] == meta["label"]  # e2 == E2q
        assert meta["seq"][p - 2] == meta["Tq"]
        assert meta["seq"][p - 3] == meta["Eq"]


def test_build_eval_batch_shapes():
    b = build_eval_batch(16, offset=100)
    assert b.tokens.ndim == 2 and b.tokens.shape[0] == 16
    assert b.labels.shape == (16,)
    assert b.q_position.shape == (16,)
    for i in range(16):
        L = int(b.lengths[i].item())
        # Q at last position of the actual sequence
        assert int(b.tokens[i, L - 1].item()) == Q
        assert int(b.q_position[i].item()) == L - 1
        # target SEP is a SEP
        p = int(b.target_fact_sep_position[i].item())
        assert int(b.tokens[i, p].item()) == SEP
