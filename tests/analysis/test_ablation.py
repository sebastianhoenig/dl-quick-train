import pytest
import torch

from dl_quick_train.analysis.ablation import build_sep_mask, _mean_ablate_hook_factory
from dl_quick_train.analysis.data import build_eval_batch
from minimal_sae_train import SEP


def test_build_sep_mask():
    b = build_eval_batch(4, offset=0)
    mask = build_sep_mask(b.tokens)
    for i in range(4):
        for j in range(mask.shape[1]):
            want = int(b.tokens[i, j].item()) == SEP
            assert bool(mask[i, j].item()) == want


def test_noop_hook_equivalence():
    """A hook that replaces activations at an empty mask should leave the forward
    pass unchanged."""
    pytest.importorskip("transformer_lens")
    from minimal_sae_train import build_model, load_weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("needs CUDA for reasonable speed")
    model = build_model().to(device)
    load_weights(model, device)
    b = build_eval_batch(8, offset=500_000).to(device)
    clean = model(b.tokens)

    # Hook with empty mask at resid_post
    empty_mask = torch.zeros_like(b.tokens, dtype=torch.bool)
    hook = _mean_ablate_hook_factory(
        torch.zeros(model.cfg.d_model), empty_mask,
    )
    patched = model.run_with_hooks(
        b.tokens,
        fwd_hooks=[("blocks.0.hook_resid_post", hook)],
    )
    assert torch.allclose(clean.float(), patched.float(), atol=1e-4)
