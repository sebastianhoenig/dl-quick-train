"""Mean-ablation hooks with a fixed reference distribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer

from minimal_sae_train import SEP

from .data import EvalBatch


def build_sep_mask(tokens: torch.Tensor) -> torch.Tensor:
    return tokens == SEP


@torch.no_grad()
def compute_reference_means(
    model: HookedTransformer,
    batch: EvalBatch,
    device: str,
    chunk: int = 256,
) -> Dict[str, torch.Tensor]:
    """Compute per-site means over all SEP positions in ``batch``.

    Returns dict with:
      - "hook_z" : [n_heads, d_head]
      - "hook_resid_post" : [d_model]
    """
    model.eval()
    batch = batch.to(device)
    names = ["blocks.0.attn.hook_z", "blocks.0.hook_resid_post"]

    total_z = None
    total_resid = None
    total_count = 0
    for start in range(0, batch.tokens.shape[0], chunk):
        tok = batch.tokens[start : start + chunk]
        _, cache = model.run_with_cache(tok, names_filter=names)
        z = cache["blocks.0.attn.hook_z"]           # [b, S, H, dh]
        resid = cache["blocks.0.hook_resid_post"]   # [b, S, d_model]
        mask = build_sep_mask(tok).to(device)       # [b, S]
        mask_flat = mask.reshape(-1)
        z_flat = z.reshape(-1, z.shape[-2], z.shape[-1])[mask_flat]
        resid_flat = resid.reshape(-1, resid.shape[-1])[mask_flat]
        if total_z is None:
            total_z = z_flat.sum(dim=0)
            total_resid = resid_flat.sum(dim=0)
        else:
            total_z += z_flat.sum(dim=0)
            total_resid += resid_flat.sum(dim=0)
        total_count += int(mask_flat.sum().item())

    return {
        "hook_z": (total_z / total_count).cpu(),
        "hook_resid_post": (total_resid / total_count).cpu(),
        "n_positions": torch.tensor(total_count),
    }


def _mean_ablate_hook_factory(
    mean_vec: torch.Tensor,
    mask: torch.Tensor,
    head: Optional[int] = None,
):
    """Return a TransformerLens hook that overwrites activations at positions
    where ``mask`` is True with ``mean_vec``.
    """
    def hook(act, hook=None):
        act = act.clone()
        dev = act.device
        mv = mean_vec.to(dev, act.dtype)
        m = mask.to(dev)
        if act.ndim == 4:  # hook_z: [B, S, H, dh]
            if head is None:
                act[m] = mv.view(*([1] * (act.ndim - mv.ndim)), *mv.shape).to(act.dtype)
            else:
                # mean over chosen head only
                act[..., head, :][m] = mv.to(act.dtype)
        elif act.ndim == 3:  # resid: [B, S, d_model]
            act[m] = mv.to(act.dtype)
        else:
            raise ValueError(f"unsupported activation rank {act.ndim}")
        return act
    return hook


@dataclass
class AblationMetrics:
    accuracy: float
    logit_diff: float
    routing_correct: float
    n: int

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "logit_diff": self.logit_diff,
            "routing_correct": self.routing_correct,
            "n": self.n,
        }


def _in_context_entities(tokens: torch.Tensor) -> List[torch.Tensor]:
    """For each sequence, return the list of entity ids in (0..E-1) that appear in-context
    (i.e., tokens < 100). Used for 'routing correctness': did argmax over in-context
    entities pick the target label?"""
    out = []
    for row in tokens:
        ents = row[row < 100]
        out.append(ents.unique())
    return out


@torch.no_grad()
def evaluate(model: HookedTransformer, batch: EvalBatch, logits: torch.Tensor) -> AblationMetrics:
    B = batch.tokens.shape[0]
    rows = torch.arange(B, device=logits.device)
    q_logits = logits[rows, batch.q_position.to(logits.device)]  # [B, V]
    preds = q_logits.argmax(dim=-1)
    target = batch.labels.to(logits.device)
    accuracy = (preds == target).float().mean().item()

    # logit_diff: logit(target) - mean over other entities in context
    ic_ents = _in_context_entities(batch.tokens.cpu())
    ld = []
    routing = []
    for b in range(B):
        ents = ic_ents[b].to(logits.device)
        t = int(target[b].item())
        logit_t = q_logits[b, t]
        others = ents[ents != t]
        if others.numel() == 0:
            ld.append(0.0)
        else:
            ld.append((logit_t - q_logits[b, others].mean()).item())
        # routing: restrict argmax to in-context entities
        restricted = q_logits[b, ents]
        routing.append(int(ents[restricted.argmax()].item()) == t)

    return AblationMetrics(
        accuracy=accuracy,
        logit_diff=float(sum(ld) / max(len(ld), 1)),
        routing_correct=float(sum(routing) / max(len(routing), 1)),
        n=B,
    )


@torch.no_grad()
def run_condition(
    model: HookedTransformer,
    batch: EvalBatch,
    means: Dict[str, torch.Tensor],
    device: str,
    ablate_head_z: Optional[List[int]] = None,
    ablate_resid: bool = False,
    sep_subset: str = "all",  # "all" | "target_only" | "non_target"
) -> AblationMetrics:
    """Run one ablation condition and return metrics.

    ``ablate_head_z``: list of L0 head indices to mean-ablate at SEP.
    ``ablate_resid``: also mean-ablate L0 hook_resid_post at SEP (strong upper bound).
    ``sep_subset``:
      "all"         — ablate at every SEP
      "target_only" — ablate at the target-fact SEP only
      "non_target"  — ablate at SEPs other than the target-fact SEP
    """
    batch = batch.to(device)
    tokens = batch.tokens
    B, S = tokens.shape
    all_sep_mask = build_sep_mask(tokens)  # [B, S]
    target_mask = torch.zeros_like(all_sep_mask)
    rows = torch.arange(B, device=device)
    target_mask[rows, batch.target_fact_sep_position] = True
    if sep_subset == "all":
        mask = all_sep_mask
    elif sep_subset == "target_only":
        mask = target_mask
    elif sep_subset == "non_target":
        mask = all_sep_mask & ~target_mask
    else:
        raise ValueError(sep_subset)

    fwd_hooks = []
    if ablate_head_z:
        z_mean = means["hook_z"]  # [H, dh]
        for h in ablate_head_z:
            fwd_hooks.append((
                "blocks.0.attn.hook_z",
                _mean_ablate_hook_factory(z_mean[h], mask, head=h),
            ))
    if ablate_resid:
        fwd_hooks.append((
            "blocks.0.hook_resid_post",
            _mean_ablate_hook_factory(means["hook_resid_post"], mask),
        ))

    if fwd_hooks:
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    else:
        logits = model(tokens)
    return evaluate(model, batch, logits)


__all__ = [
    "AblationMetrics",
    "build_sep_mask",
    "compute_reference_means",
    "evaluate",
    "run_condition",
]
