"""Activation patching utilities for the 2L2H toy model.

The ``head_role_scan`` routine identifies which Layer-0 head carries the
"address" (entity-binding) signal and which carries the "payload" (tail-entity)
signal. It does so by running two donor forwards with surgical token-level
corruptions of the target fact:

- *Address-swap donor*: replaces the target fact's head entity ``e`` with a
  random different entity. L0H0 (if it is the address head) will encode the
  replacement into ``hook_z`` at the target-fact SEP; patching from donor into
  clean at that position should then break retrieval of ``E2q``.

- *Payload-swap donor*: replaces the target fact's tail entity ``e2`` with a
  random different entity. Analogous logic isolates the payload head.

For each (corruption type, candidate L0 head) we record the drop in the clean
logit at the ``E2q`` target. The head with the largest drop under
address-swap is the address head; same under payload-swap for payload.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformer_lens import HookedTransformer

from minimal_sae_train import E

from .data import EvalBatch, gather_at_positions


@dataclass
class HeadRoles:
    address_head: int
    payload_head: int
    address_scores: List[float]
    payload_scores: List[float]
    clean_accuracy: float
    n_examples: int

    def to_dict(self) -> dict:
        return {
            "layer": 0,
            "address_head": self.address_head,
            "payload_head": self.payload_head,
            "address_scores": self.address_scores,
            "payload_scores": self.payload_scores,
            "clean_accuracy": self.clean_accuracy,
            "n_examples": self.n_examples,
        }


def _random_other_entity(exclude: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """For each row, sample a random entity id in [0, E) different from ``exclude[row]``."""
    B = exclude.shape[0]
    out = torch.randint(0, E, (B,), generator=rng)
    clash = out == exclude
    while clash.any():
        out[clash] = torch.randint(0, E, (int(clash.sum()),), generator=rng)
        clash = out == exclude
    return out


def _logit_at(logits: torch.Tensor, q_pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gather ``logits[b, q_pos[b], target[b]]``. logits: [B, S, V]."""
    B = logits.shape[0]
    rows = torch.arange(B, device=logits.device)
    return logits[rows, q_pos.to(logits.device), target.to(logits.device)]


def _run_and_cache_z(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    name = "blocks.0.attn.hook_z"
    _, cache = model.run_with_cache(tokens, names_filter=[name])
    return cache[name]  # [B, S, n_heads, d_head]


def _patched_logits_at_sep(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    donor_z: torch.Tensor,
    target_sep: torch.Tensor,
    head: int,
) -> torch.Tensor:
    """Run clean forward; at ``blocks.0.attn.hook_z`` replace ``(b, target_sep[b], head, :)``
    with ``donor_z[b, target_sep[b], head, :]``."""
    B = clean_tokens.shape[0]
    rows = torch.arange(B, device=clean_tokens.device)
    tgt = target_sep.to(clean_tokens.device)

    def hook(z, hook=None):
        z = z.clone()
        z[rows, tgt, head, :] = donor_z[rows, tgt, head, :].to(z.device, z.dtype)
        return z

    return model.run_with_hooks(
        clean_tokens,
        fwd_hooks=[("blocks.0.attn.hook_z", hook)],
    )


def head_role_scan(
    model: HookedTransformer,
    batch: EvalBatch,
    device: str = "cuda",
    seed: int = 0,
) -> HeadRoles:
    """Run per-head activation patching to identify address vs payload roles.

    Returns scores for each Layer-0 head. address_score[h] is the mean drop in
    logit(E2q) at the Q position when patching donor L0 hook_z at (target-SEP,
    head=h) under an address-swap corruption; payload_score is analogous under
    a payload-swap corruption.
    """
    model.eval()
    batch = batch.to(device)
    tokens = batch.tokens
    B, S = tokens.shape
    tgt_sep = batch.target_fact_sep_position
    rows = torch.arange(B, device=device)

    # Positions of e and e2 in the target fact.
    e_pos = tgt_sep - 3
    e2_pos = tgt_sep - 1

    rng = torch.Generator()
    rng.manual_seed(seed)

    # Build donor token sets.
    eq_tok = batch.eq_token.cpu()
    e2_tok = batch.labels.cpu()
    e_alt = _random_other_entity(eq_tok, rng).to(device)
    e2_alt = _random_other_entity(e2_tok, rng).to(device)

    addr_tokens = tokens.clone()
    addr_tokens[rows, e_pos] = e_alt

    payload_tokens = tokens.clone()
    payload_tokens[rows, e2_pos] = e2_alt

    with torch.no_grad():
        clean_logits = model(tokens)
        baseline = _logit_at(clean_logits, batch.q_position, batch.labels)
        clean_pred = clean_logits[rows, batch.q_position].argmax(dim=-1)
        clean_acc = (clean_pred == batch.labels).float().mean().item()

        addr_z = _run_and_cache_z(model, addr_tokens)
        payload_z = _run_and_cache_z(model, payload_tokens)

        address_scores: List[float] = []
        payload_scores: List[float] = []
        for head in range(model.cfg.n_heads):
            logits_addr = _patched_logits_at_sep(
                model, tokens, addr_z, tgt_sep, head
            )
            logits_pay = _patched_logits_at_sep(
                model, tokens, payload_z, tgt_sep, head
            )
            addr_drop = (
                baseline - _logit_at(logits_addr, batch.q_position, batch.labels)
            ).mean().item()
            pay_drop = (
                baseline - _logit_at(logits_pay, batch.q_position, batch.labels)
            ).mean().item()
            address_scores.append(addr_drop)
            payload_scores.append(pay_drop)

    # Use |score|: a head is "the address head" if address-swap corruption at
    # that head moves the logit substantially, regardless of sign. In practice
    # the address head produces a large positive drop (retrieval broken) but a
    # mirror effect at the other head can yield a negative score of equal size
    # if the downstream circuit re-routes to a distractor.
    address_head = int(max(range(len(address_scores)), key=lambda h: abs(address_scores[h])))
    payload_head = int(
        max(
            (h for h in range(len(payload_scores)) if h != address_head),
            key=lambda h: abs(payload_scores[h]),
        )
    )
    return HeadRoles(
        address_head=address_head,
        payload_head=payload_head,
        address_scores=address_scores,
        payload_scores=payload_scores,
        clean_accuracy=clean_acc,
        n_examples=B,
    )


__all__ = ["HeadRoles", "head_role_scan"]
