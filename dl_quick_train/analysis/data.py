"""Evaluation batch construction.

Replays ``produce_example_by_index`` with full meta so every analysis script can
share the same SEP / Q / target-fact-SEP positions. The target-fact SEP is the
SEP token terminating the ``(Eq, Tq, E2q)`` fact in the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from minimal_sae_train import (
    E,
    IGNORE_INDEX,
    PAD,
    Q,
    SEP,
    T,
)

_MIN_FACTS, _MAX_FACTS, _SEED = 4, 8, 0


@dataclass
class EvalBatch:
    tokens: torch.Tensor            # [B, S] long
    labels: torch.Tensor            # [B] long, target entity id (E2q)
    lengths: torch.Tensor           # [B] long, actual sequence length
    q_position: torch.Tensor        # [B] long
    eq_position: torch.Tensor       # [B] long, position of Eq token (right before Q)
    tq_position: torch.Tensor       # [B] long, position of Tq token
    target_fact_sep_position: torch.Tensor  # [B] long, SEP ending target fact
    eq_token: torch.Tensor          # [B] long, entity id used in query
    tq_token: torch.Tensor          # [B] long, relation id used in query

    def to(self, device) -> "EvalBatch":
        return EvalBatch(
            tokens=self.tokens.to(device),
            labels=self.labels.to(device),
            lengths=self.lengths.to(device),
            q_position=self.q_position.to(device),
            eq_position=self.eq_position.to(device),
            tq_position=self.tq_position.to(device),
            target_fact_sep_position=self.target_fact_sep_position.to(device),
            eq_token=self.eq_token.to(device),
            tq_token=self.tq_token.to(device),
        )


def _produce_example_with_meta(idx: int, allow_self_loops: bool = False):
    """Reimplementation of ``produce_example_by_index`` that also returns the
    index (within ``facts``, after any distractor insertion) of the target fact.
    """
    rng = np.random.default_rng(np.random.SeedSequence([_SEED, idx]))
    k = int(rng.integers(_MIN_FACTS, _MAX_FACTS + 1))
    facts, seen = [], set()
    while len(facts) < k:
        e = int(rng.integers(0, E))
        t = int(rng.integers(0, T)) + E
        if (e, t) in seen:
            continue
        e2 = int(rng.integers(0, E))
        while (not allow_self_loops) and e2 == e:
            e2 = int(rng.integers(0, E))
        seen.add((e, t))
        facts.append((e, t, e2))
    q_idx = int(rng.integers(0, k))
    Eq, Tq, E2q = facts[q_idx]
    target_idx = q_idx
    if rng.random() < 0.75 and len(facts) < _MAX_FACTS:
        distractor_t = int(rng.integers(0, T)) + E
        while distractor_t == Tq:
            distractor_t = int(rng.integers(0, T)) + E
        distractor_e2 = int(rng.integers(0, E))
        while distractor_e2 == E2q:
            distractor_e2 = int(rng.integers(0, E))
        if (Eq, distractor_t) not in seen:
            ins = int(rng.integers(0, len(facts) + 1))
            facts.insert(ins, (Eq, distractor_t, distractor_e2))
            if ins <= target_idx:
                target_idx += 1
    seq: List[int] = []
    for e, t, e2 in facts:
        seq.extend([e, t, e2, SEP])
    seq.extend([Tq, Eq, Q])
    # Token layout per fact j: [e@4j, t@4j+1, e2@4j+2, SEP@4j+3]
    target_fact_sep = 4 * target_idx + 3
    return {
        "seq": seq,
        "label": E2q,
        "Eq": Eq,
        "Tq": Tq,
        "target_fact_sep": target_fact_sep,
        "n_facts": len(facts),
    }


def build_eval_batch(n: int, offset: int = 100_000) -> EvalBatch:
    """Build a padded batch of ``n`` examples starting at ``offset``.

    The default offset sits outside the training stream (20k–99,999) and outside
    the SAE ablation mean-cache range (100k–109,999) when ``n <= 10000``; if
    you need non-overlapping eval samples, pass ``offset=110_000`` or similar.
    """
    examples = [_produce_example_with_meta(offset + i) for i in range(n)]
    max_len = max(len(ex["seq"]) for ex in examples)
    tokens = torch.full((n, max_len), PAD, dtype=torch.long)
    labels = torch.empty(n, dtype=torch.long)
    lengths = torch.empty(n, dtype=torch.long)
    q_pos = torch.empty(n, dtype=torch.long)
    eq_pos = torch.empty(n, dtype=torch.long)
    tq_pos = torch.empty(n, dtype=torch.long)
    tgt_sep = torch.empty(n, dtype=torch.long)
    eq_tok = torch.empty(n, dtype=torch.long)
    tq_tok = torch.empty(n, dtype=torch.long)
    for i, ex in enumerate(examples):
        L = len(ex["seq"])
        tokens[i, :L] = torch.tensor(ex["seq"], dtype=torch.long)
        labels[i] = ex["label"]
        lengths[i] = L
        q_pos[i] = L - 1                # Q is the final token
        eq_pos[i] = L - 2               # Eq is right before Q
        tq_pos[i] = L - 3               # Tq is right before Eq
        tgt_sep[i] = ex["target_fact_sep"]
        eq_tok[i] = ex["Eq"]
        tq_tok[i] = ex["Tq"]
    return EvalBatch(
        tokens=tokens,
        labels=labels,
        lengths=lengths,
        q_position=q_pos,
        eq_position=eq_pos,
        tq_position=tq_pos,
        target_fact_sep_position=tgt_sep,
        eq_token=eq_tok,
        tq_token=tq_tok,
    )


def gather_at_positions(act: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather activations at per-row positions.

    act: [B, S, ...], positions: [B] long. Returns [B, ...].
    """
    assert act.shape[0] == positions.shape[0]
    idx = positions.to(act.device).view(-1, 1, *([1] * (act.ndim - 2))).expand(
        -1, 1, *act.shape[2:]
    )
    return act.gather(1, idx).squeeze(1)


def sep_positions_tensor(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask [B, S] of all SEP positions."""
    return tokens == SEP


def q_position_tensor(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask [B, S] with the single Q position per row."""
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for b in range(tokens.shape[0]):
        pos = (tokens[b] == Q).nonzero(as_tuple=False)
        assert pos.numel() >= 1
        mask[b, int(pos[0, 0].item())] = True
    return mask


__all__ = [
    "EvalBatch",
    "build_eval_batch",
    "gather_at_positions",
    "sep_positions_tensor",
    "q_position_tensor",
]
