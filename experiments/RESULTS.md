# Experimental results

Results from the interpretability suite defined in `experiments.md`. All artifacts
under `experiments/results/` are reproduced from the scripts listed at the bottom.

Model: `sebastianhoenig/2L2H_Final` (2-layer, 2-head attention-only transformer,
`d_model=256`, `d_head=128`) loaded via `build_model()` + `load_weights()` in
`minimal_sae_train.py`. Task: relational recall over synthetic
`(entity, relation, entity)` facts.

All extraction happens at the **target-fact SEP position** — the SEP that
terminates the `(Eq, Tq, E2q)` fact in the prompt.

---

## P0 pre-step — head role identification

`scripts/run_verification.py` → `experiments/results/verification/head_roles.json`

Method: per-head activation patching on `blocks.0.attn.hook_z`. For each L0
head we patch donor activations at the target-fact SEP under two surgical
corruptions and record the change in the logit on `E2q`:

- **Address swap** (donor replaces `e` in target fact with a random entity):
  patching the address head should break retrieval → large |score|.
- **Payload swap** (donor replaces `e2`): patching the payload head should
  route to the wrong tail → large |score|.

| head | address-swap score | payload-swap score |
|---|---:|---:|
| L0H0 | **−27.21** | 0.00 |
| L0H1 | 0.01 | **+25.68** |

**Roles:** L0H0 = address head, L0H1 = payload head. Clean accuracy on 512
held-out examples: **1.000**.

Note: address-swap patching on H0 lands at **−27**, not +27. Retrieval after
the swap re-routes to a distractor fact whose tail entity is consistently
more extreme in the output-head direction than the original correct target;
the **magnitude** is the load-bearing signal, not the sign.

---

## P0.1 — geometry

`scripts/run_geometry.py` → `experiments/results/geometry/`

Pairwise cosine similarity (off-diagonal, N=2048 sequences; ≈2.1M pairs each):

| vector | dim | mean cos | p5 | p95 | std |
|---|---:|---:|---:|---:|---:|
| E1_z (L0H0 `hook_z` at target SEP) | 128 | +0.385 | +0.093 | +0.664 | 0.172 |
| T_z (L0H1 `hook_z` at target SEP) | 128 | +0.503 | +0.066 | +0.720 | 0.198 |
| E1+T (`resid_post` at target SEP) | 256 | +0.289 | −0.037 | +0.621 | 0.198 |

Also emits top-2 PCA coordinates per vector as `.npy` plus `meta.json` mapping
rows to `(Eq, Tq, E2q)` for plot coloring.

**Reading:** the roadmap predicted higher cosine for the composed state than
for the isolated components; empirically E1_z and T_z have individually higher
cosine (they live in narrow 128-dim head subspaces) while E1+T at resid_post
sits in a 256-dim space with more angular spread. The signal the paper needs
here is really the **structure** of PCA clusters (coloring by `Eq`/`Tq`), not
the scalar cosine — PCA artifacts are saved and ready for plotting.

---

## P0.2 — QK / OV weight analysis

`scripts/run_qk_ov.py` → `experiments/results/qk_ov/`

Projection energy of top-8 symmetric-part eigenvectors of Layer-1's
`W_Q W_Kᵀ` onto the **address basis** (top-32 right singular vectors of L0H0's
OV) and **payload basis** (top-32 of L0H1's OV):

| L1 head | on address basis | on payload basis | on random basis |
|---|---:|---:|---:|
| H0 | **0.928** | 0.057 | 0.119 |
| H1 | **0.930** | 0.055 | 0.118 |

Both Layer-1 heads concentrate ~93 % of their QK eigen-energy on the address
subspace, ≈8× more than on a random subspace of equal rank. Top QK eigenvalues
(L1 H0): 20.8, 19.5, 18.7, 18.4, 17.0, 12.4, 11.2, 10.5 — clearly low-rank.

**Payload-identity check**: `W_V W_O` applied to each payload-basis column
yields a cosine of −0.08 with the input column — Layer-1 does **not** copy the
payload as an identity; it applies a non-trivial transform. Qualifies the
roadmap's "OV acts as identity on the payload subspace" hypothesis.

---

## P0.3 — mean ablation

`scripts/build_mean_cache.py` caches per-site means over 2,000 held-out
examples at indices `[100 000, 102 000)`. `scripts/run_ablation.py` evaluates
on 512 further held-out examples at offset 400,000.

| condition | accuracy | routing_correct | logit_diff |
|---|---:|---:|---:|
| clean | 1.000 | 1.000 | +24.70 |
| mean-ablate L0H0 (address) at all SEP | **0.016** | 0.076 | −0.58 |
| mean-ablate L0H1 (payload) at all SEP | 0.016 | 0.100 | +0.20 |
| mean-ablate both at all SEP | 0.014 | 0.072 | −0.84 |
| mean-ablate L0H0 at non-target SEP only | **1.000** | 1.000 | +24.82 |
| mean-ablate resid_post at all SEP | 0.014 | 0.072 | −0.84 |

**Reading:** ablating either L0 head at SEP collapses accuracy from 1.000 to
~chance (0.016 ≈ 1/100). Restricting ablation to non-target SEPs has zero
effect — the circuit lives at the target fact's SEP specifically. Ablating
the full `resid_post` is an upper bound and matches "both heads" exactly,
confirming these two heads explain essentially all of the SEP-stored signal.

---

## P0.4 — SAE recovery on composed vs isolated state

`scripts/run_sae_eval.py` → `experiments/results/sae_eval/results.{json,md}`

Evaluates existing Top-k SAE checkpoints under `sae_ckpts/` on the paper's
core thesis. Per (dict_size, k) we report reconstruction FVU + mean L0, plus
the macro-F1 gap between a logistic probe on SAE features vs. raw
activations, predicting {Eq, Tq, E2q} at the target-fact SEP.

**Composed state** (`blocks.0.hook_resid_post`, d=256), step 1,000,000.
Negative gap means SAE > raw.

| dict | k | FVU | L0 | F1_raw Eq | F1_sae Eq | gap Eq | F1_raw Tq | F1_sae Tq | gap Tq | F1_raw E2q | F1_sae E2q | gap E2q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.003 | 8.0 | 0.769 | 0.915 | −0.146 | 1.000 | 0.996 | +0.004 | 0.383 | 0.807 | **−0.423** |
| 1024 | 16 | 0.000 | 16.0 | 0.769 | 0.848 | −0.079 | 1.000 | 0.998 | +0.002 | 0.383 | 0.694 | **−0.310** |
| 2048 | 8 | 0.003 | 8.0 | 0.769 | 0.822 | −0.053 | 1.000 | 0.998 | +0.002 | 0.383 | 0.815 | **−0.431** |
| 2048 | 16 | 0.000 | 16.0 | 0.769 | 0.814 | −0.045 | 1.000 | 1.000 | +0.000 | 0.383 | 0.732 | **−0.349** |
| 4096 | 8 | 0.006 | 8.0 | 0.769 | 0.741 | +0.028 | 1.000 | 0.996 | +0.004 | 0.383 | 0.832 | **−0.448** |
| 4096 | 16 | 0.000 | 16.0 | 0.769 | 0.815 | −0.046 | 1.000 | 1.000 | +0.000 | 0.383 | 0.749 | **−0.366** |

**Isolated address** (`blocks.0.attn.hook_z`, head 0, d=128), step 1,250,000:

| dict | k | FVU | L0 | F1_raw Eq | F1_sae Eq | gap Eq | F1_raw Tq | F1_sae Tq | gap Tq | F1_raw E2q | F1_sae E2q | gap E2q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.001 | 7.9 | 0.433 | 0.621 | −0.187 | 1.000 | 1.000 | 0.000 | 0.002 | 0.001 | +0.001 |
| 1024 | 16 | 0.000 | 15.8 | 0.433 | 0.400 | +0.033 | 1.000 | 1.000 | 0.000 | 0.002 | 0.002 | 0.000 |
| 2048 | 8 | 0.001 | 7.9 | 0.433 | 0.645 | −0.212 | 1.000 | 1.000 | 0.000 | 0.002 | 0.001 | +0.001 |
| 2048 | 16 | 0.000 | 15.9 | 0.433 | 0.368 | +0.066 | 1.000 | 1.000 | 0.000 | 0.002 | 0.001 | +0.001 |
| 4096 | 8 | 0.001 | 7.9 | 0.433 | 0.617 | −0.184 | 1.000 | 1.000 | 0.000 | 0.002 | 0.001 | +0.001 |
| 4096 | 16 | 0.000 | 15.9 | 0.433 | 0.371 | +0.062 | 1.000 | 1.000 | 0.000 | 0.002 | 0.001 | 0.000 |

**Isolated payload** (`blocks.0.attn.hook_z`, head 1, d=128), step 29,999:

| dict | k | FVU | L0 | F1_raw Eq | F1_sae Eq | gap Eq | F1_raw Tq | F1_sae Tq | gap Tq | F1_raw E2q | F1_sae E2q | gap E2q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.000 | 8.0 | 0.003 | 0.003 | 0.000 | 0.087 | 0.089 | −0.002 | 0.988 | 0.975 | +0.013 |
| 1024 | 16 | 0.000 | 15.9 | 0.003 | 0.003 | 0.000 | 0.087 | 0.098 | −0.010 | 0.988 | 0.985 | +0.003 |
| 2048 | 8 | 0.000 | 8.1 | 0.003 | 0.003 | 0.000 | 0.087 | 0.110 | −0.022 | 0.988 | 0.988 | 0.000 |
| 2048 | 16 | 0.000 | 16.1 | 0.003 | 0.003 | 0.000 | 0.087 | 0.093 | −0.006 | 0.988 | 0.988 | 0.000 |
| 4096 | 8 | 0.000 | 8.0 | 0.003 | 0.003 | +0.001 | 0.087 | 0.101 | −0.013 | 0.988 | 0.988 | 0.000 |
| 4096 | 16 | 0.000 | 15.8 | 0.003 | 0.002 | +0.001 | 0.087 | 0.097 | −0.010 | 0.988 | 0.976 | +0.012 |

**Reading — the composed site *inverts* the thesis.** At target-SEP,
Top-k SAE features give a **much better** logistic probe for E2q than
the raw 256-dim residual (0.38 → 0.69–0.83). Across all six (d, k)
combinations the gap is −0.31 to −0.45; k=8 is stronger than k=16.
The interpretation: raw resid_post packs E2q into a diffuse superposed
direction the linear probe cannot cleanly separate, while the SAE's
Top-k feature set makes it one-hot. This is the opposite of the paper's
thesis — the SAE *rescues* linear decodability of the retrieved
payload, rather than destroying it.

**Head roles at SEP, re-read from the corrected tables:**

- **H0 (address).** Raw Tq F1 = 1.000 and raw E2q F1 = 0.002.
  H0 writes the *relation* at SEP. Raw Eq F1 = 0.433 is well above
  chance (1/100) — H0 also retains a partial view of the query
  entity. SAE at k=8 *boosts* H0 Eq F1 by ~0.2 points; k=16 slightly
  hurts it.
- **H1 (payload).** Raw E2q F1 = 0.988 and raw Eq F1 = 0.003.
  H1 writes the *tail entity* at SEP with near-perfect linear
  decodability. SAE matches raw within ±0.01 — the toy-model thesis
  signature (SAE degrades isolated-head decodability) is essentially
  absent.

Patching labels hold (L0H0 routes, L0H1 carries payload) but the SEP
names are "relation head" and "tail-entity head", not "query-entity
encoder" and "payload mover".

**Honest statement for the paper.** On this 2L2H attn-only setup,
Top-k SAEs don't *fail* on the composed state — they **win**, by a
very large margin on E2q. The only site with a ≥ 0.05 positive gap
anywhere is none (previous "H1 k=16 loses 5 F1 points" claim came
from a stale snapshot; the current artifact shows gaps within ±0.013
on H1). A dark-matter story for this toy model does not survive a
static linear-probe test; a causal-intervention test is the only
remaining path (see `scripts/run_sae_causal.py`, TBD).

---

## P1 — Gemma-2B in-the-wild SAE test

`scripts/gemma/run_gemma_patching.py` → `experiments/results/gemma/gemma_head_roles.json`
`scripts/gemma/run_gemma_sae.py` → `experiments/results/gemma/summary.json`

**Router identification** (layer × head denoising patching, N=200 prompts,
clean logit 13.62, corrupt 12.47): the max positive denoising score is
**layer 8, head 1 = +0.98** (recovers ~85 % of the clean–corrupt gap). A
handful of deeper cells (L14H5, L16H2, L17H4) produce large *negative*
scores, consistent with those heads moving the retrieval forward rather than
being the router itself.

**SAE recovery on the identified router** (`gemma-scope-2b-pt-res`, L8,
N=500 prompts):

| SAE | mse_normalized | F1_raw | F1_sae | gap (raw − sae) |
|---|---:|---:|---:|---:|
| `width_16k/average_l0_71` | 0.413 | 0.208 | 0.137 | **+0.071** |
| `width_65k/average_l0_59` | 0.395 | 0.208 | 0.207 | **+0.002** |

**Reading — the thesis signal is bucket-dependent and collapses at
width_65k.** Reconstruction error is comparable across widths (mse ≈
0.40), but the probe-F1 gap evaporates when the SAE is wider: +0.071
at width_16k shrinks to +0.002 at width_65k. In other words, the
tail-entity information is present in the raw residual *and* in a
wider SAE's features; width_16k just doesn't have enough capacity to
keep it linearly decodable. That is a width-under-provisioning story,
not a fundamental dark-matter failure of Top-k SAEs.

Implication for the paper: the original P1 headline ("Gemma Scope
leaves a 7-point F1 gap") needs to be re-scoped as "narrow Gemma Scope
buckets leave a gap that wider buckets close". The external-validity
claim for the strong dark-matter thesis no longer holds on the
evidence collected here.

---

## P2 — multi-seed LM training

`scripts/seeds/run_seed_sweep.py` → `experiments/results/seeds/table.md`
(seeds {0..4}, 200k steps each, verification pre-step re-run per seed).

| seed | acc | address head | payload head | addr score | pay score | circuit |
|---:|---:|---|---|---:|---:|:---:|
| 0 | 1.000 | L0H0 | L0H1 | −103.54 | +22.99 | ✅ |
| 1 | 1.000 | L0H1 | L0H0 | −86.90  | +22.60 | ✅ |
| 2 | 0.999 | L0H1 | L0H0 | +7.19   | +16.58 | ✅ |
| 3 | 1.000 | L0H1 | L0H0 | +4.66   | +22.71 | ✅ |
| 4 | 1.000 | L0H0 | L0H1 | −49.84  | +22.07 | ✅ |

**Reading.** **5 / 5 seeds** reach ≥ 99.9 % clean accuracy and yield a
clean address/payload factorisation — well above the 3 / 5 acceptance
criterion. The head *index* is arbitrary across seeds (3 seeds map
address→H1, 2 map address→H0), but the two-head staged-retrieval
topology is reproducible. Seeds 2 and 3 have noticeably smaller
address-swap magnitudes (|7|, |5|) than seeds 0, 1, 4 (|104|, |87|,
|50|); worth flagging but does not change the binary emergence result.

---

## P3 — position sweep (Q vs target-SEP)

`scripts/run_sae_eval.py --position q` → `experiments/results/sae_eval_q/results.md`.
Same SAE ckpts, labels, and N=4096 as the SEP run above, but
activations extracted at the Q token instead of target-fact SEP.

Notable probe-F1 shifts, **raw activations** (macro-F1 on {Eq, Tq, E2q}):

| site | Eq SEP → Q | Tq SEP → Q | E2q SEP → Q |
|---|:---:|:---:|:---:|
| `resid_post` (composed) | 0.77 → **0.89** | 1.00 → 1.00 | 0.38 → **0.005** |
| `hook_z` H0 (address) | 0.43 → **0.52** | 1.00 → 1.00 | 0.00 → 0.00 |
| `hook_z` H1 (payload) | 0.003 → 0.11 | 0.09 → 1.00 | **0.99 → 0.001** |

**Reading.** Eq lives at Q, E2q lives at SEP — as expected for a
retrieval circuit: the query entity is read at the Q token, and the
retrieved tail is assembled at the SEP that terminates the target
fact. H1 at Q completely *lacks* E2q (0.001) and instead has perfect
Tq — so "H1 = payload" is a SEP-only role; at Q the same head is
passing the relation. Patching labels from P0 stand, but they're
time-resolved: H0 reads Tq throughout, H1 switches from relation
carrier (at Q) to tail-entity carrier (at SEP).

**SAE-side caveat.** H1 SAEs were trained on SEP-position `hook_z`
activations, so they do not fit the Q-token distribution:
reconstruction FVU blows up to 1.2 → 15.7 (vs ~0 at SEP). Q-position
numbers for H1 therefore reflect out-of-distribution SAE behaviour;
the SEP result is what matters for H1 claims. H0 and resid_post SAEs
were also SEP-trained but retain FVU < 0.02 at Q — consistent with
those sites having more position-invariant structure.

---

## Tests

`tests/analysis/` — 9 pytest tests (`pytest tests/analysis`):

- `test_data.py` — batch shape invariants, target-fact SEP correctness,
  byte-for-byte agreement with the original `produce_example_by_index`.
- `test_weights.py` — QK/OV shapes, descending real eigenvalues,
  projection-energy bounds, orthonormal random basis.
- `test_ablation.py` — SEP mask correctness, no-op hook equals clean forward.

---

## What's left to run

### Re-running P0 from scratch (takes < 2 minutes)

```bash
PYTHONPATH=. python scripts/run_verification.py --n 2048
PYTHONPATH=. python scripts/run_geometry.py --n 8192
PYTHONPATH=. python scripts/run_qk_ov.py
PYTHONPATH=. python scripts/build_mean_cache.py --n 10000
PYTHONPATH=. python scripts/run_ablation.py --n 2048
PYTHONPATH=. python -m pytest tests/analysis -q
```

The numbers above were produced with smaller `--n` (512/2048) for speed;
larger N tightens error bars but has so far moved point estimates by <1 %.
