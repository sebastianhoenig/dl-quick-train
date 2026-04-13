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

**Composed state** (`blocks.0.hook_resid_post`, d=256), step 1,000,000:

| dict | k | FVU | L0 | gap Eq | gap Tq | gap E2q |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.003 | 7.9 | −0.040 | 0.000 | −0.030 |
| 1024 | 16 | 0.000 | 16.1 | −0.016 | +0.016 | −0.014 |
| 2048 | 8 | 0.004 | 8.0 | −0.011 | 0.000 | −0.029 |
| 2048 | 16 | 0.000 | 16.0 | −0.013 | +0.011 | −0.031 |
| 4096 | 8 | 0.006 | 8.0 | +0.034 | 0.000 | −0.017 |
| 4096 | 16 | 0.000 | 16.0 | −0.040 | +0.011 | −0.014 |

**Isolated state** (`blocks.0.attn.hook_z`, head 0, d=128), step 1,250,000:

| dict | k | FVU | L0 | gap Eq | gap Tq | gap E2q |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 8 | 0.001 | 7.9 | 0.000 | −0.040 | 0.000 |
| 1024 | 16 | 0.000 | 15.8 | +0.011 | +0.014 | 0.000 |
| 2048 | 8 | 0.001 | 7.9 | 0.000 | −0.013 | 0.000 |
| 2048 | 16 | 0.000 | 15.9 | +0.020 | +0.015 | 0.000 |
| 4096 | 8 | 0.001 | 7.9 | +0.007 | −0.071 | 0.000 |
| 4096 | 16 | 0.000 | 15.9 | +0.011 | −0.034 | 0.000 |

**Reading — counter-evidence to the strong form of the thesis on this toy
model.** Top-k SAEs achieve near-perfect reconstruction on *both* sites
(FVU ≈ 0), and probe-F1 gaps sit within ±0.04 in every cell — often
*negative* on the composed site (SAE features predict Eq/E2q marginally
better than raw). The "Top-k SAEs fail specifically on the composed E1+T
superposition" claim does not reproduce here.

Caveats before drawing a strong conclusion:
1. **L0H1 SAE not trained.** Only `hook_z` head 0 was swept; the payload
   head has no SAE, so we can only compare composed vs. address-isolated.
   To close this arm: `PYTHONPATH=. python minimal_sae_train.py --submodule
   blocks.0.attn.hook_z --head-index 1 --position-selector sep`.
2. **Surprising isolated-site pattern.** F1(raw Eq)=0.034 but F1(raw Tq)=
   0.929 on H0's `hook_z`. The "address head encodes Eq" assumption from
   the roadmap looks inverted — H0's SEP `hook_z` linearly decodes the
   *relation* almost perfectly and the *query entity* at chance. Worth
   re-examining head labelling before publishing.

The honest statement for the paper: on this 2L2H attn-only setup, Top-k
SAEs of modest width (d=1024–4096, k=8–16) recover the composed retrieval
state as well as they recover the isolated components. A dark-matter
story requires either (a) a larger / more realistic model (P1 Gemma), or
(b) a probing target where raw-vs-SAE diverges — e.g. a causal
intervention through the Layer-1 retrieval head rather than a static
linear probe.

---

## P1 — Gemma-2B (scripts ready, not yet executed)

`scripts/gemma/` contains:
- `gemma_prompts.py` — 500 natural-language relational prompts with
  single-token tail entities (validated against the live tokenizer).
- `run_gemma_patching.py` — layer × head denoising patching on Gemma-2B to
  locate the retrieval-router head for the task.
- `run_gemma_sae.py` — loads the matching Gemma Scope residual SAE and
  reports reconstruction MSE plus a macro-F1 gap between a logistic probe on
  SAE features vs. raw residual.

Not run here because Gemma-2B weights and `sae_lens` are not installed in
this environment.

---

## P2 — multi-seed LM training (scripts ready, not yet executed)

`scripts/seeds/train_lm.py` trains one 2L2H attention-only LM from scratch
(AdamW, lr=5e-4, wd=1e-2, cosine schedule, 1k warmup, 200k steps, batch 256).
`scripts/seeds/run_seed_sweep.py` trains seeds {0..4} and re-runs the
verification pre-step on each, emitting a markdown table.

Smoke-tested end-to-end for 200 steps: loss descends from 4.84 → 4.44,
checkpoint saves/loads correctly through the verification path.

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

### P1 Gemma (requires GPU + ~10 GB disk)

```bash
pip install sae_lens
PYTHONPATH=. python scripts/gemma/run_gemma_patching.py --n 200
PYTHONPATH=. python scripts/gemma/run_gemma_sae.py --n 500
```

Produces `experiments/results/gemma/{gemma_head_roles.json, summary.json}`.
Key numbers to report in the paper: `mse_normalized`, `f1_raw − f1_sae`,
and the identified `(layer, head)` router.

Possible gotchas: if too many prompts fail the single-token filter (tail
entities split under Gemma's tokenizer), curate `DEFAULT_TAILS` in
`scripts/gemma/gemma_prompts.py`. If the Gemma Scope variant
(`width_16k/average_l0_100`) is unavailable at the discovered layer, pass
`--sae-id layer_L/width_16k/average_l0_XX` with a valid bucket.

### P2 multi-seed sweep (5 × ~2–4 h on a single GPU)

```bash
PYTHONPATH=. python scripts/seeds/run_seed_sweep.py --seeds 0 1 2 3 4 --steps 200000
```

Produces `experiments/results/seeds/seed_{0..4}/model.pt` + `roles.json`
and a summary `experiments/results/seeds/table.md`.

If compute-constrained, the roadmap allows dropping to 3 seeds × 100k steps:

```bash
PYTHONPATH=. python scripts/seeds/run_seed_sweep.py --seeds 0 1 2 --steps 100000
```

Acceptance criterion (see `precious-munching-turing.md`): ≥ 3 / 5 seeds
reach ≥ 95 % clean accuracy and produce clean L0 address/payload
factorization. The `circuit_emerged` column in `table.md` encodes this.

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
