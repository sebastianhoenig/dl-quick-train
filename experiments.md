# Experimental plan — current state & what's left to run

Companion to `experiments/RESULTS.md` (which records outcomes). This doc is
the forward-looking plan: what's done, what's pending, and exactly which
commands to run next.

Model under test: `sebastianhoenig/2L2H_Final` (2L2H attn-only,
`d_model=256`, `d_head=128`), loaded via `build_model() + load_weights()`
in `minimal_sae_train.py`. Task: relational recall over synthetic
`(e, t, e2)` facts. Extraction locus throughout: the **target-fact SEP**
position.

---

## Status summary

| # | Experiment | Script | Status | Artifact |
|---|---|---|---|---|
| P0.0 | Head role identification | `scripts/run_verification.py` | ✅ done | `verification/head_roles.json` |
| P0.1 | Combinatorial geometry | `scripts/run_geometry.py` | ✅ done | `geometry/*.npy` + stats |
| P0.2 | QK/OV weight analysis | `scripts/run_qk_ov.py` | ✅ done | `qk_ov/*.npz` |
| P0.3 | Mean-ablation necessity | `scripts/build_mean_cache.py` + `scripts/run_ablation.py` | ✅ done | `ablation/*.json` |
| P0.4 | SAE recovery (composed vs isolated) | `scripts/run_sae_eval.py` | ✅ done (counter-evidence; see below) | `sae_eval/results.{json,md}` |
| P0.4b | Train L0H1 (payload) SAE sweep + eval | `minimal_sae_train.py` + `run_sae_eval.py` | ✅ done (30k steps; see RESULTS P0.4) | `sae_ckpts/b0_hookz_h1_sep_sweep_d1024-4096_k8-16/` |
| P1 | Gemma-2B in-the-wild SAE test | `scripts/gemma/*` | ❌ not run (needs `sae_lens` + GPU) | `gemma/summary.json` |
| P2 | Multi-seed LM training | `scripts/seeds/*` | ❌ not run (needs ~2–4h × 5 on GPU) | `seeds/table.md` |

Plus: `tests/analysis/` (9 tests) pass.

---

## Where the evidence stands vs. the paper's thesis

P0.1–P0.3 support the staged-retrieval circuit claim cleanly:
- L0H0 = address head, L0H1 = payload head (|patching score| > 25).
- Either head ablated at target-SEP → accuracy 1.00 → 0.016 (≈ chance).
- L1 QK eigen-energy concentrates 0.93 on L0 OV's address basis vs. 0.12
  on a random subspace.

**P0.4 is counter-evidence to the strong form of the thesis.** Top-k SAEs
(d ∈ {1024,2048,4096}, k ∈ {8,16}) achieve FVU ≈ 0 on *both* the composed
`resid_post` site and the isolated `hook_z` site. Raw-vs-SAE probe F1 gaps
are within ±0.04 everywhere — often *negative* on the composed site. The
"Top-k SAEs fail specifically on the composed E1+T superposition on this
toy model" claim does not reproduce.

Two confounds that must be resolved before publishing the dark-matter
narrative:

1. **Head labels mis-named.** Raw-F1 at SEP: H0 → Tq=0.929, E2q=0.000;
   H1 → Tq=0.063, E2q=0.205. Neither head encodes Eq. "H0 = address"
   really means "writes the *relation*"; "H1 = payload" means "writes
   the *tail entity*". Patching results still hold; only the names
   need fixing.
2. **One thesis-consistent signal (H1, k=16).** On L0H1 payload,
   k=16 SAEs lose ~5 F1 points of E2q vs raw at every dict width
   (0.205 → ~0.15); k=8 is fine. Confounded by the H1 sweep only
   running 30k steps (H0: 1.25M, resid_post: 1M).

---

## What to run next

### 1 — Train the missing L0H1 (payload) SAE sweep
Required to close P0.4. ~same wall-clock as the H0 sweep already shipped.

```bash
PYTHONPATH=. python minimal_sae_train.py \
    --submodule blocks.0.attn.hook_z \
    --head-index 1 \
    --position-selector sep
# then re-evaluate:
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096
```

Expected: if the thesis is correct in any form, H1 (payload) SAEs should
recover Tq cleanly; composed SAEs should show the F1 gap that's missing
today. If composed still looks fine with both heads trained, the paper
needs a different framing (causal intervention rather than static probe;
see §3 below).

### 2 — Re-examine head roles at `hook_z` SEP
The verification script patches `hook_z` to score each head's causal
effect on the downstream E2q logit. The F1 pattern at SEP disagrees with
the patching label. Candidate resolutions:

- Patching may identify the head that *moves information through* (causal)
  while the static F1 at SEP reflects what's *written there*. Not the
  same thing.
- Re-run the probe at the Q token rather than the SEP, and compare.

Small script, no new infra needed — extend `scripts/run_sae_eval.py` to
take `--position {sep, q}`.

### 3 — P1 Gemma-2B validation (Priority 1 from the old roadmap)

The toy-model null result on P0.4 makes the Gemma arm load-bearing for
the paper's external-validity claim. Scripts are ready in `scripts/gemma/`;
blocked on environment only.

```bash
pip install sae_lens
PYTHONPATH=. python scripts/gemma/run_gemma_patching.py --n 200
PYTHONPATH=. python scripts/gemma/run_gemma_sae.py       --n 500
```

Environment note: the current CUDA driver (12080) is too old for the
torchvision that was pulled in during P0.4. Either (a) pin torchvision
back to 0.19.1, or (b) update the CUDA driver, before running on GPU.

Key numbers to report: `mse_normalized`, `f1_raw − f1_sae`, identified
`(layer, head)` router. Gotcha: if too many prompts fail the single-token
filter under Gemma's tokenizer, curate `DEFAULT_TAILS` in
`scripts/gemma/gemma_prompts.py`.

### 4 — P2 multi-seed sweep (fallback if P1 can't run)

```bash
PYTHONPATH=. python scripts/seeds/run_seed_sweep.py --seeds 0 1 2 3 4 --steps 200000
```

Accept: ≥ 3/5 seeds reach ≥ 95 % clean acc and show clean L0
address/payload factorization. Compute-constrained fallback: 3 seeds ×
100k steps.

### 5 — Causal SAE-recovery test (new, motivated by P0.4 null)

A static linear probe on SAE features is the weakest possible recovery
test — it asks "is the info *present*," not "does the SAE factorise it
into *usable* directions." A stronger test the paper could adopt:

- Identify the address direction written by L0H0 into resid_post.
- Intervene on SAE features (ablate / activate) and measure whether the
  L1 retrieval head's attention pattern shifts as predicted.
- A well-factored SAE should give a *sparse* set of features whose
  intervention moves attention; a superposed SAE should require dense
  combinations.

This is a new script (`scripts/run_sae_causal.py`, TBD). Worth drafting
before the H1 SAE sweep finishes so both tests can land together.

---

## Reproducing P0 from scratch (< 2 minutes on GPU)

```bash
PYTHONPATH=. python scripts/run_verification.py  --n 2048
PYTHONPATH=. python scripts/run_geometry.py      --n 8192
PYTHONPATH=. python scripts/run_qk_ov.py
PYTHONPATH=. python scripts/build_mean_cache.py  --n 10000
PYTHONPATH=. python scripts/run_ablation.py      --n 2048
PYTHONPATH=. python scripts/run_sae_eval.py      --n 4096
PYTHONPATH=. python -m pytest tests/analysis -q
```

All numbers currently in `experiments/RESULTS.md` were produced with
smaller `--n` (512/2048) for speed; larger N has so far moved point
estimates by < 1 %.
