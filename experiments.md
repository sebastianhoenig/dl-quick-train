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
| P1 | Gemma-2B in-the-wild SAE test (width_16k) | `scripts/gemma/*` | ✅ done (L8H1 router; gap +0.07) | `gemma/summary.json` |
| P1b | Gemma wider-SAE replication (width_65k) | `scripts/gemma/run_gemma_sae.py` | ✅ done (**gap collapses to +0.002**) | `gemma/summary_w65k.json` |
| P2 | Multi-seed LM training (seeds 0–4, 200k steps) | `scripts/seeds/*` | ✅ done (5/5 circuits emerged) | `seeds/table.md` |
| P3 | Position sweep (Q vs SEP) on toy SAE eval | `scripts/run_sae_eval.py --position {sep,q}` | ✅ done (Eq at Q, E2q at SEP) | `sae_eval_{q,sep}/results.{json,md}` |
| P4 | Per-seed L0 payload SAE sweep | `minimal_sae_train.py` + `run_sae_eval.py` (both take `--model-ckpt`) | ❌ not run | — |
| P5 | Causal SAE-recovery test (feature-ablation through L1) | `scripts/run_sae_causal.py` | ⏳ script written, smoke-tested; not yet run at full N | `sae_causal/results.json` |

Plus: `tests/analysis/` (9 tests) pass.

---

## Where the evidence stands vs. the paper's thesis

P0.1–P0.3 support the staged-retrieval circuit claim cleanly:
- L0H0 = address head, L0H1 = payload head (|patching score| > 25).
- Either head ablated at target-SEP → accuracy 1.00 → 0.016 (≈ chance).
- L1 QK eigen-energy concentrates 0.93 on L0 OV's address basis vs. 0.12
  on a random subspace.

**P0.4 inverts the thesis on the toy model.** Top-k SAEs on the
composed `resid_post` site *beat* raw activations for predicting the
tail entity E2q by 31–45 F1 points (raw 0.38 → SAE 0.69–0.83). The
SAE's discrete feature basis makes E2q linearly decodable where the
raw residual does not. On isolated H1 at SEP, raw F1 for E2q is
already 0.988 and SAE matches within 0.013 — so the previously
reported "H1 k=16 loses 5 F1 points" signal does not exist in the
current artifacts. Net: no thesis-consistent signal on the toy model.

**P1 reversed by P1b.** The Gemma width_16k run showed a +0.07 F1
gap; the width_65k replication collapses it to +0.002 at comparable
reconstruction error. The original P1 headline was a narrow-SAE
capacity artifact, not a dark-matter signature. The paper's
external-validity claim no longer holds on the evidence collected.

**P2 confirms circuit reproducibility.** 5/5 seeds at 200k steps reach
≥ 99.9 % accuracy with a clean two-head address/payload factorisation.
Head *indices* swap across seeds (3 seeds: address=H1; 2: address=H0)
but topology is invariant. Address-swap magnitudes span 20× (|5| to
|104|) — worth explaining in the paper but does not block the
emergence claim.

**P3 clarifies head roles across positions.** Raw-F1 decoding shifts
sharply with extraction position: Eq lives at Q (composed raw F1 =
0.89), E2q lives at SEP (composed raw F1 = 0.38). H1 at Q holds the
relation (Tq F1 = 1.00), at SEP holds the tail entity (E2q F1 =
0.99). Patching labels from P0 are time-resolved, not position-
invariant.

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

**Status of the strong thesis after P0.4 / P1 / P1b / P3:** no
toy-model signal (composed SAE *beats* raw on E2q), no Gemma signal
after widening SAE (P1b gap +0.002). The two remaining live paths are
(a) checking whether the P0.4 result is a per-seed artifact, and
(b) trying a stronger recovery test than a static linear probe.

### 1 — P4 per-seed L0 payload SAE sweep

Checks whether the (now-absent) H1 SAE gap and the composed SAE-beats-raw
inversion reproduce across seeds trained from scratch. Per-seed payload
heads (from `seeds/table.md`): s0→H1, s1→H0, s2→H0, s3→H0, s4→H1.

`minimal_sae_train.py` appends `b0_hookz_h{head}_{pos}_sweep_...` under
`--save-dir`, so `--save-dir sae_ckpts/seed_<s>` gives a path whose
basename matches `SITE_MAP` in `run_sae_eval.py`. Both scripts accept
`--model-ckpt`.

```bash
# ----- TRAIN (expensive: ~2h × 5 on one GPU) -----
PYTHONPATH=. python minimal_sae_train.py --submodule blocks.0.attn.hook_z --head-index 1 --position-selector sep --model-ckpt experiments/results/seeds/seed_0/model.pt --save-dir sae_ckpts/seed_0 --steps 200000
PYTHONPATH=. python minimal_sae_train.py --submodule blocks.0.attn.hook_z --head-index 0 --position-selector sep --model-ckpt experiments/results/seeds/seed_1/model.pt --save-dir sae_ckpts/seed_1 --steps 200000
PYTHONPATH=. python minimal_sae_train.py --submodule blocks.0.attn.hook_z --head-index 0 --position-selector sep --model-ckpt experiments/results/seeds/seed_2/model.pt --save-dir sae_ckpts/seed_2 --steps 200000
PYTHONPATH=. python minimal_sae_train.py --submodule blocks.0.attn.hook_z --head-index 0 --position-selector sep --model-ckpt experiments/results/seeds/seed_3/model.pt --save-dir sae_ckpts/seed_3 --steps 200000
PYTHONPATH=. python minimal_sae_train.py --submodule blocks.0.attn.hook_z --head-index 1 --position-selector sep --model-ckpt experiments/results/seeds/seed_4/model.pt --save-dir sae_ckpts/seed_4 --steps 200000

# ----- EVAL (each uses the seed's LM so activations match) -----
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096 --ckpt-root sae_ckpts/seed_0 --model-ckpt experiments/results/seeds/seed_0/model.pt --out experiments/results/sae_eval_seed_0
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096 --ckpt-root sae_ckpts/seed_1 --model-ckpt experiments/results/seeds/seed_1/model.pt --out experiments/results/sae_eval_seed_1
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096 --ckpt-root sae_ckpts/seed_2 --model-ckpt experiments/results/seeds/seed_2/model.pt --out experiments/results/sae_eval_seed_2
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096 --ckpt-root sae_ckpts/seed_3 --model-ckpt experiments/results/seeds/seed_3/model.pt --out experiments/results/sae_eval_seed_3
PYTHONPATH=. python scripts/run_sae_eval.py --n 4096 --ckpt-root sae_ckpts/seed_4 --model-ckpt experiments/results/seeds/seed_4/model.pt --out experiments/results/sae_eval_seed_4
```

Acceptance to report: per-seed {composed E2q gap, isolated-payload E2q gap}.
If the "SAE beats raw by 30+ points on composed E2q" pattern reproduces on
5/5 seeds, the inverted result is the paper's headline (and needs a
different framing from the original thesis).

### 2 — P5 causal SAE-recovery test (motivated by P0.4/P1b nulls)

A static linear probe on SAE features asks "is the info *present*,"
not "does the SAE factorise it into *usable* directions." Stronger
test: per feature at the target SEP, ablate it, measure the change in
Layer-1 retrieval-head attention from Q to target-SEP.

- A well-factored SAE → a *sparse* set of features account for most
  of the attention drop.
- A superposed SAE → effect distributed densely across features.

`scripts/run_sae_causal.py` implements this. Smoke-tested on N=64 with
4 probed features: clean L1 attn-to-target = 0.98, full SAE recon
preserves 0.98, top-active feature ablations drop it by up to 0.03,
unused features drop by 0.00. Trainer-id ↔ (d, k) map: 0=(1024,8),
1=(1024,16), 2=(2048,8), 3=(2048,16), 4=(4096,8), 5=(4096,16).

Full run on the strongest composed SAE (d=4096, k=8, trainer_4):

```bash
PYTHONPATH=. python scripts/run_sae_causal.py --n 2048 --dict-size 4096 --k 8 --top-features 64 --sae-ckpt sae_ckpts/b0_residpost_sep_sweep_d1024-4096_k8-16/trainer_4/checkpoints/ae_1000000.pt --out experiments/results/sae_causal
```

Key numbers to report: `top_feature_shift_mean` vs
`random_control_shift_mean`, and `n_top_features_above_random_max`
(how sparse the causal set is). Worth also running on a k=16 SAE
(trainer_5) to test whether wider-k distributes the effect more
densely, which would be the first positive dark-matter signal.

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
