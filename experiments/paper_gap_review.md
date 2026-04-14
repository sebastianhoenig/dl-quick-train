# Paper vs. Experiments — Gap Review

Cross-checks each load-bearing claim in [paper.tex](../paper.tex) against the
artifacts in [experiments/results/](results/) and the narrative in
[experiments/RESULTS.md](RESULTS.md) / [experiments.md](../experiments.md).

Status legend: ✅ supported · ⚠️ partial / needs rewording · ❌ contradicted by current evidence.

---

## Claims the experiments support ✅

### C1. Staged retrieval circuit (two L0 heads factorize address/payload)

- Paper: [paper.tex:30](../paper.tex#L30), [paper.tex:84-89](../paper.tex#L84-L89).
- Evidence — activation-patching at target SEP ([verification/head_roles.json](results/verification/head_roles.json)):
  - L0H0 address-swap score **−27.21**, payload-swap **0.00**.
  - L0H1 address-swap **0.01**, payload-swap **+25.68**.
  - clean accuracy = **1.000** on 512 examples.
- Evidence — mean-ablation necessity ([experiments/RESULTS.md:98-105](RESULTS.md#L98-L105)):
  - ablate L0H0 at SEP → acc **0.016** (chance = 1/100).
  - ablate L0H1 at SEP → acc **0.016**.
  - restrict ablation to non-target SEPs → acc **1.000** (circuit is target-SEP local).

### C2. Layer 1 does same-head Q-K matching on the address basis

- Paper: [paper.tex:120-130](../paper.tex#L120-L130).
- Evidence ([experiments/RESULTS.md:73-83](RESULTS.md#L73-L83)):
  - L1H0 QK eigen-energy on address basis = **0.928** (vs. 0.119 random).
  - L1H1 QK eigen-energy on address basis = **0.930**.
  - Top QK eigenvalues (L1H0): 20.8, 19.5, 18.7, 18.4 — clearly low-rank.
- Caveat — **OV ≠ identity** on payload subspace: `W_V W_O` cosine with input column = **−0.08** ([experiments/RESULTS.md:86-88](RESULTS.md#L86-L88)). Paper's Fig. 1 caption ([paper.tex:79-81](../paper.tex#L79-L81)) implies pass-through; re-check wording.

### C3. Circuit reproducibility across seeds

- Paper implicit (no multi-seed claim yet) — but this is a strength worth adding.
- Evidence ([seeds/table.md](results/seeds/table.md) summarised at [experiments/RESULTS.md:240-252](RESULTS.md#L240-L252)): 5/5 seeds reach ≥ 99.9 % accuracy with a clean address/payload factorization. Head index swaps across seeds (s0, s4 → address=H0; s1, s2, s3 → address=H1).

---

## Claims the experiments contradict ❌

### D1. Dark-matter headline: "SAE feature recovery drops 0.778 → 0.195" on composed state

- Paper: abstract [paper.tex:30](../paper.tex#L30); intro [paper.tex:38](../paper.tex#L38); results [paper.tex:144-161](../paper.tex#L144-L161); conclusion [paper.tex:167-168](../paper.tex#L167-L168).
- **Current artifacts show the opposite direction.** At `blocks.0.hook_resid_post`, target-SEP, Top-k SAE vs. raw logistic probe on E2q ([experiments/RESULTS.md:127-134](RESULTS.md#L127-L134)):

  | dict | k | F1_raw E2q | F1_sae E2q | gap (raw − sae) |
  |---:|---:|---:|---:|---:|
  | 1024 | 8  | 0.383 | 0.807 | **−0.423** |
  | 1024 | 16 | 0.383 | 0.694 | **−0.310** |
  | 2048 | 8  | 0.383 | 0.815 | **−0.431** |
  | 2048 | 16 | 0.383 | 0.732 | **−0.349** |
  | 4096 | 8  | 0.383 | 0.832 | **−0.448** |
  | 4096 | 16 | 0.383 | 0.749 | **−0.366** |

- The "0.195" joint-F1 number in [paper.tex:159](../paper.tex#L159) is not reproducible from the current SAE checkpoints. The honest statement is in [experiments/RESULTS.md:158-166](RESULTS.md#L158-L166): *"Top-k SAE features give a much better logistic probe for E2q than the raw 256-dim residual."*

### D2. Head-role labels at SEP are mis-named

- Paper: [paper.tex:89](../paper.tex#L89) ("L0H0 acts as an 'address head', transporting E1 and T; L0H1 acts as a 'payload head', cleanly carrying E2") and probe table [paper.tex:99-106](../paper.tex#L99-L106).
- **Raw-F1 probes at SEP tell a different story** ([experiments/RESULTS.md:136-156](RESULTS.md#L136-L156)):

  | site | Eq F1_raw | Tq F1_raw | E2q F1_raw |
  |---|---:|---:|---:|
  | `hook_z` H0 | 0.433 | **1.000** | 0.002 |
  | `hook_z` H1 | 0.003 | 0.087 | **0.988** |

  H0 at SEP encodes the **relation**, not the joint `(E1, T)` address. H1 at SEP encodes the **tail entity**.
- Paper's probe table [paper.tex:101-106](../paper.tex#L101-L106) claims "E1 @ H0 = 100.0 %", "(E1, T) @ H0 = 100.0 %". No current artifact reproduces these — Eq F1 on H0 is 0.43, not 1.00.
- `experiments.md` explicitly flags this as a confound to resolve before publishing ([experiments.md:76-84](../experiments.md#L76-L84)).

### D3. External validity (Gemma) claim does not survive wider SAE

- Paper has no Gemma section yet, but the dark-matter framing ([paper.tex:162-168](../paper.tex#L162-L168)) is stated as a general SAE failure, and P1 was intended to provide external validity.
- Evidence ([experiments/RESULTS.md:209-215](RESULTS.md#L209-L215)):

  | SAE | mse_normalized | F1_raw | F1_sae | gap |
  |---|---:|---:|---:|---:|
  | `width_16k/average_l0_71` | 0.413 | 0.208 | 0.137 | **+0.071** |
  | `width_65k/average_l0_59` | 0.395 | 0.208 | 0.207 | **+0.002** |

  Reconstruction error is comparable; widening the SAE collapses the gap. This is a **width-under-provisioning** story, not a dark-matter signature ([experiments/RESULTS.md:216-229](RESULTS.md#L216-L229)).

### D4. Position-invariant head roles

- Paper presents head roles ([paper.tex:89](../paper.tex#L89)) without specifying that they are position-dependent.
- Evidence — P3 sweep ([experiments/RESULTS.md:262-277](RESULTS.md#L262-L277)):
  - H1 raw F1 at Q: Tq = **1.00**, E2q = **0.001**.
  - H1 raw F1 at SEP: Tq = 0.09, E2q = **0.99**.
  - Same head switches from relation-carrier (at Q) to tail-entity carrier (at SEP).
- Patching labels from P0 remain correct but need time-resolved framing in the paper.

---

## Gaps still open ⚠️

### G1. P0.4b H1 payload SAE sweep only trained 30 k steps

- Paper's claim depends on the isolated-head SAE baseline.
- Evidence ([experiments.md:80-84](../experiments.md#L80-L84)): H1 SAEs at 30k steps vs. H0 (1.25M) and resid_post (1M) — step-count confound for the one remaining thesis-consistent blip (H1, k=16 loses ~0.01 F1, not 5 F1 as originally claimed). Numbers at [experiments/RESULTS.md:149-156](RESULTS.md#L149-L156) confirm |gap| ≤ 0.013 everywhere.

### G2. P4 per-seed SAE sweep not run

- Needed to rule out that the P0.4 inversion is a single-seed artifact.
- Status: [experiments.md:29](../experiments.md#L29) — "❌ not run". Commands listed at [experiments.md:107-121](../experiments.md#L107-L121).

### G3. P5 causal SAE-recovery test not run at full N

- The only remaining path to a weaker dark-matter claim: linear probes may miss factorisation failures that a causal intervention would catch.
- Status: [experiments.md:30](../experiments.md#L30) — script written, smoke-tested on N=64 only. Full command at [experiments.md:147-149](../experiments.md#L147-L149).

### G4. Figures referenced in paper not verified against current artifacts

- `circuit.pdf` ([paper.tex:79](../paper.tex#L79)), `l1_qk_retrieval.png` ([paper.tex:127](../paper.tex#L127)), `causal_intervention.png` ([paper.tex:139](../paper.tex#L139)), `hookz_sae.png` / `residpost_sae.png` ([paper.tex:149-153](../paper.tex#L149-L153)) — source of numbers in figure captions is unclear. The "mean F1 = 0.958" for E2q ([paper.tex:155](../paper.tex#L155)) is in the ballpark of current artifacts (0.694–0.832), but "(E1, T) joint F1 = 0.195" has no matching artifact.

---

## Bottom line

Two viable paths:

1. **Rescue the dark-matter story.** Requires G2 (per-seed reproducibility of the inversion) **and** G3 (causal test distinguishes superposed vs. factored SAE features). If G3 shows a dense causal spread, re-frame dark-matter as a causal-factorisation failure, not a linear-probe failure — and re-label H0/H1 per D2.

2. **Reframe the paper around the inverted finding.** "Top-k SAEs *rescue* linear decodability of composed retrieval targets" is a publishable result, supported cleanly by D1's table and consistent with P1b's width-dependence story — but it is a different paper from the current draft.
