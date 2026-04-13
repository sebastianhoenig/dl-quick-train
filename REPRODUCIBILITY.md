# Reproducibility notes (model, data, SAE training)

This document summarizes what is **defined in this repository** versus what is **only implied by external artifacts** (e.g. Hugging Face checkpoint filenames). Primary sources: `minimal_sae_train.py`, `dl_quick_train/pipeline.py`, and the upstream **`dictionary_learning`** implementation (see [§3.1](#31-batchtopktrainer-dictionary_learning-package)).

---

## 1. Model architecture

The toy language model is instantiated with **TransformerLens** `HookedTransformerConfig` in `minimal_sae_train.py`.

### Specifications

| Quantity | Value |
|----------|--------|
| Layers (`n_layers`) | 2 |
| Attention heads (`n_heads`) | 2 |
| Residual stream width (`d_model`) | 256 |
| Head dimension (`d_head`) | 128 |
| Context length (`n_ctx`) | 64 |
| Full vocabulary size (`d_vocab`) | 113 (see [Dataset](#2-dataset-prompt-construction)) |
| Output vocabulary (`d_vocab_out`) | 100 (entities only) |
| Positional embeddings | Rotary (`positional_embedding_type="rotary"`) |
| Normalization | LayerNorm (`normalization_type="LN"`) |

### Modifications and constraints

- **`attn_only=True`**: attention-only transformer (no MLP blocks in this configuration).
- **Narrow LM head**: `d_vocab_out=E` restricts the prediction logits to entity IDs `0 … E-1`, not relation or delimiter tokens.
- **Weights**: the model is **not** trained inside this repo. Weights are loaded from Hugging Face:

  - **Repo:** `sebastianhoenig/2L2H_Final`
  - **File:** `D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt`

  The filename suggests pretraining used learning rate **5×10⁻⁴** and weight decay **0.01**. **Optimizer type, schedule, and total LM training steps are not specified in this codebase.**

### Analysis-time masking (SAE pipeline)

In `dl_quick_train/pipeline.py`, activations used for dictionary learning can be filtered to:

- all positions where the token id is **SEP (110)**, or
- the single **Q (111)** position per sequence,

and for `blocks.0.attn.hook_z`, a **single attention head** can be selected. This is an experimental protocol choice, not a change to the base LM architecture.

---

## 2. Dataset (prompt construction)

Defined in `minimal_sae_train.py` with explicit token-id constants.

### Token inventory

| Symbol | Role | ID (formula) |
|--------|------|----------------|
| `E` | Number of entities | 100 (ids `0 … 99`) |
| `T` | Number of relation types | 10 (ids `100 … 109`) |
| `SEP` | Fact separator | `E + T` = **110** |
| `Q` | Question marker | `E + T + 1` = **111** |
| `PAD` | Padding | `E + T + 2` = **112** |
| `D_VOCAB` | Full vocab size | **113** |

### Fact and query format

- Each **fact** is a triple `(e, t, e2)`:

  - `e`: entity token in `0 … E-1`
  - `t`: relation token in `E … E+T-1`
  - `e2`: tail entity in `0 … E-1`

  On the wire: `[e, t, e2, SEP]`; multiple facts are concatenated.

- The **query** is `[Tq, Eq, Q]` where `(Eq, Tq, E2q)` is one of the sampled facts; the **supervision target** is the tail entity **`E2q`** (an integer in `0 … E-1`).

### Generative distribution (`produce_example_by_index`)

- **Determinism:** uses `numpy.random.default_rng(np.random.SeedSequence([SEED, idx]))` with **`SEED = 0`** and example index **`idx`**, so each index yields a fixed example unless you change `SEED` or the sampling logic.
- **Number of facts:** `k` drawn uniformly from **4 … 8** (inclusive).
- **Uniqueness:** `(e, t)` pairs are not repeated within an example (`seen` set).
- **Self-loops:** disallowed by default (`e2 != e` unless `allow_self_loops=True`).
- **Distractor (optional):** with probability **0.75**, if the fact count is still below `MAX_FACTS` (8), a **spurious** triple sharing `Eq` but with a different relation and tail may be inserted at a random index (only if `(Eq, distractor_t)` is not already present).

### Streaming (`TrainStream`)

- **`size`:** 80,000 examples per logical pass.
- **`offset`:** 20,000 — indices passed to `produce_example_by_index` are **`offset + local`** for `local ∈ {0, …, size-1}` (i.e. **20,000 … 99,999** before wrap).
- **`set_epoch`:** currently a no-op. The implementation computes `start = (self._epoch * self.size) % self.size`, which is `0` for every epoch regardless of the value passed in. The default training path does not call it, and calling it has no effect — `_epoch` is effectively dead state.

### Batching (`collate_fn`)

- Sequences are padded with **`PAD` (112)**.
- Labels use **`IGNORE_INDEX = -100`** everywhere except at the **single `Q` position**, where the label is the target entity id (the SAE pipeline in this repo uses tokens only and ignores labels).

---

## 3. Training pipeline

### Sparse autoencoder (this repository)

Configured in `minimal_sae_train.py` via `BatchTopKTrainer` from `dictionary_learning`.

| Hyperparameter | Value |
|----------------|--------|
| Learning rate | `1e-4` |
| Warmup steps | `1000` |
| Training steps (CLI default) | `30000` (`--steps`) |
| Training steps used for shipped `sae_ckpts/**` | **1,250,000** (`hook_z` heads) / **1,000,000** (`resid_post`) — see `sae_ckpts/**/config.json` |
| Batch size (CLI default) | `64` (`--batch-size`) |
| Sequence length passed to pipeline (default) | `64` (`--seq-len`) |
| Dictionary sizes (sweep) | 1024, 2048, 4096 |
| Top‑`k` (sweep) | 8, 16 |
| Submodule sites (script-enumerated) | `blocks.0.hook_resid_post` (sep), or `blocks.0.attn.hook_z` heads 0–1 (sep) |

**Logging / checkpoints:** `log_steps=100` by default; checkpoints at steps given by `--save-steps` or final step only.

### 3.1 BatchTopKTrainer (dictionary_learning package)

The following is traced from the upstream package **[saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning)** (`dictionary_learning/trainers/batch_top_k.py` and `dictionary_learning/trainers/trainer.py`). Pin the **`dictionary-learning` PyPI version** (or a git SHA) in the paper so reviewers can reproduce identical behavior if the library changes.

#### Optimizer

- **Class:** `torch.optim.Adam` on **all** `BatchTopKSAE` parameters.
- **Betas:** `(0.9, 0.999)`.
- **Weight decay:** not passed → PyTorch default **`0`** (not AdamW). Note: this is the **SAE trainer**; the separate LM-pretraining recipe in `scripts/seeds/train_lm.py` uses AdamW with `wd=1e-2`. Two distinct optimizers, do not conflate.
- **Learning rate:** base LR is `1e-4` in `minimal_sae_train.py` (when `lr` is omitted in other callers, the trainer auto-scales from dict size: `2e-4 / sqrt(dict_size / 2**14)`).

#### Learning-rate schedule

- **Scheduler:** `torch.optim.lr_scheduler.LambdaLR` with `get_lr_schedule(steps, warmup_steps, decay_start)`.
- **`minimal_sae_train.py`:** `warmup_steps=1000`, `decay_start` not passed → **`None`**.
- **Effect:** linear warmup from **0 → full LR** over steps `[0, warmup_steps)`; then multiplier **1.0** for the rest of training (no decay phase unless you set `decay_start`).

#### Loss and auxiliary (“dead feature”) objective

- **Reconstruction:** mean squared error on the residual `x - x_hat` (sum over activation dim, mean over batch).
- **Encoding:** batch top‑`k` on post-ReLU encoder activations (`encode(..., use_threshold=False)` during training); after `threshold_start_step`, a running **threshold** is updated from the minimum active activation (EMA with `threshold_beta`).
- **Aux loss:** `auxk_alpha * normalized_auxk_loss` with default **`auxk_alpha = 1/32`** (`minimal_sae_train.py` does not override this).
- **Dead features:** latents that have not “fired” for **`dead_feature_threshold = 10_000_000`** tokens (counter reset when active); auxiliary loss targets up to **`top_k_aux = activation_dim // 2`** of those (so **128** for `d_model=256`, **64** for `d_head=128`).
- **Total loss:** `l2_loss + auxk_alpha * auxk_loss`.

#### Other training mechanics

- **Step 0:** `b_dec` is set to the **geometric median** of the first activation batch (iterative Weiszfeld-style loop in-trainer).
- **Gradients:** component of the decoder weight gradient **parallel to decoder columns** is removed (`remove_gradient_parallel_to_decoder_directions`); **global grad clip** `clip_grad_norm_(..., max_norm=1.0)`.
- **After each step:** decoder columns are renormalized to **unit norm** (`set_decoder_norm_to_unit_norm`).
- **`k` annealing:** optional `k_anneal_steps`; **not** used by `minimal_sae_train.py` (defaults to `None`).

#### `BatchTopKSAE` architecture (for the methods section)

- **Decoder:** `nn.Linear(dict_size, activation_dim, bias=False)` — columns kept unit-norm during training.
- **Encoder:** `nn.Linear(activation_dim, dict_size)` with bias; weights initialized as **transpose of decoder**; encoder bias zero.
- **Learned pre-bias:** `b_dec` (vector in activation space), subtracted before encoding; added back on decode.

### Language model pretraining

**Not implemented in this repository.** For the paper, cite the original training recipe or Hugging Face model card for `sebastianhoenig/2L2H_Final` if available; otherwise report only what is **explicitly** in the checkpoint metadata or filename.

### Artifact caveat

Saved `sae_ckpts/**/config.json` files may show **different** `steps` (e.g. 1,500,000) than the current script defaults. Treat those JSON files as logs of **specific runs**, not as the canonical definition unless you align your prose with the command line used for that run.

---

## 4. Related work (not in repo)

This codebase does **not** include a bibliography or discussion of related mechanistic interpretability work. For the paper, you may want to connect to:

- **Sparse autoencoders / dictionary learning** on transformer activations (aligned with the `dictionary_learning` dependency and `BatchTopK` training).
- **Hook-based interpretability tooling** (e.g. TransformerLens-style activation extraction at chosen submodules).
- **Toy compositional / relational tasks** in small transformers, depending on the claims you make about binding or retrieval.

---

## 5. Quick file reference

| Topic | File |
|--------|------|
| Model config, data generation, SAE CLI, HF checkpoint load | `minimal_sae_train.py` |
| Position masking, `hook_z` head slice, training loop | `dl_quick_train/pipeline.py` |
| Generic package entry / Pile-style defaults (not toy LM) | `README.md`, `dl_quick_train/__init__.py` |
