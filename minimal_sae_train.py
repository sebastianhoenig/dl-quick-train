#!/usr/bin/env python3
import argparse
import os
import torch
import numpy as np

from torch.utils.data import DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download
from dictionary_learning.trainers.standard import StandardTrainer
from dl_quick_train.pipeline import run_pipeline


E = 100
T = 10
SEP = E + T        # 110
Q   = E + T + 1    # 111
PAD = E + T + 2    # 112
D_VOCAB = E + T + 3

IGNORE_INDEX = -100

def produce_example_by_index(idx: int, *, allow_self_loops: bool = False):
    MIN_FACTS, MAX_FACTS, SEED = 4, 8, 0
    rng = np.random.default_rng(np.random.SeedSequence([SEED, idx]))
    k = int(rng.integers(MIN_FACTS, MAX_FACTS + 1))
    facts, seen = [], set()
    while len(facts) < k:
        e = int(rng.integers(0, E)); t = int(rng.integers(0, T)) + E
        if (e, t) in seen: continue
        e2 = int(rng.integers(0, E))
        while (not allow_self_loops) and e2 == e:
            e2 = int(rng.integers(0, E))
        seen.add((e, t)); facts.append((e, t, e2))
    q_idx = int(rng.integers(0, k))
    Eq, Tq, E2q = facts[q_idx]
    if rng.random() < 0.75 and len(facts) < MAX_FACTS:
        distractor_t = int(rng.integers(0, T)) + E
        while distractor_t == Tq: distractor_t = int(rng.integers(0, T)) + E
        distractor_e2 = int(rng.integers(0, E))
        while distractor_e2 == E2q: distractor_e2 = int(rng.integers(0, E))
        if (Eq, distractor_t) not in seen:
            ins = int(rng.integers(0, len(facts)+1))
            facts.insert(ins, (Eq, distractor_t, distractor_e2))
    seq = []
    for (e, t, e2) in facts:
        seq.extend([e, t, e2, SEP])
    seq.extend([Tq, Eq, Q])
    label = E2q
    return seq, label

class TrainStream(IterableDataset):
    def __init__(self, size=80_000, offset=20_000):
        super().__init__()
        self.size = size
        self.offset = offset
        self._epoch = 0
    def __len__(self): return self.size
    def set_epoch(self, e:int): self._epoch = e
    def __iter__(self):
        start = (self._epoch * self.size) % self.size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.size):
            local = (start + i) % self.size
            idx = self.offset + local
            seq, label = produce_example_by_index(idx)
            yield torch.tensor(seq, dtype=torch.long, device=device), torch.tensor(label, dtype=torch.long, device=device)

def collate_fn(batch):
    B = len(batch)
    max_len = max(len(seq) for seq, _ in batch)
    toks = torch.full((B, max_len), PAD, dtype=torch.long)
    # labels are ignored by the pipeline, but we return them for compatibility
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (seq, label) in enumerate(batch):
        seq = torch.tensor(seq if isinstance(seq, list) else seq.tolist(), dtype=torch.long)
        L = len(seq)
        toks[i, :L] = seq
        q_pos = (seq == Q).nonzero(as_tuple=False).squeeze()
        assert q_pos.numel() == 1
        labels[i, q_pos.item()] = int(label)
    return toks, labels

class CustomDatasetWrapper:
    """
    Small iterator wrapper so pipeline can do `next(loader)` and also reset via
    `loader.iterator = None` on StopIteration.
    """
    def __init__(self, dataset, batch_size=64):
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterator = None
    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn))
        return self.iterator
    def __len__(self): return len(self.dataset)
    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn))
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collate_fn))
            return next(self.iterator)


def build_model():
    cfg = HookedTransformerConfig(
        n_layers=2, n_heads=2, d_model=256, d_head=128,
        n_ctx=64, d_vocab=D_VOCAB, d_vocab_out=E,
        attn_only=True, normalization_type="LN", positional_embedding_type="rotary",
    )
    return HookedTransformer(cfg)

def load_weights(model, device):
    repo = "sebastianhoenig/2L2H_Final"
    fname = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
    path = hf_hub_download(repo_id=repo, filename=fname)
    sd = torch.load(path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(sd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="./sae_ckpts")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="hoenigsebastian-eth-z-rich")
    parser.add_argument("--wandb-project", type=str, default="SAE")
    parser.add_argument("--submodule", type=str, required=True,
                        help='e.g. "blocks.0.hook_resid_post" or "blocks.0.attn.hook_z"')
    parser.add_argument("--position-selector", type=str, default="sep", choices=["sep", "q"],
                        help="Select activations at SEP (110) or Q (111) positions (default: sep).")
    parser.add_argument("--head-index", type=int, default=None,
                        help="For blocks.0.attn.hook_z: choose a head (e.g., 0)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    load_weights(model, device)

    # prepare tokenized stream (no tokenizer needed)
    train_stream = TrainStream()
    wrapped = CustomDatasetWrapper(train_stream, batch_size=args.batch_size)

    # single SAE config (edit as needed)
    trainer_cfg = dict(
        trainer=StandardTrainer,
        steps=args.steps,
        activation_dim=256,  # for resid_post; for hook_z head selection we pass d_head via the trainer config usually
        dict_size=4096,
        lr=1e-4,
        l1_penalty=1e-1,
        warmup_steps=1000,
        sparsity_warmup_steps=2000,
        lm_name="toy_binding",
        wandb_name="toy_binding_sae",
    )

    run_pipeline(
        [trainer_cfg],
        device=device,
        model_name="custom",
        submodule=args.submodule,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        use_wandb=args.use_wandb,
        use_transformer_lens=True,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        save_dir=args.save_dir,
        log_steps=100,
        verbose=True,
        custom_model=model,
        custom_dataset=wrapped,
        position_selector=args.position_selector,
        head_index=args.head_index,
    )

if __name__ == "__main__":
    main()
