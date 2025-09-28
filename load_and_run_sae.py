#!/usr/bin/env python3
"""
Script to load a trained SAE model and run it over the dataset.
This script loads the SAE from the specified checkpoint and processes the dataset.
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from transformer_lens import HookedTransformer, HookedTransformerConfig
from huggingface_hub import hf_hub_download
from dictionary_learning.trainers.jumprelu import JumpReluAutoEncoder
import json
from typing import Optional

# Constants from the training code
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

def build_model():
    """Build the transformer model"""
    cfg = HookedTransformerConfig(
        n_layers=2, n_heads=2, d_model=256, d_head=128,
        n_ctx=64, d_vocab=D_VOCAB, d_vocab_out=E,
        attn_only=True, normalization_type="LN", positional_embedding_type="rotary",
    )
    return HookedTransformer(cfg)

def load_weights(model, device):
    """Load pretrained weights for the transformer"""
    repo = "sebastianhoenig/2L2H_Final"
    fname = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
    path = hf_hub_download(repo_id=repo, filename=fname)
    sd = torch.load(path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(sd)

def load_sae_from_checkpoint(checkpoint_path, device):
    """Load SAE model from checkpoint"""
    print(f"Loading SAE from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the config to get SAE parameters
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    trainer_config = config["trainer"]
    activation_dim = trainer_config["activation_dim"]
    dict_size = trainer_config["dict_size"]
    
    print(f"SAE config: activation_dim={activation_dim}, dict_size={dict_size}")
    
    # Create SAE model
    sae = JumpReluAutoEncoder(activation_dim=activation_dim, dict_size=dict_size, device=device)
    sae.load_state_dict(checkpoint)
    sae.eval()
    
    return sae, trainer_config

def extract_activations_at_positions(activations, tokens, submodule, position_selector="sep", head_index=None):
    """Extract activations at specific positions (SEP or Q tokens)
    Returns activations_2d and a list of (batch_index, position_index) for mapping back.
    """
    if position_selector == "sep":
        mask = tokens == SEP
    elif position_selector == "q":
        B, S = tokens.shape
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        q_locs = (tokens == Q).nonzero(as_tuple=False)
        for b in range(B):
            row = (q_locs[:, 0] == b).nonzero(as_tuple=False).squeeze(-1)
            if row.numel() >= 1:
                q_pos = q_locs[row[0], 1].item()
                mask[b, q_pos] = True
    else:
        raise ValueError(f"Unknown position selector: {position_selector}")
    
    # Handle different activation shapes
    if submodule.endswith(".attn.hook_z"):
        if head_index is not None:
            activations = activations[:, :, head_index, :]
    elif submodule.endswith(".hook_resid_post"):
        pass  # Already correct shape
    
    # Flatten to 2D while tracking indices
    B, S = mask.shape
    d = activations.shape[-1]
    flat_mask = mask.reshape(B * S)
    activations_2d = activations.reshape(B * S, d)[flat_mask]
    # indices of selected positions (batch_idx, seq_pos)
    nonzero_flat = flat_mask.nonzero(as_tuple=False).squeeze(-1)
    pos_indices = [(int(i // S), int(i % S)) for i in nonzero_flat]
    
    return activations_2d, pos_indices

def _format_example_text(tokens_1d: torch.Tensor, position_selector="sep", label: Optional[int] = None):
    """
    Convert a token sequence into a human-readable string, replacing SEP with ',' and Q with '?'.
    Example: "E82 T4 E56, E49 T5 E66, ... E49 T5?"
    Returns: (text, positions)
      - text: formatted string
      - positions: list of indices where ',' or '?' appears, depending on position_selector
    """
    toks = tokens_1d.detach().cpu().tolist()
    # Remove trailing PADs
    end = len(toks)
    while end > 0 and toks[end - 1] == PAD:
        end -= 1
    toks = toks[:end]

    text_parts = []
    positions = []
    i = 0
    # Parse all but the last triplet (which ends with Q)
    while i + 3 < len(toks):
        e = toks[i]
        t = toks[i + 1]
        e2 = toks[i + 2]
        sep_or_q = toks[i + 3]
        text_parts.append(f"E{e} T{t - E + 1} E{e2}")
        if sep_or_q == SEP:
            text_parts.append(",")
            if position_selector == "sep":
                positions.append(i + 3)
        elif sep_or_q == Q:
            text_parts.append("?")
            if position_selector == "q":
                positions.append(i + 3)
        else:
            # Unexpected token, just continue
            pass
        i += 4

    # Handle the last triplet if it ends with Q (should always be the case)
    if i + 2 < len(toks) and toks[-1] == Q:
        t = toks[i]
        e = toks[i + 1]
        q = toks[i + 2]
        text_parts.append(f"T{t - E + 1} E{e} ")
        text_parts.append("?")
        if position_selector == "q":
            positions.append(i + 2)

    text = " ".join(text_parts).replace(" ,", ",").replace(" ?", "?")
    if label is not None:
        text = f"{text} E{label}"
    return text, positions

def run_sae_on_dataset(model, sae, dataset, submodule, position_selector="sep", head_index=None, device="cuda", output_prefix=None, max_examples_per_feature=None):
    """Run SAE on the dataset, collect statistics, and map features to examples.
    If output_prefix is provided, saves two files:
      - f"{output_prefix}.pt": torch.save dict with feature_to_examples and example_data
      - f"{output_prefix}.txt": human-readable mapping per feature
    """
    model.eval()
    sae.eval()
    
    all_activations = []
    all_reconstructions = []
    all_features = []
    all_labels = []
    
    total_samples = 0
    total_l0 = 0.0
    total_mse = 0.0
    total_variance_explained = 0.0
    
    print(f"Running SAE on dataset with {len(dataset)} samples...")

    # Mappings
    # feature_id -> list of entries {example_id, position, value}
    feature_to_examples = {}
    # example_id -> {text, comma_positions, positions: [{pos, feature_values}]}
    example_data = {}
    examples_seen = 0
    
    with torch.no_grad():
        for batch_idx, (tokens, labels) in enumerate(dataset):
            if batch_idx % 100 == 0:
                print(f"Processing batch {batch_idx}")
            
            # Get activations from the model
            _, cache = model.run_with_cache(
                tokens.to(device),
                names_filter=[submodule],
            )
            activations = cache[submodule]
            
            # Extract activations at specific positions
            activations_2d, pos_indices = extract_activations_at_positions(
                activations, tokens, submodule, position_selector, head_index
            )
            
            if activations_2d.shape[0] == 0:
                continue
            
            # Run through SAE
            reconstructions, features = sae(activations_2d, output_features=True)
            
            # Compute statistics
            l0 = (features != 0).float().sum(dim=-1).mean().item()
            mse = torch.nn.functional.mse_loss(activations_2d, reconstructions).item()
            
            # Variance explained
            total_variance = torch.var(activations_2d, dim=0).sum().item()
            residual_variance = torch.var(activations_2d - reconstructions, dim=0).sum().item()
            variance_explained = 1 - residual_variance / total_variance if total_variance > 0 else 0
            
            # Update running totals
            batch_size = activations_2d.shape[0]
            total_samples += batch_size
            total_l0 += l0 * batch_size
            total_mse += mse * batch_size
            total_variance_explained += variance_explained * batch_size
            
            # Prepare per-example mapping and strings
            B, S = tokens.shape
            # initialize example entries for this batch
            for b in range(B):
                ex_id = examples_seen + b
                if ex_id not in example_data:
                    # Derive label for this example (first non-IGNORE_INDEX entry in labels[b])
                    lb_row = labels[b]
                    nz = (lb_row != IGNORE_INDEX).nonzero(as_tuple=False).squeeze(-1)
                    ex_label: Optional[int] = None
                    if nz.numel() >= 1:
                        ex_label = int(lb_row[nz[0]].item())
                    text, comma_positions = _format_example_text(tokens[b], position_selector, label=ex_label)
                    example_data[ex_id] = {
                        "text": text,
                        "comma_positions": comma_positions,
                        "positions": [],
                    }
            # For each selected position row, attach feature vector to the example and map top activations
            for row_idx, (b_idx, pos_idx) in enumerate(pos_indices):
                ex_id = examples_seen + b_idx
                fv = features[row_idx].detach().cpu()
                example_data[ex_id]["positions"].append({
                    "pos": int(pos_idx),
                    "feature_values": fv,
                })
                # Map nonzero features
                nz = (fv != 0).nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(nz, int):
                    nz = [nz]
                for feat_id in nz:
                    # optional cap per feature list size
                    lst = feature_to_examples.setdefault(int(feat_id), [])
                    if (max_examples_per_feature is not None) and (len(lst) >= max_examples_per_feature):
                        continue
                    # record value at this row
                    value = float(fv[int(feat_id)].item())
                    lst.append({
                        "example_id": int(ex_id),
                        "position": int(pos_idx),
                        "value": value,
                    })
            
            # Store for analysis (limit memory usage for raw tensors)
            if batch_idx < 10:  # Only store first 10 batches for analysis
                all_activations.append(activations_2d.cpu())
                all_reconstructions.append(reconstructions.cpu())
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

            # increment global example counter by batch size
            examples_seen += B
    
    # Compute final statistics
    avg_l0 = total_l0 / total_samples if total_samples > 0 else 0
    avg_mse = total_mse / total_samples if total_samples > 0 else 0
    avg_variance_explained = total_variance_explained / total_samples if total_samples > 0 else 0
    
    print(f"\nSAE Results:")
    print(f"Total samples processed: {total_samples}")
    print(f"Average L0 (sparsity): {avg_l0:.4f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average variance explained: {avg_variance_explained:.4f}")

    # Save mappings if requested
    if output_prefix is not None:
        save_obj = {
            "feature_to_examples": feature_to_examples,
            "example_data": example_data,
            "stats": {
                "total_samples": total_samples,
                "avg_l0": avg_l0,
                "avg_mse": avg_mse,
                "avg_variance_explained": avg_variance_explained,
            },
            "meta": {
                "position_selector": position_selector,
                "submodule": submodule,
            },
        }
        pt_path = f"{output_prefix}.pt"
        torch.save(save_obj, pt_path)
        # Also write a compact human-readable text file grouped by feature
        txt_path = f"{output_prefix}.txt"
        with open(txt_path, "w") as f:
            f.write(f"SAE feature → examples mapping (position_selector={position_selector})\n")
            # Sort features for stable output
            for feat_id in sorted(feature_to_examples.keys()):
                f.write(f"\nFeature {feat_id}:\n")
                for entry in feature_to_examples[feat_id]:
                    ex = example_data.get(entry["example_id"], None)
                    if ex is None:
                        continue
                    text = ex["text"]
                    commas = ex["comma_positions"]
                    f.write(f"  ex={entry['example_id']} pos={entry['position']} val={entry['value']:.6f} | {text} | commas={commas}\n")
    
    return {
        'total_samples': total_samples,
        'avg_l0': avg_l0,
        'avg_mse': avg_mse,
        'avg_variance_explained': avg_variance_explained,
        'activations': torch.cat(all_activations, dim=0) if all_activations else None,
        'reconstructions': torch.cat(all_reconstructions, dim=0) if all_reconstructions else None,
        'features': torch.cat(all_features, dim=0) if all_features else None,
        'labels': torch.cat(all_labels, dim=0) if all_labels else None,
    }

def main():
    parser = argparse.ArgumentParser(description='Load SAE model and run on dataset')
    parser.add_argument('--checkpoint-path', type=str, 
                       default='sae_ckpts_susie_l0_7_dim_tuning/trainer_0/checkpoints/ae_99500.pt',
                       help='Path to SAE checkpoint')
    parser.add_argument('--submodule', type=str, default='blocks.0.hook_resid_post',
                       help='Submodule to extract activations from')
    parser.add_argument('--position-selector', type=str, default='sep', choices=['sep', 'q'],
                       help='Position selector: sep or q')
    parser.add_argument('--head-index', type=int, default=None,
                       help='Head index for attention activations')
    parser.add_argument('--dataset-size', type=int, default=1000,
                       help='Number of samples to process')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--output-prefix', type=str, default=None,
                       help='If set, save mappings to {prefix}.pt and {prefix}.txt')
    parser.add_argument('--max-examples-per-feature', type=int, default=None,
                       help='Optional cap on number of examples saved per feature')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load transformer model
    print("Loading transformer model...")
    model = build_model().to(device)
    load_weights(model, device)
    print("Transformer model loaded successfully")
    
    # Load SAE
    sae, sae_config = load_sae_from_checkpoint(args.checkpoint_path, device)
    print("SAE loaded successfully")
    
    # Create dataset
    print("Creating dataset...")
    dataset = TrainStream(size=args.dataset_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Run SAE on dataset
    results = run_sae_on_dataset(
        model, sae, dataloader, 
        submodule=args.submodule,
        position_selector=args.position_selector,
        head_index=args.head_index,
        device=device,
        output_prefix=args.output_prefix,
        max_examples_per_feature=args.max_examples_per_feature,
    )
    
    print(f"\nFinal Results Summary:")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Submodule: {args.submodule}")
    print(f"Position selector: {args.position_selector}")
    print(f"Head index: {args.head_index}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Average L0: {results['avg_l0']:.4f}")
    print(f"Average MSE: {results['avg_mse']:.6f}")
    print(f"Average variance explained: {results['avg_variance_explained']:.4f}")

if __name__ == "__main__":
    main()
