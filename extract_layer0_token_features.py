#!/usr/bin/env python3
"""
Extract token-level features for Layer 0 SAE.
The original extraction only saved final token features, but we need all token features.
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download
from dictionary_learning import AutoEncoder
import os
from collections import Counter

# Constants from original script
E = 100  # num entities
T = 10   # num types/relations
SEP = E + T
Q = E + T + 1
PAD = E + T + 2
D_VOCAB = E + T + 3
IGNORE_INDEX = -100
VAL_SIZE = 20_000
SEED = 0
SAE_DIM = 4096

def produce_example_by_index(idx: int, *, allow_self_loops: bool = False):
    """Same data generation as original"""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, idx]))
    k = int(rng.integers(4, 8 + 1))  # MIN_FACTS=4, MAX_FACTS=8

    facts = []
    seen_head_rel = set()
    while len(facts) < k:
        e = int(rng.integers(0, E))
        t = int(rng.integers(0, T)) + E
        if (e, t) in seen_head_rel:
            continue
        e2 = int(rng.integers(0, E))
        while (not allow_self_loops) and e2 == e:
            e2 = int(rng.integers(0, E))
        seen_head_rel.add((e, t))
        facts.append((e, t, e2))

    q_idx = int(rng.integers(0, k))
    Eq, Tq, E2q = facts[q_idx]

    if rng.random() < 0.75 and len(facts) < 8:  # MAX_FACTS
        distractor_t = int(rng.integers(0, T)) + E
        while distractor_t == Tq:
            distractor_t = int(rng.integers(0, T)) + E
        distractor_e2 = int(rng.integers(0, E))
        while distractor_e2 == E2q:
            distractor_e2 = int(rng.integers(0, E))
        if (Eq, distractor_t) not in seen_head_rel:
            distractor_fact = (Eq, distractor_t, distractor_e2)
            insert_pos = int(rng.integers(0, len(facts) + 1))
            facts.insert(insert_pos, distractor_fact)

    seq = []
    for (e, t, e2) in facts:
        seq.extend([e, t, e2, SEP])
    seq.extend([Tq, Eq, Q])
    label = E2q
    return seq, label

class ValDataset(torch.utils.data.Dataset):
    def __len__(self): 
        return VAL_SIZE
    
    def __getitem__(self, i):
        seq, label = produce_example_by_index(i)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """Same collate function as original"""
    max_len = max(len(seq) for seq, _ in batch)
    B = len(batch)
    toks = torch.full((B, max_len), PAD, dtype=torch.long)
    target = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (seq, label) in enumerate(batch):
        x = torch.tensor(seq if isinstance(seq, list) else seq.tolist(), dtype=torch.long)
        L = len(x)
        toks[i, :L] = x
        q_pos = (x == Q).nonzero(as_tuple=False).squeeze()
        assert q_pos.numel() == 1, "Each example must have exactly one Q"
        target[i, q_pos.item()] = int(label)
    return toks, target

def build_model(n_layers: int, n_heads: int) -> HookedTransformer:
    """Build the same model as original"""
    d_model = 256
    n_ctx = 64
    d_head = d_model // n_heads

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        n_ctx=n_ctx,
        d_vocab=D_VOCAB,
        d_vocab_out=E,
        attn_only=True,
        normalization_type="LN",
        positional_embedding_type="rotary",
    )
    return HookedTransformer(cfg)

def extract_layer0_token_features():
    """Extract token-level features for layer 0"""
    
    print("🔍 Extracting Layer 0 Token-Level Features")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = build_model(2, 2)  # N_LAYERS=2, HEADS=2
    model = model.to(device)
    
    # Load pretrained weights
    REPO_ID = "sebastianhoenig/2L2H_Final"
    FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    pretrained_weights = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = pretrained_weights["model"]
    model.load_state_dict(state_dict)
    print("✓ Model loaded successfully")
    
    # Load Layer 0 SAE
    # First, check if we have the SAE checkpoint
    checkpoint_dir = "./sae_checkpoints_0_hook_resid_post"
    model_checkpoint_dir = f"{checkpoint_dir}/trainer_0/checkpoints/"
    
    if not os.path.exists(model_checkpoint_dir):
        print(f"❌ SAE checkpoint directory not found: {model_checkpoint_dir}")
        print("Please train Layer 0 SAE first by setting LAYER_TO_TRAIN=0 and running the main script")
        return
    
    checkpoints = [f for f in os.listdir(model_checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        print(f"❌ No SAE checkpoints found in {model_checkpoint_dir}")
        return
    
    # Use the latest checkpoint
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest_checkpoint = checkpoints[-1]
    sae_path = os.path.join(model_checkpoint_dir, latest_checkpoint)
    print(f"✓ Using SAE checkpoint: {latest_checkpoint}")
    
    # Load SAE
    sae = AutoEncoder(activation_dim=256, dict_size=SAE_DIM)
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae = sae.to(device)
    sae.eval()
    print("✓ SAE loaded successfully")
    
    # Create validation dataset
    val_dataset = ValDataset()
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)  # Smaller batch for memory
    print(f"✓ Validation dataset created ({len(val_dataset)} examples)")
    
    # Extract token-level features
    act_name = "blocks.0.hook_resid_post"  # Layer 0
    
    all_token_features = []
    all_sequences = []
    all_labels = []
    all_sequence_lengths = []
    
    model.eval()
    print("\n🚀 Extracting features...")
    
    with torch.no_grad():
        for batch_idx, (toks, labels) in enumerate(val_loader):
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx * toks.size(0)} / {len(val_dataset)} examples")
            
            toks = toks.to(device)
            labels = labels.to(device)
            
            # Get activations from the model for all tokens
            cache = model.run_with_cache(toks, names_filter=[act_name])[1]
            acts = cache[act_name]  # shape: [batch, seq_len, d_model]
            
            # Extract features for all tokens
            batch_size, seq_len, d_model = acts.shape
            
            # Reshape to process all tokens at once: [batch * seq_len, d_model]
            acts_flat = acts.reshape(-1, d_model)
            features_flat = sae.encode(acts_flat)  # [batch * seq_len, dict_size]
            
            # Reshape back to [batch, seq_len, dict_size]
            features = features_flat.reshape(batch_size, seq_len, SAE_DIM)
            
            # Store features and metadata for each example
            for i in range(batch_size):
                # Find actual sequence length (excluding padding)
                seq = toks[i]
                actual_len = (seq != PAD).sum().item()
                
                # Store the features for this sequence (only non-padded tokens)
                token_features = features[i, :actual_len, :].cpu()  # [actual_len, dict_size]
                
                all_token_features.append(token_features)
                all_sequences.append(seq[:actual_len].cpu())
                all_labels.append(labels[i].cpu())
                all_sequence_lengths.append(actual_len)
    
    print(f"✓ Feature extraction complete!")
    print(f"  Total examples: {len(all_token_features)}")
    print(f"  Average sequence length: {np.mean(all_sequence_lengths):.1f} tokens")
    print(f"  Token feature shape per example: [seq_len, {SAE_DIM}]")
    
    # Extract label entities for each example
    all_label_entities = []
    for i, labels in enumerate(all_labels):
        # Find the label entity (where target is not IGNORE_INDEX)
        label_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
        if len(label_positions) > 0:
            label_entity = labels[label_positions[0]].item()
        else:
            label_entity = -1  # No label found
        all_label_entities.append(label_entity)
    
    # Save the results
    save_data = {
        'token_features': all_token_features,  # List of tensors, each [seq_len, dict_size]
        'sequences': all_sequences,           # List of tensors, each [seq_len]
        'labels': all_labels,                 # List of tensors, each [seq_len] with mostly IGNORE_INDEX
        'label_entities': all_label_entities, # List of ints, the target entity for each example
        'sequence_lengths': all_sequence_lengths,  # List of ints
        'metadata': {
            'layer': 0,
            'sae_dim': SAE_DIM,
            'num_examples': len(all_token_features),
            'avg_seq_len': np.mean(all_sequence_lengths),
            'checkpoint_used': latest_checkpoint,
            'description': 'All token features with label entity associations'
        }
    }
    
    save_path = "layer_0_token_features_d4096.pt"
    torch.save(save_data, save_path)
    print(f"✅ Token-level features saved to {save_path}")
    
    # Quick analysis
    print(f"\n📊 Quick Analysis:")
    
    # Show label entity distribution
    label_counts = Counter(all_label_entities)
    valid_labels = [(entity, count) for entity, count in label_counts.items() if entity != -1]
    valid_labels.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Label entity distribution (top 10):")
    for entity, count in valid_labels[:10]:
        print(f"    Entity {entity}: appears as label in {count} examples")
    
    # Analyze sparsity across all tokens
    total_features = 0
    total_zeros = 0
    
    for token_features in all_token_features[:100]:  # Sample first 100 for speed
        total_features += token_features.numel()
        total_zeros += (token_features == 0).sum().item()
    
    sparsity = total_zeros / total_features
    print(f"  Sparsity (first 100 examples): {sparsity:.1%}")
    
    # Show example with label entity
    example_idx = 0
    example_features = all_token_features[example_idx]
    example_seq = all_sequences[example_idx]
    example_label_entity = all_label_entities[example_idx]
    
    print(f"\n  Example {example_idx}:")
    print(f"    Sequence length: {len(example_seq)} tokens")
    print(f"    Feature shape: {example_features.shape}")
    print(f"    Label entity: {example_label_entity}")
    
    # Show which tokens have the most active features
    active_counts = (example_features > 0).sum(dim=1)  # Active features per token
    print(f"    Active features per token: {active_counts.tolist()}")
    
    # Show readable sequence with label highlighting
    readable_tokens = []
    for i, token in enumerate(example_seq):
        token_val = token.item()
        if token_val < E:
            token_str = f"E{token_val}"
            if token_val == example_label_entity:
                token_str += "*"  # Mark label entity
        elif token_val < E + T:
            token_str = f"T{token_val - E}"
        elif token_val == SEP:
            token_str = "SEP"
        elif token_val == Q:
            token_str = "Q"
        else:
            token_str = f"UNK{token_val}"
        readable_tokens.append(token_str)
    
    print(f"    Readable sequence: {' '.join(readable_tokens)}")
    print(f"    (* = label entity)")

if __name__ == "__main__":
    extract_layer0_token_features()