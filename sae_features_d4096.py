# !pip install transformer-lens dictionary-learning

SAE_DIM = 4096 #16384 #1024 # 4096
LAYER_TO_TRAIN = 1  # also edit pipeline.py if changing this
TRAIN_LAST_LAYER = True if LAYER_TO_TRAIN == 1 else False 

"""### Train SAE on Toy Transformer using run_pipeline for w&b logging and parallel training"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader
import os
import json

from huggingface_hub import hf_hub_download
from dl_quick_train.pipeline import run_pipeline
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning import AutoEncoder, utils

try:
    from IPython.display import clear_output
    clear_output()
except ImportError:
    pass  # clear_output not available, continue anyway

import numpy as np
import pandas as pd
from tqdm import tqdm

E = 100  # num entities
T = 10   # num types/relations

SEP = E + T
Q = E + T + 1
PAD = E + T + 2
D_VOCAB = E + T + 3

IGNORE_INDEX = -100
ENTITIES = np.arange(0, E)
TYPES    = np.arange(E, E + T)

N_WORLDS = 80_000
MIN_FACTS, MAX_FACTS = 4, 8
SEED = 0

rng = np.random.default_rng(SEED)

def produce_example_by_index(idx: int, *, allow_self_loops: bool = False):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, idx]))

    k = int(rng.integers(MIN_FACTS, MAX_FACTS + 1))

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

    if rng.random() < 0.75 and len(facts) < MAX_FACTS:
        distractor_t = int(rng.integers(0, T)) + E

        while distractor_t == Tq: # Ensure the relation is different
            distractor_t = int(rng.integers(0, T)) + E

        distractor_e2 = int(rng.integers(0, E))
        while distractor_e2 == E2q: # Ensure the tail is different
            distractor_e2 = int(rng.integers(0, E))

        # Add the distractor fact IF it doesn't create a collision
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

TOTAL_TRAIN = 16_100_000
BLOCK_SIZE  = 80_000
VAL_SIZE = 20_000
TRAIN_OFFSET = VAL_SIZE
TRAIN_SIZE   = TOTAL_TRAIN
BASE_SEED = 0

class ValDataset(torch.utils.data.Dataset):
    def __len__(self): return VAL_SIZE
    def __getitem__(self, i):
        seq, label = produce_example_by_index(i)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(seq, dtype=torch.long, device=device), torch.tensor(label, dtype=torch.long, device=device)


class TrainStream(torch.utils.data.IterableDataset):
    def __init__(self, block_size=BLOCK_SIZE, offset=TRAIN_OFFSET, size=TRAIN_SIZE):
        super().__init__()
        self.block_size = block_size
        self.offset = offset
        self.size = size
        self._epoch = 0

    def set_epoch(self, epoch:int):
        self._epoch = epoch

    def __iter__(self):
        # compute which block to serve this epoch, with wrap-around
        start_in_train = (self._epoch * self.block_size) % self.size
        # stream exactly block_size samples each epoch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(self.block_size):
            local_idx = (start_in_train + i) % self.size
            global_idx = self.offset + local_idx
            seq, label = produce_example_by_index(global_idx)
            x = torch.tensor(seq, dtype=torch.long, device=device)
            y = torch.tensor(label, dtype=torch.long, device=device)
            yield x, y

    def __len__(self):
        return self.block_size

val_dataset = ValDataset()
train_dataset = TrainStream()

N_LAYERS = 2
HEADS = 2

d_model = 256
n_ctx   = 64

def build_model(n_layers: int, n_heads: int) -> HookedTransformer:
    if d_model % n_heads != 0:
        return None
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

def collate_fn(batch):
    max_len = max(len(seq) for seq,_ in batch)
    B = len(batch)
    toks   = torch.full((B, max_len), PAD, dtype=torch.long)
    target = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)

    for i, (seq, label) in enumerate(batch):
        x = torch.tensor(seq if isinstance(seq, list) else seq.tolist(), dtype=torch.long)
        L = len(x)
        toks[i, :L] = x
        q_pos = (x == Q).nonzero(as_tuple=False).squeeze()
        assert q_pos.numel() == 1, "Each example must have exactly one Q"
        target[i, q_pos.item()] = int(label)
    return toks, target

# Create custom dataset wrapper for run_pipeline
class CustomDatasetWrapper:
    def __init__(self, dataset, collate_fn, batch_size=64):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.iterator = None
        
    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn))
        return self.iterator
    
    def __len__(self):
        return len(self.dataset)
    
    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn))
        try:
            return next(self.iterator)
        except StopIteration:
            # Reset iterator for next epoch
            self.iterator = None
            self.iterator = iter(DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn))
            return next(self.iterator)

def train_sae_with_pipeline(model, layer_to_train=1, sae_dim=SAE_DIM, use_wandb=True, checkpoint_dir=None):
    """Train a single SAE using run_pipeline"""
    
    try:
        # Create simple trainer configuration for single SAE
        trainer_config = {
            "trainer": StandardTrainer,
            "steps": 100_000,
            "activation_dim": 256,  # d_model
            "dict_size": sae_dim,
            "layer": layer_to_train,
            "lm_name": f"entity_binding_model_layer_{layer_to_train}",
            "wandb_name": f"StandardTrainer_L{layer_to_train}_D{sae_dim}",
            "lr": 1e-4,
            "l1_penalty": 1e-1,
            "warmup_steps": 1000,
            "sparsity_warmup_steps": 2000,
        }
        
        print(f"Created trainer configuration for layer {layer_to_train}")
        
        # Wrap our custom dataset
        wrapped_dataset = CustomDatasetWrapper(train_dataset, collate_fn, batch_size=64)

        print("Starting pipeline training...")
        print(f"Pipeline device: {device}")
        
        # Test GPU memory before training
        if device.type == 'cuda':
            print(f"GPU memory before training: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
        # Define submodule based on the layer to train
        submodule = f"blocks.{layer_to_train}.hook_resid_post"
        print(f"Using submodule: {submodule}")
        
        # Run pipeline
        try:
            run_ids = run_pipeline(
                [trainer_config],  # Single config in a list
                device=str(device),
                model_name="custom",
                dataset_name="custom",
                submodule=submodule,
                steps= 20_000,  # Reduced for testing
                batch_size=64,
                seq_len=64,
                use_wandb=use_wandb,
                use_transformer_lens=True,
                wandb_entity="iamsusie-columbia-university",
                wandb_project="sae_lens_training",
                save_dir=checkpoint_dir,
                log_steps=500,
                verbose=True,
                save_steps=[19_500],
                custom_model=model,
                custom_dataset=wrapped_dataset
            )
            
            return run_ids, trainer_config
        except Exception as e:
            print(f"Error in run_pipeline: {e}")
            print("Returning None for run_ids and continuing...")
            return [], trainer_config
        
    except Exception as e:
        print(f"Error in train_sae_with_pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise

def find_separator_after_correct_entity(tokens, labels):
    """
    Find the separator token that comes right after the correct entity label in the context.
    
    Args:
        tokens: tensor of shape [seq_len] containing the token sequence
        labels: tensor of shape [seq_len] containing the labels (IGNORE_INDEX for non-query positions)
    
    Returns:
        int: position of the separator token after the correct entity, or None if not found
    """
    # Find the query position (where Q token is)
    q_pos = (tokens == Q).nonzero(as_tuple=False)
    if q_pos.numel() == 0:
        return None
    q_pos = q_pos[0].item()
    
    # Get the correct answer (entity) from the label at query position
    correct_entity = labels[q_pos].item()
    if correct_entity == IGNORE_INDEX:
        return None
    
    # Look through the context (before the query) to find the correct entity
    context = tokens[:q_pos]
    
    # Find all positions where the correct entity appears
    entity_positions = (context == correct_entity).nonzero(as_tuple=False).squeeze()
    if entity_positions.numel() == 0:
        return None
    
    # Handle both single position and multiple positions
    if entity_positions.dim() == 0:
        # Single position case
        entity_positions = [entity_positions.item()]
    else:
        # Multiple positions case
        entity_positions = entity_positions.tolist()
    
    # For each entity position, check if it's followed by a separator
    for entity_pos in entity_positions:
        # Check if the next token is a separator
        if entity_pos + 1 < len(context) and context[entity_pos + 1] == SEP:
            return entity_pos + 1
    
    return None

def extract_sae_features(layer, sae_path, val_loader, device, model):
    """Extract SAE features from a trained model"""
    
    # Convert device string to torch device if needed
    if isinstance(device, str):
        torch_device = torch.device(device)
    else:
        torch_device = device
    
    # Create SAE instance
    sae = AutoEncoder(
        activation_dim=256,
        dict_size=SAE_DIM
    )
    
    try:
        sae.load_state_dict(torch.load(sae_path, map_location=torch_device))
        print(f"Successfully loaded SAE from {sae_path}")
    except FileNotFoundError:
        print(f"SAE file not found at {sae_path}. Please train a SAE first.")
        return None
    
    act_name = f"blocks.{layer}.hook_resid_post"
    
    # Extract features
    all_features = []
    if TRAIN_LAST_LAYER:
        all_tokens_features = []
    all_activations = []
    all_reconstructions = []
    all_labels = []
    
    sae.eval()
    model.eval()
    print_once = False if TRAIN_LAST_LAYER else True
    with torch.no_grad():
        for toks, labels in val_loader:
            toks = toks.to(torch_device)
            labels = labels.to(torch_device)
            
            # Get activations from the model
            cache = model.run_with_cache(toks, names_filter=[act_name])[1]
            acts = cache[act_name]  # shape: [batch, seq, d_model]
            
            if TRAIN_LAST_LAYER:
            # Extract activations at question mark token position instead of last token
                batch_size = acts.shape[0]
                acts_at_q = torch.zeros(batch_size, acts.shape[2], device=acts.device)
                for i in range(batch_size):
                    # Find Q token position in this sequence
                    q_pos = (toks[i] == Q).nonzero(as_tuple=False)
                    if q_pos.numel() > 0:
                        q_pos = q_pos[0].item()
                        acts_at_q[i] = acts[i, q_pos, :]
                    else:
                        raise ValueError(f"Q token not found in sequence {toks[i]}")
                        # Fallback to last token if Q not found
                        # acts_at_q[i] = acts[i, -1, :]
                acts = acts_at_q
            # Extract features
            features = sae.encode(acts)
            reconstruction = sae.decode(features)
            if print_once:
                # For each row in the first batch, print which feature indices are > 0
                features_np = features[0, :, :].cpu().numpy() if features.device.type != 'cpu' else features[0, :, :].numpy()
                for i, row in enumerate(features_np):
                    active_indices = np.where(row > 0)[0]
                    active_values = row[active_indices]
                    print(f"Row {i}: Indices > 0: {active_indices.tolist()}")
                    print(f"        Feature Values: {active_values.tolist()}")
                
                # Print first batch tokens and labels in readable format
                print("\n=== First Tokens and Labels ===")
                # Get sequence and labels for this example
                seq = toks[0]
                label = labels[0]
                
                # Find the position of the query token (Q)
                q_pos = (seq == Q).nonzero(as_tuple=False).squeeze()
                if q_pos.numel() > 0:
                    q_pos = q_pos.item()
                    # Get the sequence up to the query
                    context_seq = seq[:q_pos]
                    # Remove padding tokens
                    context_seq = context_seq[context_seq != PAD]
                    
                    # Convert to readable format
                    readable_seq = []
                    for token in context_seq:
                        if token < E:
                            readable_seq.append(f"E{token.item()}")
                        elif token < E + T:
                            readable_seq.append(f"T{token.item() - E}")
                        elif token == SEP:
                            readable_seq.append("SEP")
                        else:
                            readable_seq.append(f"UNK{token.item()}")
                    
                    # Get the query components and answer
                    query_rel = seq[q_pos - 2].item()  # Relation type (Tq)
                    query_ent = seq[q_pos - 1].item()  # Entity (Eq)
                    answer = label[q_pos].item() if label[q_pos] != IGNORE_INDEX else "IGNORE"
                    
                    # Find the separator position after the correct entity
                    sep_pos = find_separator_after_correct_entity(seq, label)
                    
                    print(f"  Tokens: {' '.join(readable_seq)} Q")
                    print(f"  Query: T{query_rel - E} E{query_ent} Q")
                    print(f"  Answer: E{answer}")
                    if sep_pos is not None:
                        print(f"  Separator after correct entity at position: {sep_pos}")
                        # Show the context around the separator
                        start_idx = max(0, sep_pos - 3)
                        end_idx = min(len(seq), sep_pos + 2)
                        context_around_sep = []
                        for i in range(start_idx, end_idx):
                            token = seq[i]
                            marker = " <-- SEP" if i == sep_pos else ""
                            if token < E:
                                context_around_sep.append(f"E{token.item()}{marker}")
                            elif token < E + T:
                                context_around_sep.append(f"T{token.item() - E}{marker}")
                            elif token == SEP:
                                context_around_sep.append(f"SEP{marker}")
                            elif token == Q:
                                context_around_sep.append(f"Q{marker}")
                            else:
                                context_around_sep.append(f"UNK{token.item()}{marker}")
                        print(f"  Context around separator: {' '.join(context_around_sep)}")
                    else:
                        print(f"  No separator found after correct entity")
                else:
                    print(f"  No query token found in sequence")
                    # Still show the sequence in readable format
                    readable_seq = []
                    for token in seq:
                        if token == PAD:
                            break
                        elif token < E:
                            readable_seq.append(f"E{token.item()}")
                        elif token < E + T:
                            readable_seq.append(f"T{token.item() - E}")
                        elif token == SEP:
                            readable_seq.append("SEP")
                        elif token == Q:
                            readable_seq.append("Q")
                        else:
                            readable_seq.append(f"UNK{token.item()}")
                    print(f"  Tokens: {' '.join(readable_seq)}")
                
                print_once = False
            if TRAIN_LAST_LAYER:
                all_features.append(features.cpu())
                all_tokens_features.append(features.cpu())
            else:
                # all_features should keep features at the separator token position that comes after the correct entity label
                batch_size = features.shape[0]
                features_at_sep = torch.zeros(batch_size, features.shape[2], device=features.device)
                for i in range(batch_size):
                    # Find the separator token that comes after the correct entity label
                    sep_pos = find_separator_after_correct_entity(toks[i], labels[i])
                    if sep_pos is not None:
                        features_at_sep[i] = features[i, sep_pos, :]
                    else:
                        raise ValueError(f"Separator after correct entity not found in sequence {toks[i]}")
                all_features.append(features_at_sep.cpu())
                
            all_activations.append(acts.cpu())
            all_reconstructions.append(reconstruction.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)
    all_activations = torch.cat(all_activations, dim=0)
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    res = {
        'features': all_features,
        'activations': all_activations,
        'reconstructions': all_reconstructions,
        'labels': all_labels
    }
    if TRAIN_LAST_LAYER:
        res['tokens_features'] = all_tokens_features

    return res

def analyze_features(features_dict, val_loader=None):
    """Analyze extracted SAE features"""
    all_features = features_dict['features']
    all_reconstructions = features_dict['reconstructions']
    all_labels = features_dict['labels']
    
    print("\n=== SAE Feature Analysis ===")
    
    # Sparsity analysis
    sparsity = (all_features == 0).float().mean()
    print(f"Overall feature sparsity: {sparsity:.4f} ({sparsity*100:.2f}% of features are zero)")
    
    # Feature activation frequency
    feature_activations = (all_features > 0).float()
    feature_frequency = feature_activations.mean(dim=0)
    # print("feature_frequency", feature_frequency)
    feature_sum = feature_activations.sum(dim=0)
    # print("feature_sum", feature_sum)
    print(f"\nTop 10 most active features:")
    top_features = feature_sum.topk(10)
    for i, (idx, freq) in enumerate(zip(top_features.indices, top_features.values)):
        print(f"  Feature {idx.item()}: {freq.item():.4f}")
    
    # Sparsity per sample
    sparsity_per_sample = (all_features == 0).float().mean(dim=1)
    print(f"\nSparsity per sample - Mean: {sparsity_per_sample.mean():.4f}, Std: {sparsity_per_sample.std():.4f}")
    
    # Show examples for top features if validation loader is provided
    if val_loader is not None:
        print(f"\n=== Examples for Top 10 Features ===")
        show_feature_examples(all_features, top_features.indices, val_loader)
        # Also show examples for 3 random features
        num_features = all_features.shape[1]
        np.random.seed(12345)
        random_feature_indices = np.random.choice(num_features, 3, replace=False)
        print(f"\n=== Examples for 3 Random Features: {random_feature_indices.tolist()} ===")
        show_feature_examples(all_features, torch.tensor(random_feature_indices), val_loader)
    
    return {
        'feature_frequency': feature_frequency,
        'sparsity_per_sample': sparsity_per_sample
    }

def show_feature_examples(all_features, top_feature_indices, val_loader):
    """Show 10 random examples from validation dataset for each top feature"""
    
    # Convert to numpy for easier indexing
    features_np = all_features.numpy()
    
    for feature_idx in top_feature_indices:
        feature_idx = feature_idx.item()
        print(f"\n--- Feature {feature_idx} ---")
        
        # Find all samples where this feature was activated (value > 0)
        activated_samples = np.where(features_np[:, feature_idx] > 0)[0]
        
        if len(activated_samples) == 0:
            print("  No activations found for this feature")
            continue
            
        print(f"  Total activations: {len(activated_samples)}")
        
        # Calculate activation statistics
        activation_values = features_np[activated_samples, feature_idx]
        print(f"  Activation range: {activation_values.min():.4f} to {activation_values.max():.4f}")
        print(f"  Mean activation: {activation_values.mean():.4f}")
        
        # Sample 10 random examples (or all if less than 10)
        num_examples = min(10, len(activated_samples))
        if len(activated_samples) > 10:
            # Randomly sample 10 examples
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(activated_samples, num_examples, replace=False)
        else:
            sample_indices = activated_samples[:num_examples]
        
        # Get the actual examples from validation dataset
        val_examples = []
        val_labels = []
        
        # Reset the validation loader and iterate through to get examples
        val_loader_iter = iter(val_loader)
        sample_count = 0
        batch_start_idx = 0
        
        for batch_idx, (toks, labels) in enumerate(val_loader):
            batch_size = toks.size(0)
            batch_end_idx = batch_start_idx + batch_size
            
            # Check if any of our target samples are in this batch
            batch_sample_indices = []
            for sample_idx in sample_indices:
                if batch_start_idx <= sample_idx < batch_end_idx:
                    batch_sample_indices.append((sample_idx - batch_start_idx, sample_idx))
            
            # Extract examples from this batch
            for batch_sample_idx, global_sample_idx in batch_sample_indices:
                if sample_count < num_examples:
                    # Get the sequence and label
                    seq = toks[batch_sample_idx]
                    label = labels[batch_sample_idx]
                    
                    # Find the position of the query token (Q)
                    q_pos = (seq == Q).nonzero(as_tuple=False).squeeze()
                    if q_pos.numel() > 0:
                        q_pos = q_pos.item()
                        # Get the sequence up to the query
                        context_seq = seq[:q_pos]
                        # Remove padding tokens
                        context_seq = context_seq[context_seq != PAD]
                        
                        # Convert to readable format
                        readable_seq = []
                        for token in context_seq:
                            if token < E:
                                readable_seq.append(f"E{token.item()}")
                            elif token < E + T:
                                readable_seq.append(f"T{token.item() - E}")
                            elif token == SEP:
                                readable_seq.append("SEP")
                            else:
                                readable_seq.append(f"UNK{token.item()}")
                        
                        # Get the query components and answer
                        query_rel = seq[q_pos - 2].item()  # Relation type (Tq)
                        query_ent = seq[q_pos - 1].item()  # Entity (Eq)
                        answer = label[q_pos].item() if label[q_pos] != IGNORE_INDEX else "IGNORE"
                        
                        # Find the separator position after the correct entity
                        sep_pos = find_separator_after_correct_entity(seq, label)
                        
                        print(f"  Example {sample_count + 1}:")
                        print(f"    Context: {' '.join(readable_seq)}")
                        print(f"    Query: T{query_rel - E} E{query_ent} Q")
                        print(f"    Answer: E{answer}")
                        if sep_pos is not None:
                            print(f"    Separator after correct entity at position: {sep_pos}")
                        else:
                            print(f"    No separator found after correct entity")
                        print(f"    Feature value: {features_np[global_sample_idx, feature_idx]:.4f}")
                        
                        sample_count += 1
                        
                        if sample_count >= num_examples:
                            break
            
            if sample_count >= num_examples:
                break
                
            batch_start_idx = batch_end_idx

def main():
    """Main function to run SAE training and feature extraction"""
    
    try:
        # Configuration - single layer
        use_wandb = True
        
        # Get the global device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Main function using device: {device}")
        
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
        print("Created val_loader in main function")
        
        if device.type == 'cuda':
            print(f"GPU memory before main: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        # Load model
        model = build_model(N_LAYERS, HEADS)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Moving model to device: {device}")
        
        # Load pretrained weights
        REPO_ID = "sebastianhoenig/2L2H_Final"
        FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        pretrained_weights = torch.load(weights_path, map_location=device, weights_only=True)
        state_dict = pretrained_weights["model"]
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
        print(f"Model device: {next(model.parameters()).device}")
        # Verify that checkpoints were created or find existing ones
        print("\nVerifying checkpoint availability...")
        checkpoint_dir = f"./sae_checkpoints_{LAYER_TO_TRAIN}_hook_resid_post"
        if os.path.exists(f"{checkpoint_dir}/trainer_0/checkpoints/"):
            checkpoints = [f for f in os.listdir(f"{checkpoint_dir}/trainer_0/checkpoints/") if f.endswith('.pt')]
            print(f"Layer {LAYER_TO_TRAIN}: Found {len(checkpoints)} checkpoints: {checkpoints}")
        else:
            print(f"Layer {LAYER_TO_TRAIN}: No checkpoints directory found")

            print("Starting SAE training with pipeline...")
            print(f"Training SAE on layer {LAYER_TO_TRAIN}")
            
            # Train SAE
            run_ids, trainer_config = train_sae_with_pipeline(
                model=model,
                layer_to_train=LAYER_TO_TRAIN,
                sae_dim=SAE_DIM,
                use_wandb=use_wandb,
                checkpoint_dir=checkpoint_dir
            )
            
            if run_ids:
                print(f"Training completed! Run IDs: {run_ids}")
            else:
                print("Training completed but no run IDs returned")
        
        # Extract features for the trained SAE
        try:
            print(f"\nExtracting features for layer {LAYER_TO_TRAIN}...")
            
            # Look for available checkpoint
            model_checkpoint_dir = f"{checkpoint_dir}/trainer_0/checkpoints/"
            if os.path.exists(model_checkpoint_dir):
                checkpoints = [f for f in os.listdir(model_checkpoint_dir) if f.endswith('.pt')]
                if checkpoints:
                    # Use the latest checkpoint (highest step number)
                    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                    latest_checkpoint = checkpoints[-1]
                    sae_path = os.path.join(model_checkpoint_dir, latest_checkpoint)
                    print(f"Using checkpoint: {latest_checkpoint}")
                else:
                    print(f"Warning: No checkpoints found in {model_checkpoint_dir}, skipping feature extraction")
                    return
            else:
                print(f"Warning: Checkpoint directory not found: {model_checkpoint_dir}, skipping feature extraction")
                return
            
            # Extract features
            features_dict = extract_sae_features(LAYER_TO_TRAIN, sae_path, val_loader, device, model)
            
            if features_dict is not None:
                # Analyze features
                analysis = analyze_features(features_dict, val_loader)
                
                # Save features and analysis
                save_path = f"layer_{LAYER_TO_TRAIN}_sae_features_d{SAE_DIM}.pt"
                torch.save({
                    'features': features_dict['features'],
                    'reconstructions': features_dict['reconstructions'],
                    'labels': features_dict['labels'],
                    'analysis': analysis
                }, save_path)
                
                print(f"Features saved to {save_path}")
            else:
                print(f"Warning: Could not extract features for layer {LAYER_TO_TRAIN}")
            
            print("\n✅ SAE training and feature extraction completed!")
        except Exception as e:
            print(f"Warning: Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Verify CUDA availability and set device
        if device.type == 'cuda':
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            
            # Set default tensor type to CUDA
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            
            # Test GPU memory allocation
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"Test tensor allocated on GPU: {test_tensor.device}")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            
            # Force CUDA initialization and show process info
            torch.cuda.synchronize()
            print(f"CUDA initialized successfully")
            print(f"Process ID: {os.getpid()}")
            print(f"CUDA device properties: {torch.cuda.get_device_properties(0)}")
            
            # Set environment variables for better GPU visibility
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            print("Set CUDA environment variables")
        
        # Run main function
        main()
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()



