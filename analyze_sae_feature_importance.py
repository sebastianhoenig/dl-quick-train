#!/usr/bin/env python3
"""
Analyze which SAE features drive which entity predictions in the trained linear model.
Also find top-activating examples for selected features.
regularization: entity needs to be predicted based on few features
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from collections import defaultdict

# Import from the main files
from sae_features_d4096 import (
    SAE_DIM, LAYER_TO_TRAIN, E, T, Q, PAD, IGNORE_INDEX, SEP,
    build_model, collate_fn, val_dataset, train_dataset,
    N_LAYERS, HEADS, d_model, D_VOCAB, produce_example_by_index
)
from dictionary_learning import AutoEncoder
from huggingface_hub import hf_hub_download
from train_linear_simple import LinearClassifier

def load_trained_model(device='cpu'):
    """Load the trained linear model and associated components"""
    
    # Load the saved linear model
    model_path = "linear_model_sae_features_layer1_simple.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    saved_data = torch.load(model_path, map_location=device)
    
    # Create and load linear model
    linear_model = LinearClassifier(SAE_DIM, E).to(device)
    linear_model.load_state_dict(saved_data['model_state_dict'])
    linear_model.eval()
    
    print("Linear model loaded successfully")
    print(f"Model accuracy: {saved_data['final_accuracy']:.4f}")
    print(f"Hyperparameters: {saved_data['hyperparameters']}")
    
    return linear_model, saved_data

def load_transformer_and_sae(device='cpu'):
    """Load the transformer model and SAE"""
    
    # Load transformer model
    print("Loading pretrained transformer model...")
    model = build_model(N_LAYERS, HEADS).to(device)
    
    # Load pretrained weights
    REPO_ID = "sebastianhoenig/2L2H_Final"
    FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    pretrained_weights = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = pretrained_weights["model"]
    model.load_state_dict(state_dict)
    model.eval()
    print("Transformer model loaded successfully.")
    
    # Load SAE
    print("Loading SAE...")
    sae = AutoEncoder(activation_dim=d_model, dict_size=SAE_DIM).to(device)
    import glob
    import re
    checkpoint_dir = "sae_checkpoints_1_hook_resid_post/trainer_0/checkpoints/"
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ae_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No SAE checkpoints found in {checkpoint_dir}")
    def extract_step_num(path):
        match = re.search(r"ae_(\d+)\.pt", os.path.basename(path))
        return int(match.group(1)) if match else -1
    sae_path = max(checkpoint_files, key=extract_step_num)
    print(f"Using SAE checkpoint: {sae_path}")    
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE checkpoint not found at {sae_path}")
    
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()
    print("SAE loaded successfully.")
    
    return model, sae

def analyze_feature_importance(linear_model):
    """Analyze which features are most important for each entity prediction"""
    
    # Get the linear layer weights
    # Shape: [num_entities, sae_dim]
    weights = linear_model.linear.weight.data.cpu().numpy()  # [E, SAE_DIM]
    bias = linear_model.linear.bias.data.cpu().numpy()       # [E]
    
    print(f"Weight matrix shape: {weights.shape}")
    print(f"Bias shape: {bias.shape}")
    
    # For each entity, find the most important features (highest absolute weights)
    entity_top_features = {}
    feature_importance_stats = {}
    
    for entity in range(E):
        entity_weights = weights[entity]  # [SAE_DIM]
        
        # Get top positive and negative features
        top_positive_idx = np.argsort(entity_weights)[-10:][::-1]  # Top 10 positive
        top_negative_idx = np.argsort(entity_weights)[:10]         # Top 10 negative
        
        entity_top_features[entity] = {
            'positive': [(idx, entity_weights[idx]) for idx in top_positive_idx],
            'negative': [(idx, entity_weights[idx]) for idx in top_negative_idx],
            'max_pos_weight': entity_weights.max(),
            'max_neg_weight': entity_weights.min(),
            'weight_std': entity_weights.std()
        }
    
    # Find which features are most discriminative across all entities
    feature_max_weights = np.abs(weights).max(axis=0)  # [SAE_DIM]
    feature_variance = np.var(weights, axis=0)         # [SAE_DIM]
    
    most_discriminative_features = np.argsort(feature_max_weights)[-20:][::-1]
    highest_variance_features = np.argsort(feature_variance)[-20:][::-1]
    
    feature_importance_stats = {
        'most_discriminative': [(idx, feature_max_weights[idx]) for idx in most_discriminative_features],
        'highest_variance': [(idx, feature_variance[idx]) for idx in highest_variance_features],
        'weights_matrix': weights,
        'bias_vector': bias
    }
    
    return entity_top_features, feature_importance_stats

def print_feature_analysis(entity_top_features, feature_importance_stats, num_entities_to_show=10):
    """Print analysis results"""
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Show most discriminative features overall
    print(f"\nMost discriminative features across all entities:")
    for i, (feature_idx, max_weight) in enumerate(feature_importance_stats['most_discriminative'][:10]):
        print(f"  Feature {feature_idx:4d}: max_abs_weight = {max_weight:.4f}")
    
    print(f"\nFeatures with highest variance across entities:")
    for i, (feature_idx, variance) in enumerate(feature_importance_stats['highest_variance'][:10]):
        print(f"  Feature {feature_idx:4d}: variance = {variance:.4f}")
    
    # Show top features for specific entities
    print(f"\nTop features for first {num_entities_to_show} entities:")
    for entity in range(min(num_entities_to_show, E)):
        features = entity_top_features[entity]
        print(f"\nEntity {entity}:")
        print(f"  Weight statistics: max_pos={features['max_pos_weight']:.4f}, "
              f"max_neg={features['max_neg_weight']:.4f}, std={features['weight_std']:.4f}")
        
        print(f"  Top positive features:")
        for feature_idx, weight in features['positive'][:5]:
            print(f"    Feature {feature_idx:4d}: {weight:.4f}")
        
        print(f"  Top negative features:")
        for feature_idx, weight in features['negative'][:5]:
            print(f"    Feature {feature_idx:4d}: {weight:.4f}")

def extract_activations_for_features(transformer_model, sae, data_loader, device, 
                                   target_features, max_samples=5000):
    """Extract SAE activations for specific features across dataset"""
    
    transformer_model.eval()
    sae.eval()
    
    act_name = f"blocks.{LAYER_TO_TRAIN}.hook_resid_post"
    
    # Store activations and metadata for each target feature
    feature_activations = {feat_idx: [] for feat_idx in target_features}
    sample_metadata = []
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (toks, labels) in enumerate(tqdm(data_loader, desc="Extracting activations")):
            if max_samples and sample_count >= max_samples:
                break
                
            toks = toks.to(device)
            labels = labels.to(device)
            
            # Get activations from the model
            cache = transformer_model.run_with_cache(toks, names_filter=[act_name])[1]
            acts = cache[act_name]  # shape: [batch, seq, d_model]
            
            # For each sequence, find the Q token position and extract features there
            for i in range(toks.size(0)):
                if max_samples and sample_count >= max_samples:
                    break
                    
                seq = toks[i]
                label_seq = labels[i]
                
                # Find Q token position
                q_pos = (seq == Q).nonzero(as_tuple=False).squeeze()
                if q_pos.numel() > 0:
                    q_pos = q_pos.item()
                    if label_seq[q_pos] != IGNORE_INDEX:
                        # Extract activation at Q position
                        act_at_q = acts[i, q_pos, :]  # [d_model]
                        sae_features = sae.encode(act_at_q.unsqueeze(0)).squeeze(0)  # [sae_dim]
                        
                        # Store activations for target features
                        for feat_idx in target_features:
                            feature_activations[feat_idx].append(sae_features[feat_idx].item())
                        
                        # Store metadata about this sample
                        sample_metadata.append({
                            'sample_idx': sample_count,
                            'tokens': seq.cpu().numpy(),
                            'label': label_seq[q_pos].item(),
                            'q_position': q_pos
                        })
                        
                        sample_count += 1
    
    print(f"Extracted activations for {sample_count} samples")
    return feature_activations, sample_metadata

def find_top_activating_examples(feature_activations, sample_metadata, target_features, top_k=10):
    """Find examples with highest activations for each target feature"""
    
    top_examples = {}
    
    for feat_idx in target_features:
        activations = np.array(feature_activations[feat_idx])
        
        # Get indices of top-k activating examples
        top_indices = np.argsort(activations)[-top_k:][::-1]
        
        top_examples[feat_idx] = []
        for idx in top_indices:
            metadata = sample_metadata[idx]
            activation_value = activations[idx]
            
            top_examples[feat_idx].append({
                'activation': activation_value,
                'metadata': metadata
            })
    
    return top_examples

def tokens_to_readable_string(tokens):
    """Convert token sequence to readable string representation"""
    
    readable_parts = []
    
    for token in tokens:
        if token == PAD:
            readable_parts.append("[PAD]")
        elif token == SEP:
            readable_parts.append(" | ")
        elif token == Q:
            readable_parts.append(" ? ")
        elif 0 <= token < E:
            readable_parts.append(f"E{token}")
        elif E <= token < E + T:
            readable_parts.append(f"R{token-E}")
        else:
            readable_parts.append(f"[{token}]")
    
    return " ".join(readable_parts)

def print_top_activating_examples(top_examples, target_features):
    """Print the top activating examples in readable format"""
    
    print("\n" + "="*80)
    print("TOP ACTIVATING EXAMPLES")
    print("="*80)
    
    for feat_idx in target_features:
        print(f"\nFeature {feat_idx} - Top 5 activating examples:")
        print("-" * 60)
        
        for i, example in enumerate(top_examples[feat_idx][:5]):
            activation = example['activation']
            metadata = example['metadata']
            
            tokens = metadata['tokens']
            label = metadata['label']
            q_pos = metadata['q_position']
            
            readable_seq = tokens_to_readable_string(tokens)
            
            print(f"  {i+1}. Activation: {activation:.4f}")
            print(f"     Label (answer): E{label}")
            print(f"     Sequence: {readable_seq}")
            print(f"     Q position: {q_pos}")
            print()

def create_feature_heatmap(feature_importance_stats, entity_top_features, save_path=None):
    """Create a heatmap showing feature importance for entities"""
    
    weights = feature_importance_stats['weights_matrix']  # [E, SAE_DIM]
    
    # Select top discriminative features and first 20 entities for visualization
    top_features = [idx for idx, _ in feature_importance_stats['most_discriminative'][:50]]
    entities_to_show = min(20, E)
    
    # Create subset matrix
    subset_weights = weights[:entities_to_show, top_features]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(subset_weights, 
                xticklabels=[f"F{i}" for i in top_features],
                yticklabels=[f"E{i}" for i in range(entities_to_show)],
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Weight Value'})
    
    plt.title('Feature Importance Heatmap\n(Top 50 Discriminative Features vs First 20 Entities)')
    plt.xlabel('SAE Features')
    plt.ylabel('Entities')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
    
    plt.show()

def main():
    """Main analysis pipeline"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained linear model
    print("\nLoading trained linear model...")
    linear_model, saved_data = load_trained_model(device)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    entity_top_features, feature_importance_stats = analyze_feature_importance(linear_model)
    
    # Print analysis results
    print_feature_analysis(entity_top_features, feature_importance_stats)
    
    # Create heatmap
    print("\nCreating feature importance heatmap...")
    create_feature_heatmap(feature_importance_stats, entity_top_features, 
                          save_path="feature_importance_heatmap.png")
    
    # Select a few interesting features for detailed analysis
    # Choose most discriminative features
    top_discriminative = [idx for idx, _ in feature_importance_stats['most_discriminative'][:5]]
    print(f"\nSelected features for detailed analysis: {top_discriminative}")
    
    # Load transformer and SAE for activation extraction
    print("\nLoading transformer model and SAE...")
    transformer_model, sae = load_transformer_and_sae(device)
    
    # Create data loader for analysis
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
    
    # Extract activations for selected features
    print("\nExtracting activations for selected features...")
    feature_activations, sample_metadata = extract_activations_for_features(
        transformer_model, sae, val_loader, device, top_discriminative, max_samples=3000
    )
    
    # Find top activating examples
    print("\nFinding top activating examples...")
    top_examples = find_top_activating_examples(
        feature_activations, sample_metadata, top_discriminative, top_k=10
    )
    
    # Print top activating examples
    print_top_activating_examples(top_examples, top_discriminative)
    
    # Additional analysis: show which entities each top feature predicts
    print("\n" + "="*80)
    print("FEATURE-TO-ENTITY PREDICTION ANALYSIS")
    print("="*80)
    
    weights = feature_importance_stats['weights_matrix']
    for feat_idx in top_discriminative:
        feat_weights = weights[:, feat_idx]  # Weights for this feature across all entities
        
        # Find entities with highest positive and negative weights for this feature
        top_pos_entities = np.argsort(feat_weights)[-5:][::-1]
        top_neg_entities = np.argsort(feat_weights)[:5]
        
        print(f"\nFeature {feat_idx}:")
        print(f"  Most positively predicted entities:")
        for ent in top_pos_entities:
            print(f"    Entity {ent}: weight = {feat_weights[ent]:.4f}")
        print(f"  Most negatively predicted entities:")
        for ent in top_neg_entities:
            print(f"    Entity {ent}: weight = {feat_weights[ent]:.4f}")

if __name__ == "__main__":
    main()
