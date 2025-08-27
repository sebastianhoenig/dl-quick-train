#!/usr/bin/env python3
"""
Example usage of the pipeline-based SAE training for HookedTransformer activations.

This script demonstrates how to:
1. Train multiple SAEs in parallel using run_pipeline
2. Enable Weights & Biases logging
3. Extract and analyze SAE features
4. Save trained models and features
"""

import torch
from sae_features_d4096 import (
    train_saes_with_pipeline,
    extract_sae_features,
    analyze_features,
    val_dataset,
    collate_fn
)
from torch.utils.data import DataLoader

def main():
    """Example of training SAEs and extracting features"""
    
    # Configuration
    layers_to_train = [0, 1]  # Train SAEs on both layers
    sae_dim = 4096  # SAE dictionary size
    use_wandb = True  # Enable Weights & Biases logging
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create validation data loader
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    
    print("=== Starting SAE Training with Pipeline ===")
    print(f"Training SAEs on layers: {layers_to_train}")
    print(f"SAE dimension: {sae_dim}")
    print(f"Weights & Biases logging: {use_wandb}")
    
    # Train SAEs using the pipeline
    run_ids, trainer_configs, model = train_saes_with_pipeline(
        layers_to_train=layers_to_train,
        sae_dim=sae_dim,
        use_wandb=use_wandb
    )
    
    print(f"\n✅ Training completed!")
    print(f"Weights & Biases run IDs: {run_ids}")
    
    # Extract features for each trained SAE
    print("\n=== Extracting SAE Features ===")
    
    for i, layer in enumerate(layers_to_train):
        print(f"\nProcessing layer {layer}...")
        
        # Path to saved SAE checkpoint
        sae_path = f"./sae_checkpoints/trainer_{i}/checkpoints/ae_10000000.pt"
        
        # Extract features
        features_dict = extract_sae_features(layer, sae_path, val_loader, device, model)
        
        if features_dict is not None:
            # Analyze features
            print(f"Analyzing features for layer {layer}...")
            analysis = analyze_features(features_dict)
            
            # Save features and analysis
            save_path = f"layer_{layer}_sae_features_d{sae_dim}.pt"
            torch.save({
                'features': features_dict['features'],
                'activations': features_dict['activations'],
                'reconstructions': features_dict['reconstructions'],
                'labels': features_dict['labels'],
                'analysis': analysis,
                'layer': layer,
                'sae_dim': sae_dim
            }, save_path)
            
            print(f"✅ Features for layer {layer} saved to {save_path}")
            
            # Print some statistics
            features = features_dict['features']
            print(f"Feature shape: {features.shape}")
            print(f"Feature sparsity: {(features == 0).float().mean():.4f}")
        else:
            print(f"❌ Failed to extract features for layer {layer}")
    
    print("\n🎉 All done! SAE training and feature extraction completed successfully.")
    print("\nNext steps:")
    print("1. Check your Weights & Biases dashboard for training logs")
    print("2. Use the saved feature files for interpretability analysis")
    print("3. Load trained SAEs for inference or further analysis")

if __name__ == "__main__":
    main()
