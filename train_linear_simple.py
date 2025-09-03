#!/usr/bin/env python3
"""
Simple linear model training on SAE features for entity binding task
(Minimal dependencies version)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import from the main file
from sae_features_d4096 import (
    SAE_DIM, LAYER_TO_TRAIN, E, T, Q, PAD, IGNORE_INDEX, SEP,
    build_model, collate_fn, val_dataset, train_dataset,
    N_LAYERS, HEADS, d_model, D_VOCAB
)
from dictionary_learning import AutoEncoder
from huggingface_hub import hf_hub_download

class LinearClassifier(nn.Module):
    """Simple linear classifier for SAE features"""
    
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def extract_sae_features_for_training(model, sae, data_loader, device, max_samples=None):
    """Extract SAE features from dataset for training/evaluation"""
    
    model.eval()
    sae.eval()
    
    all_features = []
    all_labels = []
    
    act_name = f"blocks.{LAYER_TO_TRAIN}.hook_resid_post"
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (toks, labels) in enumerate(tqdm(data_loader, desc="Extracting features")):
            if max_samples and sample_count >= max_samples:
                break
                
            toks = toks.to(device)
            labels = labels.to(device)
            # Get activations from the model at the last token position
            cache = model.run_with_cache(toks, names_filter=[act_name])[1]
            acts = cache[act_name]  # shape: [batch, seq, d_model]
            
            # For each sequence, find the Q token position and extract features there
            for i in range(toks.size(0)):
                seq = toks[i]
                label_seq = labels[i]
                # Find Q token position
                q_pos = (seq == Q).nonzero(as_tuple=False).squeeze()
                if q_pos.numel() > 0:
                    q_pos = q_pos.item()
                    if label_seq[q_pos] != IGNORE_INDEX:
                        # Extract activation at Q position
                        act_at_q = acts[i, q_pos, :]  # [d_model]
                        feature = sae.encode(act_at_q.unsqueeze(0)).squeeze(0)  # [sae_dim]
                        
                        all_features.append(feature.cpu())
                        all_labels.append(label_seq[q_pos].item())
                        sample_count += 1
                        
                        if max_samples and sample_count >= max_samples:
                            break
            
            if max_samples and sample_count >= max_samples:
                break
    
    if len(all_features) == 0:
        raise ValueError("No valid features extracted!")
        
    all_features = torch.stack(all_features)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    print(f"Extracted {len(all_features)} samples")
    print(f"Feature shape: {all_features.shape}")
    print(f"Label range: {all_labels.min().item()} to {all_labels.max().item()}")
    print(f"Unique labels: {len(torch.unique(all_labels))}")
    
    return all_features, all_labels

def train_linear_model(train_features, train_labels, val_features, val_labels, 
                      num_epochs=100, lr=0.001, device='cpu', batch_size=256, l1_lambda=0.0):
    """Train linear classifier on SAE features"""
    
    num_classes = E  # Number of entities to predict
    input_dim = train_features.shape[1]
    
    print(f"\nTraining linear classifier:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {len(train_features)}")
    print(f"  Validation samples: {len(val_features)}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  L1 regularization: {l1_lambda}")
    
    # Create model
    model = LinearClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move data to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        total_train_correct = 0
        total_train_samples = 0
        num_batches = 0
        
        # Create mini-batches for training
        for i in range(0, len(train_features), batch_size):
            end_i = min(i + batch_size, len(train_features))
            batch_features = train_features[i:end_i]
            batch_labels = train_labels[i:end_i]
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Add L1 regularization if specified
            if l1_lambda > 0:
                l1_penalty = torch.tensor(0., device=device)
                for param in model.parameters():
                    l1_penalty += torch.norm(param, 1)
                loss = loss + l1_lambda * l1_penalty
            
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy for this batch
            with torch.no_grad():
                train_predictions = torch.argmax(outputs, dim=1)
                total_train_correct += (train_predictions == batch_labels).sum().item()
                total_train_samples += batch_labels.size(0)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_acc = total_train_correct / total_train_samples
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features)
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_acc = (val_predictions == val_labels).float().mean().item()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Calculate final training accuracy with best model
    model.eval()
    with torch.no_grad():
        final_train_outputs = model(train_features)
        final_train_predictions = torch.argmax(final_train_outputs, dim=1)
        final_train_acc = (final_train_predictions == train_labels).float().mean().item()
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    
    return model, best_val_acc, final_train_acc

def evaluate_model(model, val_features, val_labels, device='cpu'):
    """Evaluate the trained model"""
    
    model.eval()
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    
    with torch.no_grad():
        outputs = model(val_features)
        predictions = torch.argmax(outputs, dim=1)
        
        # Calculate accuracy
        accuracy = (predictions == val_labels).float().mean().item()
        
        # Calculate per-class accuracy
        num_classes = E
        per_class_acc = torch.zeros(num_classes)
        per_class_count = torch.zeros(num_classes)
        
        for i in range(num_classes):
            mask = val_labels == i
            if mask.sum() > 0:
                per_class_acc[i] = (predictions[mask] == val_labels[mask]).float().mean()
                per_class_count[i] = mask.sum()
        
        # Calculate metrics
        valid_classes = per_class_count > 0
        macro_acc = per_class_acc[valid_classes].mean().item()
        
    return accuracy, macro_acc, per_class_acc, per_class_count

def main():
    """Main training and evaluation pipeline"""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading pretrained transformer model...")
    model = build_model(N_LAYERS, HEADS).to(device)
    
    # Load pretrained weights
    REPO_ID = "sebastianhoenig/2L2H_Final"
    FILENAME = "D256_L2_H2_attnOnly1_lr5.0e-04_wd0.01.pt"
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    pretrained_weights = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = pretrained_weights["model"]
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    
    # Load SAE
    print("\nLoading SAE...")
    sae = AutoEncoder(activation_dim=d_model, dict_size=SAE_DIM).to(device)
    sae_path = "sae_checkpoints_1_hook_resid_post/trainer_0/checkpoints/ae_15000.pt"
    
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE checkpoint not found at {sae_path}")
    
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    print("SAE loaded successfully.")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    
    # Extract training features (limit to reasonable size for training)
    print("\nExtracting training features...")
    
    # Get first item from the iterable dataset
    # first_item = next(iter(train_dataset))
    # print("train_dataset first item:", first_item)
    # print("  - tokens shape:", first_item[0].shape)
    # print("  - tokens:", first_item[0])
    # print("  - labels shape:", first_item[1].shape) 
    # print("  - labels:", first_item[1]) # 53
    # return
    train_features, train_labels = extract_sae_features_for_training(
        model, sae, train_loader, device, max_samples=50000  # Limit for memory
    )
    # Extract validation features
    print("\nExtracting validation features...")
    val_features, val_labels = extract_sae_features_for_training(
        model, sae, val_loader, device, max_samples=None  # Use all validation data
    )
    
    # Train linear model
    print("\n" + "="*50)
    print("TRAINING LINEAR CLASSIFIER")
    print("="*50)
    
    linear_model, best_val_acc, final_train_acc = train_linear_model(
        train_features, train_labels, val_features, val_labels,
        num_epochs=150, lr=0.001, device=device, batch_size=512, l1_lambda=0.001
    )
    
    # Evaluate model
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    accuracy, macro_acc, per_class_acc, per_class_count = evaluate_model(
        linear_model, val_features, val_labels, device
    )
    
    # Print results
    print(f"\nFinal Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Average Accuracy: {macro_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    
    # Show per-class statistics for first 20 classes
    print(f"\nPer-class accuracy (first 20 classes):")
    for i in range(min(20, E)):
        if per_class_count[i] > 0:
            print(f"  Entity {i:2d}: {per_class_acc[i]:.4f} ({per_class_count[i].int()} samples)")
        else:
            print(f"  Entity {i:2d}: No samples")
    
    # Save model and results
    save_dict = {
        'model_state_dict': linear_model.state_dict(),
        'best_val_acc': best_val_acc,
        'final_train_acc': final_train_acc,
        'final_accuracy': accuracy,
        'macro_accuracy': macro_acc,
        'per_class_acc': per_class_acc,
        'per_class_count': per_class_count,
        'hyperparameters': {
            'sae_dim': SAE_DIM,
            'layer': LAYER_TO_TRAIN,
            'num_classes': E,
            'lr': 0.001,
            'epochs': 150,
            'l1_lambda': 0.001,
            'train_samples': len(train_features),
            'val_samples': len(val_features)
        }
    }
    
    save_path = f"linear_model_sae_features_layer{LAYER_TO_TRAIN}_simple.pt"
    torch.save(save_dict, save_path)
    
    print(f"\nTraining completed!")
    print(f"Results saved to {save_path}")
    
    return linear_model, accuracy, macro_acc, final_train_acc

if __name__ == "__main__":
    main()
