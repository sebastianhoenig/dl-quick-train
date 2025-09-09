#!/usr/bin/env python3
"""
Advanced entity analysis using token-level features for Layer 0 and final token features for Layer 1.
This script properly associates features with entities by analyzing which tokens correspond to which entities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
from typing import Dict, List, Tuple

# Constants
E = 100  # num entities
T = 10   # num types/relations
SEP = E + T
Q = E + T + 1
PAD = E + T + 2
IGNORE_INDEX = -100

def load_features():
    """Load both layer 0 and layer 1 SAE features"""
    print("Loading features...")
    
    # Load Layer 0 SAE features
    try:
        layer_0_data = torch.load("layer_0_sae_features_d4096.pt", map_location='cpu', weights_only=False)
        print(f"✓ Loaded Layer 0 SAE features: {layer_0_data['features'].shape}")
        print(f"  Labels shape: {layer_0_data['labels'].shape}")
    except FileNotFoundError:
        print("❌ Layer 0 SAE features not found")
        layer_0_data = None
    
    # Load Layer 1 SAE features
    try:
        layer_1_data = torch.load("layer_1_sae_features_d4096.pt", map_location='cpu', weights_only=False)
        print(f"✓ Loaded Layer 1 SAE features: {layer_1_data['features'].shape}")
        print(f"  Labels shape: {layer_1_data['labels'].shape}")
    except FileNotFoundError:
        print("❌ Layer 1 SAE features not found")
        layer_1_data = None
    
    return layer_0_data, layer_1_data

def identify_discriminative_features(layer_0_data, layer_1_data, max_entity_percentage=0.2):
    """
    Identify features that are discriminative by filtering out features that activate 
    on more than max_entity_percentage of entities.
    
    Args:
        layer_0_data: Layer 0 feature data
        layer_1_data: Layer 1 feature data  
        max_entity_percentage: Maximum percentage of entities a feature can activate on (default 0.2 = 20%)
    
    Returns:
        dict with discriminative feature sets for each layer
    """
    print(f"\n🔍 Identifying discriminative features (max {max_entity_percentage*100:.0f}% entity coverage)...")
    
    discriminative_features = {'layer_0': set(), 'layer_1': set()}
    
    # Extract label entities from labels tensor
    def extract_label_entities(labels_tensor):
        """Extract label entities from labels tensor"""
        label_entities = []
        for labels in labels_tensor:
            label_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
            if len(label_positions) > 0:
                label_entities.append(labels[label_positions[0]].item())
            else:
                label_entities.append(-1)  # Invalid label
        return label_entities
    
    # Get all unique entities from the data
    all_entities = set()
    if layer_0_data:
        label_entities = extract_label_entities(layer_0_data['labels'])
        all_entities.update(label_entities)
    elif layer_1_data:
        label_entities = extract_label_entities(layer_1_data['labels'])
        all_entities.update(label_entities)
    
    # Remove invalid entities
    all_entities = {e for e in all_entities if e != -1 and e != IGNORE_INDEX}
    total_entities = len(all_entities)
    max_entities_for_feature = int(total_entities * max_entity_percentage)
    
    print(f"  Total unique entities: {total_entities}")
    print(f"  Max entities per feature: {max_entities_for_feature}")
    
    # Analyze Layer 0 features
    if layer_0_data:
        print(f"\n  📊 Analyzing Layer 0 features...")
        feature_to_entities = defaultdict(set)
        
        num_examples = len(layer_0_data['features'])
        label_entities = extract_label_entities(layer_0_data['labels'])
        
        for example_idx in range(num_examples):
            label_entity = label_entities[example_idx]
            
            if label_entity == -1 or label_entity == IGNORE_INDEX:
                continue
            
            # Get features for this example
            features = layer_0_data['features'][example_idx]  # [4096]
            active_indices = torch.nonzero(features > 0).squeeze(-1)
            
            # Map features to entities
            for feature_idx in active_indices.tolist():
                feature_to_entities[feature_idx].add(label_entity)
        
        # Filter discriminative features
        total_features = len(feature_to_entities)
        discriminative_count = 0
        for feature_idx, entities in feature_to_entities.items():
            if len(entities) <= max_entities_for_feature:
                discriminative_features['layer_0'].add(feature_idx)
                discriminative_count += 1
        
        print(f"    Total features analyzed: {total_features}")
        print(f"    Discriminative features: {discriminative_count} ({discriminative_count/total_features*100:.1f}%)")
    
    # Analyze Layer 1 features
    if layer_1_data:
        print(f"\n  📊 Analyzing Layer 1 features...")
        feature_to_entities = defaultdict(set)
        
        num_examples = len(layer_1_data['features'])
        label_entities = extract_label_entities(layer_1_data['labels'])
        
        for example_idx in range(num_examples):
            label_entity = label_entities[example_idx]
            
            if label_entity == -1 or label_entity == IGNORE_INDEX:
                continue
            
            # Get features for this example
            features = layer_1_data['features'][example_idx]  # [4096]
            active_indices = torch.nonzero(features > 0).squeeze(-1)
            
            # Map features to entities
            for feature_idx in active_indices.tolist():
                feature_to_entities[feature_idx].add(label_entity)
        
        # Filter discriminative features
        total_features = len(feature_to_entities)
        discriminative_count = 0
        for feature_idx, entities in feature_to_entities.items():
            if len(entities) <= max_entities_for_feature:
                discriminative_features['layer_1'].add(feature_idx)
                discriminative_count += 1
        
        print(f"    Total features analyzed: {total_features}")
        print(f"    Discriminative features: {discriminative_count} ({discriminative_count/total_features*100:.1f}%)")
    
    return discriminative_features

def parse_sequence_with_positions(sequence: torch.Tensor) -> Dict:
    """Parse a sequence and return entity positions and relationships"""
    seq_list = sequence.tolist()
    
    # Find Q position
    try:
        q_pos = seq_list.index(Q)
    except ValueError:
        return None
    
    # Parse the query
    query_relation = seq_list[q_pos - 2]
    query_entity = seq_list[q_pos - 1]
    
    # Parse facts from the context (before the query)
    context = seq_list[:q_pos-2]
    
    # Parse facts and track entity positions
    facts = []
    entity_positions = defaultdict(list)  # entity_id -> list of token positions
    
    i = 0
    while i + 3 < len(context):
        if context[i + 3] == SEP:
            e1, rel, e2 = context[i], context[i + 1], context[i + 2]
            facts.append((e1, rel, e2))
            
            # Record positions of entities
            entity_positions[e1].append(i)      # head entity position
            entity_positions[e2].append(i + 2) # tail entity position
            
            i += 4
        else:
            i += 1
    
    # Add query entity position
    entity_positions[query_entity].append(q_pos - 1)
    
    # Collect all entities mentioned
    all_entities = set()
    for e1, rel, e2 in facts:
        all_entities.add(e1)
        all_entities.add(e2)
    all_entities.add(query_entity)
    
    return {
        'facts': facts,
        'query_entity': query_entity,
        'query_relation': query_relation,
        'query_position': q_pos - 1,
        'entity_positions': dict(entity_positions),
        'all_entities': all_entities,
        'sequence_length': len(seq_list)
    }

def analyze_entity_token_features(layer_0_data, layer_1_data, target_entities: List[int], discriminative_features: Dict = None) -> Dict:
    """Analyze SAE features for specific label entities"""
    print(f"\n🎯 Analyzing SAE features for label entities: {target_entities}")
    
    results = defaultdict(lambda: {'layer_0': [], 'layer_1': []})
    
    if layer_0_data is None and layer_1_data is None:
        print("❌ No feature data available")
        return {}
    
    # Extract label entities from labels tensor
    def extract_label_entities(labels_tensor):
        """Extract label entities from labels tensor"""
        label_entities = []
        for labels in labels_tensor:
            label_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
            if len(label_positions) > 0:
                label_entities.append(labels[label_positions[0]].item())
            else:
                label_entities.append(-1)  # Invalid label
        return label_entities
    
    num_examples = len(layer_0_data['features']) if layer_0_data else len(layer_1_data['features'])
    if layer_0_data:
        label_entities = extract_label_entities(layer_0_data['labels'])
    else: # layer_1_data:
        label_entities = extract_label_entities(layer_1_data['labels'])
    
    for example_idx in range(num_examples):
        # Get the label entity for this example
        if layer_0_data:
            label_entity = label_entities[example_idx]
        else: # layer_1_data:
            label_entity = label_entities[example_idx]
        
        # Skip if this label entity is not in our target list
        if label_entity not in target_entities:
            continue
        
        # Process Layer 0 SAE features (final features, not token-level)
        if layer_0_data:
            features = layer_0_data['features'][example_idx]  # [4096]
            active_features = torch.nonzero(features > 0).squeeze(-1)
            
            if len(active_features) > 0:
                # Filter to only discriminative features if provided
                if discriminative_features and 'layer_0' in discriminative_features:
                    discriminative_mask = torch.tensor([f in discriminative_features['layer_0'] for f in active_features.tolist()])
                    if discriminative_mask.any():
                        active_features = active_features[discriminative_mask]
                        feature_values = features[active_features]
                    else:
                        continue  # Skip this example if no discriminative features
                else:
                    feature_values = features[active_features]
                
                if len(active_features) > 0:
                    sorted_indices = torch.argsort(feature_values, descending=True)
                    
                    results[label_entity]['layer_0'].append({
                        'example_idx': example_idx,
                        'label_entity': label_entity,
                        'top_features': active_features[sorted_indices].tolist(),
                        'top_values': feature_values[sorted_indices].tolist()
                    })
        
        # Process Layer 1 SAE features (final features)
        if layer_1_data:
            features = layer_1_data['features'][example_idx]  # [4096]
            active_features = torch.nonzero(features > 0).squeeze(-1)
            
            if len(active_features) > 0:
                # Filter to only discriminative features if provided
                if discriminative_features and 'layer_1' in discriminative_features:
                    discriminative_mask = torch.tensor([f in discriminative_features['layer_1'] for f in active_features.tolist()])
                    if discriminative_mask.any():
                        active_features = active_features[discriminative_mask]
                        feature_values = features[active_features]
                    else:
                        continue  # Skip this example if no discriminative features
                else:
                    feature_values = features[active_features]
                
                if len(active_features) > 0:
                    sorted_indices = torch.argsort(feature_values, descending=True)
                    
                    results[label_entity]['layer_1'].append({
                        'example_idx': example_idx,
                        'label_entity': label_entity,
                        'top_features': active_features[sorted_indices].tolist(),
                        'top_values': feature_values[sorted_indices].tolist()
                    })
    
    return dict(results)

def plot_entity_token_analysis(entity_results: Dict, save_path: str = "entity_sae_analysis_filtered.png"):
    """Create visualization for entity SAE analysis"""
    print("\n📊 Creating entity SAE analysis plots...")
    
    num_entities = len(entity_results)
    if num_entities == 0:
        print("❌ No entity results to plot")
        return
    
    fig, axes = plt.subplots(num_entities, 2, figsize=(16, 5 * num_entities))
    if num_entities == 1:
        axes = axes.reshape(1, -1)
    
    for i, entity in enumerate(sorted(entity_results.keys())):
        data = entity_results[entity]
        # Layer 0 - SAE features
        ax0 = axes[i, 0]
        
        if data['layer_0']:
            # Collect features from all examples for this label entity
            all_features = []
            for example in data['layer_0']:
                top_n = min(10, len(example['top_features']))
                all_features.extend(example['top_features'][:top_n])
            
            if all_features:
                feature_counts = Counter(all_features)
                top_features = feature_counts.most_common(15)
                
                if top_features:
                    features, counts = zip(*top_features)
                    bars = ax0.bar(range(len(features)), counts, alpha=0.7, color='skyblue')
                    ax0.set_title(f'Label Entity {entity} - Layer 0 SAE Features\n'
                                f'({len(data["layer_0"])} examples)')
                    ax0.set_xlabel('Feature Index (Rank)')
                    ax0.set_ylabel('Activation Frequency')
                    ax0.set_xticks(range(len(features)))
                    ax0.set_xticklabels([str(f) for f in features], rotation=45, fontsize=8)
                    
                    # Add value labels for total counts
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax0.text(bar.get_x() + bar.get_width()/2., height,
                               f'{count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax0.text(0.5, 0.5, 'No significant features', 
                           ha='center', va='center', transform=ax0.transAxes)
                    ax0.set_title(f'Label Entity {entity} - Layer 0 SAE Features')
            else:
                ax0.text(0.5, 0.5, 'No active features found', 
                       ha='center', va='center', transform=ax0.transAxes)
                ax0.set_title(f'Label Entity {entity} - Layer 0 SAE Features')
        else:
            ax0.text(0.5, 0.5, 'No Layer 0 data', 
                   ha='center', va='center', transform=ax0.transAxes)
            ax0.set_title(f'Label Entity {entity} - Layer 0 SAE Features')
        
        # Layer 1 - SAE features
        ax1 = axes[i, 1]
        
        if data['layer_1']:
            all_features = []
            for example in data['layer_1']:
                top_n = min(10, len(example['top_features']))
                all_features.extend(example['top_features'][:top_n])
            
            if all_features:
                feature_counts = Counter(all_features)
                top_features = feature_counts.most_common(15)
                
                if top_features:
                    features, counts = zip(*top_features)
                    bars = ax1.bar(range(len(features)), counts, alpha=0.7, color='orange')
                    ax1.set_title(f'Entity {entity} - Layer 1 SAE Features\n'
                                f'({len(data["layer_1"])} examples)')
                    ax1.set_xlabel('Feature Index (Rank)')
                    ax1.set_ylabel('Activation Frequency')
                    ax1.set_xticks(range(len(features)))
                    ax1.set_xticklabels([str(f) for f in features], rotation=45, fontsize=8)
                    
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                               f'{count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax1.text(0.5, 0.5, 'No significant features', 
                           ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title(f'Entity {entity} - Layer 1 SAE Features')
            else:
                ax1.text(0.5, 0.5, 'No active features found', 
                       ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Entity {entity} - Layer 1 SAE Features')
        else:
            ax1.text(0.5, 0.5, 'No Layer 1 data', 
                   ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'Entity {entity} - Layer 1 SAE Features')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {save_path}")

def print_detailed_token_analysis(entity_results: Dict):
    """Print detailed analysis of entity SAE features"""
    print("\n" + "="*80)
    print("DETAILED ENTITY SAE ANALYSIS")
    print("="*80)
    
    for entity in sorted(entity_results.keys()):
        data = entity_results[entity]
        print(f"\n🔍 ENTITY {entity}")
        print("-" * 50)
        
        # Layer 0 analysis
        if data['layer_0']:
            print(f"\n  📊 Layer 0 (SAE features for label entity {entity}):")
            print(f"    Found in {len(data['layer_0'])} validation examples")
            
            # Aggregate features across all examples
            all_features = []
            for example in data['layer_0']:
                all_features.extend(example['top_features'][:5])
            
            if all_features:
                feature_counts = Counter(all_features)
                top_features = feature_counts.most_common(5)
                
                print(f"    Top 5 most activated features:")
                for feature_idx, count in top_features:
                    percentage = count / len(all_features) * 100
                    print(f"      Feature {feature_idx}: {count} activations ({percentage:.1f}%)")
            
            # Show example
            if data['layer_0']:
                example = data['layer_0'][0]
                print(f"    Example features (top 5):")
                for i, (feat_idx, value) in enumerate(zip(example['top_features'][:5], example['top_values'][:5])):
                    print(f"      Feature {feat_idx}: {value:.4f}")
        else:
            print(f"\n  📊 Layer 0: No data available")
        
        # Layer 1 analysis
        if data['layer_1']:
            print(f"\n  📊 Layer 1 (SAE features for label entity {entity}):")
            examples = data['layer_1']
            print(f"    Found in {len(examples)} validation examples")
            
            all_features = []
            for example in examples:
                all_features.extend(example['top_features'][:5])
            
            feature_counts = Counter(all_features)
            top_features = feature_counts.most_common(5)
            
            print(f"    Top 5 most activated features:")
            for feature_idx, count in top_features:
                percentage = count / len(all_features) * 100
                print(f"      Feature {feature_idx}: {count} activations ({percentage:.1f}%)")
            
            # Show example
            if data['layer_1']:
                example = data['layer_1'][0]
                print(f"    Example features (top 5):")
                for i, (feat_idx, value) in enumerate(zip(example['top_features'][:5], example['top_values'][:5])):
                    print(f"      Feature {feat_idx}: {value:.4f}")
        else:
            print(f"\n  📊 Layer 1: No data available")

def main():
    """Main function"""
    print("🚀 Advanced Entity SAE Feature Analysis (Discriminative Features)")
    print("=" * 70)
    print("This analysis extracts DISCRIMINATIVE SAE features (≤20% entity coverage)")
    print("for both Layer 0 and Layer 1, organized by LABEL ENTITY (the target answer)")
    print("=" * 70)
    
    # Load features
    layer_0_data, layer_1_data = load_features()
    
    if layer_0_data is None and layer_1_data is None:
        print("❌ No feature data available. Please run the SAE training script first.")
        return
    
    # Identify discriminative features (filter out features that activate on >20% of entities)
    discriminative_features = identify_discriminative_features(layer_0_data, layer_1_data, max_entity_percentage=0.2)
    
    # Show data structure info
    if layer_0_data:
        print(f"\n📋 Data Structure:")
        print(f"  ✓ Layer 0 features: {layer_0_data['features'].shape}")
        print(f"  ✓ Layer 0 labels: {layer_0_data['labels'].shape}")
    
    if layer_1_data:
        print(f"  ✓ Layer 1 features: {layer_1_data['features'].shape}")
        print(f"  ✓ Layer 1 labels: {layer_1_data['labels'].shape}")
    
    # Extract label entities to find available entities
    def extract_label_entities(labels_tensor):
        """Extract label entities from labels tensor"""
        label_entities = []
        for labels in labels_tensor:
            label_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
            if len(label_positions) > 0:
                label_entities.append(labels[label_positions[0]].item())
            else:
                label_entities.append(-1)  # Invalid label
        return label_entities
    
    # Select entities 0 to 9 that appear as labels
    if layer_0_data:
        label_entities = extract_label_entities(layer_0_data['labels'])
        available_entities = list(set(label_entities))
        available_entities = [e for e in available_entities if e != -1 and e != IGNORE_INDEX]  # Remove invalid labels
    elif layer_1_data:
        label_entities = extract_label_entities(layer_1_data['labels'])
        available_entities = list(set(label_entities))
        available_entities = [e for e in available_entities if e != -1 and e != IGNORE_INDEX]  # Remove invalid labels
    else:
        available_entities = []
    
    # Select entities 0 to 9 (if they exist in the data)
    target_entities = [e for e in range(10) if e in available_entities]
    
    print(f"\n🎯 Analyzing label entities: {sorted(target_entities)}")
    print("For each entity, we analyze SAE features from sequences")
    print("where that entity is the target answer (label)")
    print(f"Selected entities 0-9 that appear in the data: {len(target_entities)} entities found")
    
    # Analyze features (using only discriminative features)
    entity_results = analyze_entity_token_features(layer_0_data, layer_1_data, target_entities, discriminative_features)
    
    if not entity_results:
        print("❌ No entities found in the data")
        return
    
    # Print detailed analysis
    print_detailed_token_analysis(entity_results)
    
    # Create plots
    plot_entity_token_analysis(entity_results, "entity_sae_analysis_filtered.png")
    
    print("\n✅ Advanced entity SAE analysis complete!")
    print("This shows how DISCRIMINATIVE SAE features (≤20% entity coverage) relate to specific label entities.")

if __name__ == "__main__":
    main()
