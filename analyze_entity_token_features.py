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
    """Load both layer 0 token features and layer 1 final token features"""
    print("Loading features...")
    
    # Load Layer 0 token-level features
    try:
        layer_0_data = torch.load("layer_0_token_features_d4096.pt", map_location='cpu', weights_only=False)
        print(f"✓ Loaded Layer 0 token features: {len(layer_0_data['token_features'])} examples")
        print(f"  Average sequence length: {layer_0_data['metadata']['avg_seq_len']:.1f} tokens")
    except FileNotFoundError:
        print("❌ Layer 0 token features not found. Run extract_layer0_token_features.py first")
        layer_0_data = None
    
    # Load Layer 1 final token features
    try:
        layer_1_data = torch.load("layer_1_sae_features_d4096.pt", map_location='cpu', weights_only=False)
        print(f"✓ Loaded Layer 1 final token features: {layer_1_data['features'].shape}")
    except FileNotFoundError:
        print("❌ Layer 1 features not found")
        layer_1_data = None
    
    return layer_0_data, layer_1_data

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

def analyze_entity_token_features(layer_0_data, layer_1_data, target_entities: List[int]) -> Dict:
    """Analyze token-level features for specific label entities across all token positions"""
    print(f"\n🎯 Analyzing token features for label entities: {target_entities}")
    
    results = defaultdict(lambda: {'layer_0': [], 'layer_1': []})
    
    if layer_0_data is None and layer_1_data is None:
        print("❌ No feature data available")
        return {}
    
    num_examples = len(layer_0_data['token_features']) if layer_0_data else len(layer_1_data['features'])
    
    for example_idx in range(num_examples):
        # Get the label entity for this example
        if layer_0_data and 'label_entities' in layer_0_data:
            label_entity = layer_0_data['label_entities'][example_idx]
        else:
            # Fallback: extract from labels
            if layer_0_data:
                labels = layer_0_data['labels'][example_idx]
                label_positions = (labels != IGNORE_INDEX).nonzero(as_tuple=False)
                if len(label_positions) > 0:
                    label_entity = labels[label_positions[0]].item()
                else:
                    continue
            else:
                continue
        
        # Skip if this label entity is not in our target list
        if label_entity not in target_entities:
            continue
        
        # Parse the sequence for context
        if layer_0_data:
            sequence = layer_0_data['sequences'][example_idx]
            parsed = parse_sequence_with_positions(sequence)
            if parsed is None:
                continue
        
        # Process Layer 0 token features - ALL tokens, but associate with label entity
        if layer_0_data:
            token_features = layer_0_data['token_features'][example_idx]  # [seq_len, 4096]
            
            entity_data = {
                'example_idx': example_idx,
                'label_entity': label_entity,
                'parsed': parsed,
                'all_token_features': [],  # Features for ALL tokens in sequence
                'label_entity_positions': []  # Positions where label entity actually appears
            }
            
            # Store features for ALL token positions
            for pos in range(len(token_features)):
                token_feat = token_features[pos]  # [4096]
                active_features = torch.nonzero(token_feat > 0).squeeze(-1)
                
                if len(active_features) > 0:
                    feature_values = token_feat[active_features]
                    sorted_indices = torch.argsort(feature_values, descending=True)
                    
                    entity_data['all_token_features'].append({
                        'position': pos,
                        'token_id': sequence[pos].item() if pos < len(sequence) else PAD,
                        'top_features': active_features[sorted_indices].tolist(),
                        'top_values': feature_values[sorted_indices].tolist(),
                        'is_label_entity': sequence[pos].item() == label_entity if pos < len(sequence) else False
                    })
            
            # Also track where the label entity specifically appears
            if label_entity in parsed['entity_positions']:
                entity_data['label_entity_positions'] = parsed['entity_positions'][label_entity]
            
            results[label_entity]['layer_0'].append(entity_data)
        
        # Process Layer 1 final token features (associate with label entity)
        if layer_1_data:
            features = layer_1_data['features'][example_idx]  # [4096]
            active_features = torch.nonzero(features > 0).squeeze(-1)
            
            if len(active_features) > 0:
                feature_values = features[active_features]
                sorted_indices = torch.argsort(feature_values, descending=True)
                
                results[label_entity]['layer_1'].append({
                    'example_idx': example_idx,
                    'label_entity': label_entity,
                    'top_features': active_features[sorted_indices].tolist(),
                    'top_values': feature_values[sorted_indices].tolist(),
                    'parsed': parsed if layer_0_data else None
                })
    
    return dict(results)

def plot_entity_token_analysis(entity_results: Dict, save_path: str = "entity_token_analysis.png"):
    """Create visualization for entity token analysis"""
    print("\n📊 Creating entity token analysis plots...")
    
    num_entities = len(entity_results)
    if num_entities == 0:
        print("❌ No entity results to plot")
        return
    
    fig, axes = plt.subplots(num_entities, 2, figsize=(16, 5 * num_entities))
    if num_entities == 1:
        axes = axes.reshape(1, -1)
    
    for i, (entity, data) in enumerate(entity_results.items()):
        # Layer 0 - Token-level features (all positions for label entity)
        ax0 = axes[i, 0]
        
        if data['layer_0']:
            # Collect features from all token positions for this label entity
            all_token_features = []
            label_entity_features = []  # Features specifically when the token IS the label entity
            position_info = []
            
            for example in data['layer_0']:
                for token_data in example['all_token_features']:
                    # Take top 5 features per token
                    top_n = min(5, len(token_data['top_features']))
                    all_token_features.extend(token_data['top_features'][:top_n])
                    
                    # Track features specifically when the token is the label entity
                    if token_data['is_label_entity']:
                        label_entity_features.extend(token_data['top_features'][:top_n])
                    
                    position_info.append(token_data['position'])
            
            if all_token_features:
                # Show both all features and label-entity-specific features
                feature_counts = Counter(all_token_features)
                label_feature_counts = Counter(label_entity_features)
                top_features = feature_counts.most_common(15)
                
                if top_features:
                    features, counts = zip(*top_features)
                    bars = ax0.bar(range(len(features)), counts, alpha=0.7, color='skyblue', label='All tokens')
                    
                    # Overlay label entity features
                    label_counts = [label_feature_counts.get(f, 0) for f in features]
                    bars2 = ax0.bar(range(len(features)), label_counts, alpha=0.8, color='red', label='Label entity tokens')
                    
                    ax0.set_title(f'Label Entity {entity} - Layer 0 All Token Features\n'
                                f'({len(data["layer_0"])} examples, {len(all_token_features)} total activations, '
                                f'{len(label_entity_features)} from label entity)')
                    ax0.set_xlabel('Feature Index (Rank)')
                    ax0.set_ylabel('Activation Frequency')
                    ax0.set_xticks(range(len(features)))
                    ax0.set_xticklabels([str(f) for f in features], rotation=45, fontsize=8)
                    ax0.legend()
                    
                    # Add value labels for total counts
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax0.text(bar.get_x() + bar.get_width()/2., height,
                               f'{count}', ha='center', va='bottom', fontsize=8)
                else:
                    ax0.text(0.5, 0.5, 'No significant features', 
                           ha='center', va='center', transform=ax0.transAxes)
                    ax0.set_title(f'Label Entity {entity} - Layer 0 Token Features')
            else:
                ax0.text(0.5, 0.5, 'No active features found', 
                       ha='center', va='center', transform=ax0.transAxes)
                ax0.set_title(f'Label Entity {entity} - Layer 0 Token Features')
        else:
            ax0.text(0.5, 0.5, 'No Layer 0 data', 
                   ha='center', va='center', transform=ax0.transAxes)
            ax0.set_title(f'Label Entity {entity} - Layer 0 Token Features')
        
        # Layer 1 - Final token features (same as before)
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
                    ax1.set_title(f'Entity {entity} - Layer 1 Final Token\n'
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
                    ax1.set_title(f'Entity {entity} - Layer 1 Final Token')
            else:
                ax1.text(0.5, 0.5, 'No active features found', 
                       ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'Entity {entity} - Layer 1 Final Token')
        else:
            ax1.text(0.5, 0.5, 'No Layer 1 data', 
                   ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'Entity {entity} - Layer 1 Final Token')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {save_path}")

def print_detailed_token_analysis(entity_results: Dict):
    """Print detailed analysis of entity token features"""
    print("\n" + "="*80)
    print("DETAILED ENTITY TOKEN ANALYSIS")
    print("="*80)
    
    for entity, data in entity_results.items():
        print(f"\n🔍 ENTITY {entity}")
        print("-" * 50)
        
        # Layer 0 analysis
        if data['layer_0']:
            print(f"\n  📊 Layer 0 (All token features for label entity {entity}):")
            print(f"    Found in {len(data['layer_0'])} validation examples")
            
            # Count total token positions and label entity positions
            total_token_positions = sum(len(example['all_token_features']) 
                                      for example in data['layer_0'])
            label_entity_positions = sum(sum(1 for token_data in example['all_token_features'] 
                                           if token_data['is_label_entity'])
                                       for example in data['layer_0'])
            
            print(f"    Total token positions analyzed: {total_token_positions}")
            print(f"    Positions where token IS label entity: {label_entity_positions}")
            
            # Aggregate features across all token positions
            all_features = []
            label_entity_features = []
            
            for example in data['layer_0']:
                for token_data in example['all_token_features']:
                    all_features.extend(token_data['top_features'][:5])
                    if token_data['is_label_entity']:
                        label_entity_features.extend(token_data['top_features'][:5])
            
            if all_features:
                feature_counts = Counter(all_features)
                label_feature_counts = Counter(label_entity_features)
                top_features = feature_counts.most_common(5)
                
                print(f"    Top 5 most activated features (all tokens):")
                for feature_idx, count in top_features:
                    label_count = label_feature_counts.get(feature_idx, 0)
                    percentage = count / len(all_features) * 100
                    print(f"      Feature {feature_idx}: {count} total ({percentage:.1f}%), {label_count} from label entity")
            
            # Show example
            if data['layer_0']:
                example = data['layer_0'][0]
                parsed = example['parsed']
                print(f"    Example context:")
                print(f"      Facts: {parsed['facts'][:2]}{'...' if len(parsed['facts']) > 2 else ''}")
                print(f"      Label entity {entity} appears at: {example['label_entity_positions']}")
                
                # Show some token-by-token breakdown
                print(f"      Token breakdown (first few positions):")
                for token_data in example['all_token_features'][:5]:
                    token_id = token_data['token_id']
                    is_label = token_data['is_label_entity']
                    top_feat = token_data['top_features'][0] if token_data['top_features'] else 'None'
                    print(f"        Pos {token_data['position']}: Token {token_id} {'(LABEL!)' if is_label else ''} -> Feature {top_feat}")
        else:
            print(f"\n  📊 Layer 0: No data available")
        
        # Layer 1 analysis (same as before)
        if data['layer_1']:
            print(f"\n  📊 Layer 1 (Final token features):")
            examples = data['layer_1']
            print(f"    Found in {len(examples)} validation examples")
            
            all_features = []
            for example in examples:
                all_features.extend(example['top_features'][:5])
            
            feature_counts = Counter(all_features)
            top_features = feature_counts.most_common(5)
            
            print(f"    Top 5 most activated features:")
            for feature_idx, count in top_features:
                print(f"      Feature {feature_idx}: activated {count} times")
        else:
            print(f"\n  📊 Layer 1: No data available")

def main():
    """Main function"""
    print("🚀 Advanced Entity Token Feature Analysis (Label-Entity Focused)")
    print("=" * 70)
    print("This analysis extracts SAE features for ALL token positions")
    print("but organizes them by LABEL ENTITY (the target answer)")
    print("=" * 70)
    
    # Load features
    layer_0_data, layer_1_data = load_features()
    
    if layer_0_data is None and layer_1_data is None:
        print("❌ No feature data available. Please run the feature extraction scripts first.")
        return
    
    # Show data structure info
    if layer_0_data:
        print(f"\n📋 Data Structure:")
        if 'label_entities' in layer_0_data:
            print(f"  ✓ Label entities available: {len(layer_0_data['label_entities'])} examples")
            unique_labels = set(layer_0_data['label_entities'])
            print(f"  ✓ Unique label entities: {len(unique_labels)} different entities")
        else:
            print(f"  ⚠️  Using fallback label extraction from targets")
    
    # Select 5 random entities that appear as labels
    if layer_0_data and 'label_entities' in layer_0_data:
        available_entities = list(set(layer_0_data['label_entities']))
        available_entities = [e for e in available_entities if e != -1]  # Remove invalid labels
        
        if len(available_entities) < 5:
            target_entities = available_entities
        else:
            random.seed(42)
            target_entities = random.sample(available_entities, 5)
    else:
        # Fallback to random entities
        random.seed(42)
        target_entities = random.sample(range(E), 5)
    
    print(f"\n🎯 Analyzing label entities: {target_entities}")
    print("For each entity, we analyze SAE features from ALL tokens in sequences")
    print("where that entity is the target answer (label)")
    
    # Analyze features
    entity_results = analyze_entity_token_features(layer_0_data, layer_1_data, target_entities)
    
    if not entity_results:
        print("❌ No entities found in the data")
        return
    
    # Print detailed analysis
    print_detailed_token_analysis(entity_results)
    
    # Create plots
    plot_entity_token_analysis(entity_results, "entity_token_feature_analysis.png")
    
    print("\n✅ Advanced entity token analysis complete!")
    print("This shows how SAE features across ALL positions relate to specific label entities.")

if __name__ == "__main__":
    main()
