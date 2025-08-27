# SAE Training with Pipeline for HookedTransformer

This repository provides a pipeline-based approach to train Sparse Autoencoders (SAEs) on HookedTransformer activations with Weights & Biases logging and parallel training capabilities.

## Features

- **Parallel SAE Training**: Train multiple SAEs on different layers simultaneously
- **Weights & Biases Integration**: Automatic logging of training metrics, artifacts, and checkpoints
- **Custom Model Support**: Works with your own HookedTransformer models
- **Automatic Checkpointing**: Saves model checkpoints at specified intervals
- **Feature Extraction**: Extract and analyze SAE features after training
- **Modular Design**: Easy to extend and customize for different use cases

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Weights & Biases (optional but recommended):
```bash
wandb login
```

## Quick Start

### Basic Usage

```python
from sae_features_d4096 import train_saes_with_pipeline, extract_sae_features

# Train SAEs on multiple layers
run_ids, trainer_configs, model = train_saes_with_pipeline(
    layers_to_train=[0, 1],  # Train on layers 0 and 1
    sae_dim=4096,            # SAE dictionary size
    use_wandb=True           # Enable W&B logging
)

# Extract features from trained SAEs
for i, layer in enumerate([0, 1]):
    sae_path = f"./sae_checkpoints/trainer_{i}/checkpoints/ae_10000000.pt"
    features = extract_sae_features(layer, sae_path, val_loader, device, model)
```

### Example Script

Run the complete pipeline with:
```bash
python example_usage.py
```

## Configuration

### SAE Training Parameters

The pipeline supports the following key parameters:

- **`layers_to_train`**: List of layer indices to train SAEs on
- **`sae_dim`**: Dictionary size (hidden dimension) of the SAE
- **`use_wandb`**: Enable/disable Weights & Biases logging
- **`steps`**: Total training steps (default: 10M)
- **`batch_size`**: Training batch size (default: 64)
- **`lr`**: Learning rate (default: 1e-4)
- **`l1_penalty`**: Sparsity penalty (default: 1e-1)

### Weights & Biases Configuration

To customize W&B logging:

1. Set your entity and project in the training function:
```python
run_ids = run_pipeline(
    # ... other parameters ...
    wandb_entity="your_username",
    wandb_project="your_project_name",
    use_wandb=True
)
```

2. The pipeline automatically logs:
   - Training loss (MSE + sparsity)
   - Feature sparsity (L0 norm)
   - Unique active features
   - Model checkpoints as artifacts
   - Training configuration

## File Structure

```
├── sae_features_d4096.py      # Main SAE training and feature extraction
├── dl_quick_train/
│   └── pipeline.py            # Core pipeline implementation
├── example_usage.py           # Example usage script
├── requirements.txt            # Dependencies
└── SAE_PIPELINE_README.md     # This file
```

## Training Process

1. **Model Loading**: Loads your HookedTransformer model with pretrained weights
2. **SAE Initialization**: Creates StandardTrainer instances for each layer
3. **Parallel Training**: Uses multiprocessing to train SAEs simultaneously
4. **Checkpointing**: Saves model checkpoints at specified intervals
5. **W&B Logging**: Logs metrics and artifacts to Weights & Biases
6. **Feature Extraction**: Extracts and analyzes SAE features from validation set

## Output Files

### Checkpoints
- Location: `./sae_checkpoints/trainer_{i}/checkpoints/`
- Format: `ae_{step}.pt`
- Contains: SAE state dictionary

### Features
- Location: `./` (configurable)
- Format: `layer_{layer}_sae_features_d{sae_dim}.pt`
- Contains: Features, activations, reconstructions, labels, analysis

### Weights & Biases
- Training metrics and loss curves
- Model checkpoints as artifacts
- Training configuration
- Run IDs for each SAE

## Customization

### Different Activation Sites

To train on different activation sites, modify the submodule parameter:

```python
run_ids = run_pipeline(
    # ... other parameters ...
    submodule="blocks.0.attn.hook_result",  # Attention output
    # or
    submodule="blocks.1.mlp.hook_post",     # MLP output
)
```

### Custom Datasets

The pipeline supports custom datasets through the `custom_dataset` parameter:

```python
class CustomDatasetWrapper:
    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.iterator = None
        
    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(DataLoader(self.dataset, batch_size=64, collate_fn=self.collate_fn))
        return self.iterator

wrapped_dataset = CustomDatasetWrapper(your_dataset, your_collate_fn)
```

### Multiple SAE Configurations

Train SAEs with different hyperparameters:

```python
def create_custom_trainer_configs():
    configs = []
    
    # SAE for layer 0 with different hyperparameters
    configs.append({
        "trainer": StandardTrainer,
        "steps": 5_000_000,
        "activation_dim": 256,
        "dict_size": 2048,  # Smaller dictionary
        "layer": 0,
        "lr": 5e-5,         # Lower learning rate
        "l1_penalty": 5e-2, # Lower sparsity penalty
        # ... other parameters
    })
    
    # SAE for layer 1 with standard hyperparameters
    configs.append({
        "trainer": StandardTrainer,
        "steps": 10_000_000,
        "activation_dim": 256,
        "dict_size": 4096,
        "layer": 1,
        "lr": 1e-4,
        "l1_penalty": 1e-1,
        # ... other parameters
    })
    
    return configs
```

## Monitoring Training

### Weights & Biases Dashboard

1. Navigate to your project dashboard
2. View training metrics in real-time
3. Compare different runs
4. Download model checkpoints

### Local Logs

The pipeline provides verbose logging:
- Training progress every 500 steps
- Loss metrics (total, MSE, sparsity)
- Feature statistics
- Checkpoint saving notifications

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or SAE dimension
2. **W&B Connection Issues**: Check internet connection and API key
3. **Checkpoint Loading Errors**: Verify file paths and model compatibility

### Performance Tips

1. **GPU Memory**: Use gradient checkpointing for large models
2. **Batch Size**: Optimize based on your GPU memory
3. **Checkpoint Frequency**: Balance between disk space and recovery capability

## Advanced Usage

### Custom Loss Functions

Extend the pipeline with custom loss functions by modifying the trainer configuration:

```python
class CustomTrainer(StandardTrainer):
    def loss(self, x, step=None, logging=False):
        # Custom loss computation
        pass

config = {
    "trainer": CustomTrainer,
    # ... other parameters
}
```

### Multi-GPU Training

For multi-GPU setups, modify the device mapping:

```python
device = "cuda:0"  # Primary GPU
# or
device = "auto"    # Automatic device mapping
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sae_pipeline,
  title={SAE Training Pipeline for HookedTransformer},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/dl-quick-train}
}
```

## Support

For questions and issues:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the maintainers

---

Happy training! 🚀
