"""Analysis utilities for the relational-recall toy model.

Submodules:
  - data: shared batch construction with SEP/Q/target-fact-SEP positions
  - patching: activation patching and head role scans
  - ablation: mean-ablation with cached reference means
  - geometry: pairwise cosine + PCA on composed vs isolated activations
  - weights: QK/OV SVD + eigendecomposition helpers
  - subspaces: address/payload subspaces from L0 OV
"""
