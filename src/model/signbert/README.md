# SignBERT: Pretrained Transformer for Sign Language Recognition

## Overview

**SignBERT** is a transformer-based model designed for sign language recognition using skeleton/keypoint sequences. The model is inspired by BERT architecture but adapted specifically for processing temporal sequences of human pose data. SignBERT can leverage pretrained weights for transfer learning, making it effective even with limited training data.

## Architecture

### Input Format
The model accepts input tensor with shape: **(batch, frames, joints, coords)** or **(batch, frames, features)**
- **batch**: Batch size
- **frames**: Number of temporal frames (default: 64)
- **joints**: Number of skeleton joints/keypoints (default: 27 for MediaPipe)
- **coords**: Number of coordinates per joint (default: 3 for x, y, z or x, y, confidence)

### Architecture Pipeline

```
Input (B, T, V, C) or (B, T, F)
    ↓
Input Projection: (V × C) → embed_dim
    ↓
Positional Embedding (temporal)
    ↓
N × Transformer Blocks
    ├─ Layer Normalization
    ├─ Multi-Head Self-Attention
    ├─ Drop Path (Stochastic Depth)
    ├─ Layer Normalization
    └─ MLP (Feed-Forward)
    ↓
Layer Normalization
    ↓
Global Average Pooling (temporal)
    ↓
Classification Head
    ↓
Output (B, num_classes)
```

## Key Components

### 1. Input Projection
- Projects flattened keypoint features `(frames, joints × coords)` to embedding dimension
- Linear layer: `input_dim = num_joints × num_coords → embed_dim`

### 2. Positional Embedding
- Learnable positional embeddings for temporal frames
- Shape: `(1, num_frames, embed_dim)`
- Added to input embeddings before transformer blocks

### 3. Transformer Blocks
Each block consists of:
- **Multi-Head Self-Attention**: Captures temporal dependencies
  - Query, Key, Value projections
  - Scaled dot-product attention
  - Multi-head mechanism for diverse representations
- **MLP (Feed-Forward Network)**: 
  - Two linear layers with GELU activation
  - Hidden dimension = `embed_dim × mlp_ratio`
- **Drop Path (Stochastic Depth)**: Regularization technique
- **Layer Normalization**: Applied before attention and MLP

### 4. Classification Head
- Global average pooling over temporal dimension
- Two-layer MLP with dropout
- Output: Classification logits

## Configuration Parameters

### Model Parameters
- `num_joints` (int, default=27): Number of skeleton joints
- `num_coords` (int, default=3): Number of coordinates per joint
- `num_frames` (int, default=64): Temporal sequence length
- `embed_dim` (int, default=256): Embedding dimension
- `depth` (int, default=6): Number of transformer blocks
- `num_heads` (int, default=8): Number of attention heads
- `mlp_ratio` (float, default=4.0): MLP hidden dim = embed_dim × mlp_ratio
- `drop_rate` (float, default=0.1): Dropout rate
- `use_pretrained` (bool, default=False): Load pretrained weights
- `pretrained_path` (str, optional): Path to pretrained checkpoint

### Training Recommendations
- **Learning Rate**: 0.0001 (lower than typical CNNs)
- **Optimizer**: AdamW with weight decay 0.01
- **Batch Size**: 16-32 (depending on GPU memory)
- **Label Smoothing**: 0.1 (helps with overfitting on small datasets)
- **Warmup**: 5 epochs (for transformers)

## Advantages

1. **Transfer Learning**: Supports pretrained weights for better performance on small datasets
2. **Temporal Modeling**: Self-attention mechanism captures long-range temporal dependencies
3. **Flexible Input**: Handles both raw keypoint sequences and pre-processed features
4. **Regularization**: Drop path and dropout prevent overfitting

## Usage Example

```python
from src.model.signbert import SignBERT

# Create model
model = SignBERT(
    num_joints=27,
    num_coords=3,
    num_frames=64,
    embed_dim=256,
    depth=6,
    num_heads=8,
    num_classes=3,
    use_pretrained=False
)

# Forward pass
# Input: (batch, frames, joints, coords)
x = torch.randn(8, 64, 27, 3)
output = model(x)  # (8, 3)
```

## Training

```bash
python main.py --config configs/signbert.yaml
```

## References

- Inspired by BERT architecture for sequence modeling
- Adapted for skeleton-based sign language recognition
- Supports pretrained weights for transfer learning