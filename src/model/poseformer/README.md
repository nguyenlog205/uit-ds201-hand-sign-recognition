# PoseFormer: Transformer-based Model for Sign Language Recognition

## Overview

**PoseFormer** is a transformer-based model for sign language recognition that processes skeleton keypoint sequences through a two-stage architecture: spatial transformer for frame-level feature extraction and temporal transformer for sequence-level modeling. The model is adapted from the original PoseFormer designed for 3D human pose estimation, modified for sign language classification tasks.

## Architecture

### Input Format
The model accepts input tensor with shape: **(batch, frames, joints, coords)**
- **batch**: Batch size
- **frames**: Number of temporal frames (default: 64)
- **joints**: Number of skeleton joints (default: 27 for MediaPipe)
- **coords**: Number of coordinates per joint (default: 3 for x, y, z or x, y, confidence)

### Architecture Pipeline

```
Input (B, T, V, C)
    ↓
Spatial Processing (per frame)
    ├─ Patch Embedding: (V, C) → (V, embed_dim_ratio)
    ├─ Spatial Positional Embedding
    └─ N × Spatial Transformer Blocks
        ├─ Multi-Head Self-Attention (spatial)
        └─ MLP
    ↓
Spatial Features: (B, T, V, embed_dim_ratio)
    ↓
Reshape: (B, T, V × embed_dim_ratio) = (B, T, embed_dim)
    ↓
Temporal Processing
    ├─ Temporal Positional Embedding
    └─ N × Temporal Transformer Blocks
        ├─ Multi-Head Self-Attention (temporal)
        └─ MLP
    ↓
Temporal Features: (B, T, embed_dim)
    ↓
Global Average Pooling (temporal)
    ↓
Classification Head
    ↓
Output (B, num_classes)
```

## Key Components

### 1. Spatial Transformer
Processes each frame independently to extract spatial relationships between joints:
- **Patch Embedding**: Linear projection from joint coordinates to embedding space
  - Input: `(joints, coords)` → Output: `(joints, embed_dim_ratio)`
- **Spatial Positional Embedding**: Learnable embeddings for joint positions
- **Spatial Transformer Blocks**: 
  - Self-attention over joints within each frame
  - Captures spatial relationships (e.g., hand-hand, hand-body interactions)
  - Output: `(frames, joints, embed_dim_ratio)`

### 2. Temporal Transformer
Processes the temporal sequence of spatial features:
- **Temporal Embedding**: `embed_dim = embed_dim_ratio × num_joints`
  - Each frame is represented as a token with dimension `embed_dim`
- **Temporal Positional Embedding**: Learnable embeddings for frame positions
- **Temporal Transformer Blocks**:
  - Self-attention over frames
  - Captures temporal dynamics and motion patterns
  - Output: `(frames, embed_dim)`

### 3. Transformer Block Structure
Each block (both spatial and temporal) consists of:
- **Layer Normalization** → **Multi-Head Self-Attention** → **Residual Connection**
- **Layer Normalization** → **MLP (Feed-Forward)** → **Residual Connection**
- **Drop Path (Stochastic Depth)**: Regularization technique

### 4. Classification Head
- Global average pooling over temporal dimension
- Layer normalization + dropout
- Two-layer MLP with GELU activation
- Output: Classification logits

## Configuration Parameters

### Critical Parameters

#### `embed_dim_ratio` (CRITICAL)
- **Default**: 32 (recommended for skeleton data)
- **Formula**: `embed_dim = embed_dim_ratio × num_joints`
- **Warning**: Large values (e.g., 256) cause parameter explosion
  - Example: `embed_dim_ratio=256` with 27 joints → `embed_dim=6912` (too large!)
  - Recommended range: 32-64 for skeleton data

#### Model Parameters
- `num_frame` (int, default=64): Temporal sequence length
- `num_joints` (int, default=27): Number of skeleton joints
- `in_chans` (int, default=3): Input channels (x, y, z or x, y, confidence)
- `embed_dim_ratio` (int, default=32): **CRITICAL** - Controls model size
- `depth` (int, default=4): Number of transformer blocks (both spatial and temporal)
- `num_heads` (int, default=8): Number of attention heads
- `mlp_ratio` (float, default=2.0): MLP hidden dim = embed_dim × mlp_ratio
- `drop_rate` (float, default=0.1): Dropout rate
- `num_class` (int): Number of output classes

### Training Recommendations
- **Learning Rate**: 0.0001 (lower for transformers)
- **Optimizer**: AdamW with weight decay 0.01
- **Batch Size**: 16-32
- **Label Smoothing**: 0.1 (helps with overfitting)
- **Warmup**: 5 epochs
- **Early Stopping**: Patience 10

## Parameter Size Calculation

The model size is determined by `embed_dim_ratio`:

```
embed_dim = embed_dim_ratio × num_joints

Example with 27 joints:
- embed_dim_ratio = 32  → embed_dim = 864  (reasonable)
- embed_dim_ratio = 64  → embed_dim = 1728 (large but manageable)
- embed_dim_ratio = 256 → embed_dim = 6912 (too large! Parameter explosion)
```

**Recommendation**: Start with `embed_dim_ratio=32` and increase only if needed.

## Advantages

1. **Two-Stage Processing**: Separates spatial and temporal modeling for better feature extraction
2. **Attention Mechanism**: Captures long-range dependencies in both space and time
3. **Efficient Architecture**: Lower parameter count compared to full 3D transformers
4. **Flexible**: Can handle variable-length sequences with padding

## Limitations

1. **Parameter Sensitivity**: `embed_dim_ratio` must be carefully tuned to avoid parameter explosion
2. **Memory Usage**: Attention mechanism is quadratic in sequence length
3. **Small Datasets**: May overfit on very small datasets without proper regularization

## Usage Example

```python
from src.model.poseformer import PoseFormer

# Create model
model = PoseFormer(
    num_frame=64,
    num_joints=27,
    in_chans=3,
    embed_dim_ratio=32,  # CRITICAL: Keep this small!
    depth=4,
    num_heads=8,
    mlp_ratio=2.0,
    num_class=3
)

# Forward pass
# Input: (batch, frames, joints, coords)
x = torch.randn(8, 64, 27, 3)
output = model(x)  # (8, 3)
```

## Training

```bash
python main.py --config configs/poseformer.yaml
```

## Common Issues and Solutions

### Issue: Parameter Explosion
**Symptom**: Model initialization is very slow, high CPU/RAM usage
**Solution**: Reduce `embed_dim_ratio` to 32 or lower

### Issue: RuntimeError: Shape mismatch in weighted_mean
**Symptom**: Error in forward pass
**Solution**: This has been fixed in the current implementation (uses mean pooling instead)

### Issue: Overfitting on Small Dataset
**Solution**: 
- Increase dropout rate
- Use label smoothing (0.1)
- Reduce model depth
- Increase weight decay

## References

- **Original Paper**: "PoseFormer: A Simple Baseline for 3D Human Pose Estimation" (Zheng et al., 2021)
- **Original Implementation**: [PoseFormer GitHub](https://github.com/zczcwh/PoseFormer)
- Adapted for sign language recognition (classification task)