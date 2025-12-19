# SignBERT: Pretrained Transformer for Sign Language Recognition

## Overview

**SignBERT** is a transformer-based model designed for sign language recognition using skeleton/keypoint sequences. The model follows BERT architecture principles adapted for skeleton data with proper spatial-temporal modeling, joint-aware embeddings, and skeleton normalization.

## Architecture

### Input Format
The model accepts input tensor with shape: **(batch, frames, joints, coords)**
- **batch**: Batch size
- **frames**: Number of temporal frames (default: 64)
- **joints**: Number of skeleton joints/keypoints (default: 27 for MediaPipe)
- **coords**: Number of coordinates per joint (default: 3 for x, y, z or x, y, confidence)

### Architecture Pipeline

```
Input (B, T, V, C)
    ↓
Skeleton Normalization
    ├─ Root centering (subtract neck/root joint)
    └─ Scale normalization (by shoulder width)
    ↓
Feature Extraction
    ├─ Base features: (T, V, C)
    ├─ Velocity features: Δx (temporal differences)
    └─ Bone features: parent-child vectors
    ↓
Input Projection: features → embed_dim
    ↓
Restructure: (B, T, V, embed_dim) → (B, T*V, embed_dim)
    ↓
Positional Embeddings
    ├─ Temporal positional embedding (frame position)
    ├─ Joint positional embedding (joint position)
    └─ Hand type embedding (body/left_hand/right_hand)
    ↓
[CLS] Token (BERT-style)
    ↓
N × Transformer Blocks
    ├─ Layer Normalization
    ├─ Multi-Head Self-Attention (spatial-temporal)
    ├─ Drop Path (Stochastic Depth)
    ├─ Layer Normalization
    └─ MLP (Feed-Forward)
    ↓
Layer Normalization
    ↓
Extract [CLS] token representation
    ↓
Classification Head
    ↓
Output (B, num_classes)
```

## Key Components

### 1. Skeleton Normalization
- **Root Centering**: Subtracts root joint (neck) to center skeleton
- **Scale Normalization**: Normalizes by shoulder width for scale invariance
- Critical for sign language recognition as gestures are relative, not absolute

### 2. Feature Extraction
- **Base Features**: Raw keypoint coordinates (B, T, V, C)
- **Velocity Features**: Temporal differences Δx = x[t] - x[t-1] (optional, same shape)
- **Bone Features**: Parent-child joint vectors based on skeleton topology (optional, same shape)
  - Uses proper parent-child relationships from skeleton graph
  - Root joint (neck) has zero bone vector
- All features concatenated along channel dimension: (B, T, V, C') where C' = C + C_vel + C_bone

### 3. Input Projection & Tokenization
- Projects concatenated features to embedding dimension: `(B, T, V, C') → (B, T, V, embed_dim)`
- **Restructures** to joint-level tokens: `(B, T, V, embed_dim) → (B, T*V, embed_dim)`
- Each token represents a (frame, joint) pair for proper spatial-temporal attention
- **Critical**: Joint dimension is preserved during feature concatenation to maintain skeleton structure

### 4. Positional Embeddings (Multi-level)
- **Temporal Positional Embedding**: Frame position in sequence
- **Joint Positional Embedding**: Joint position in skeleton structure
- **Hand Type Embedding**: Body/Left Hand/Right Hand type (learnable)
- All embeddings added to input tokens

### 5. [CLS] Token (BERT-style)
- Learnable classification token prepended to sequence
- Has its own positional embedding (separate from frame/joint embeddings)
- Represents global sequence representation
- Used for final classification (instead of global average pooling)

### 6. Transformer Blocks
Each block consists of:
- **Multi-Head Self-Attention**: Captures spatial-temporal dependencies
  - Attention over (T*V) tokens learns both temporal and spatial relationships
  - Query, Key, Value projections
  - Scaled dot-product attention
  - Multi-head mechanism for diverse representations
- **MLP (Feed-Forward Network)**: 
  - Two linear layers with GELU activation
  - Hidden dimension = `embed_dim × mlp_ratio`
- **Drop Path (Stochastic Depth)**: Regularization technique
- **Layer Normalization**: Applied before attention and MLP

### 7. Classification Head
- Uses [CLS] token representation (not global average pooling)
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
- `use_cls_token` (bool, default=True): Use [CLS] token (BERT-style)
- `normalize_skeleton` (bool, default=True): Apply skeleton normalization
- `use_velocity` (bool, default=True): Include velocity features (Δx)
- `use_bone` (bool, default=True): Include bone vector features

### Training Recommendations
- **Learning Rate**: 0.0001 (lower than typical CNNs)
- **Optimizer**: AdamW with weight decay 0.01
- **Batch Size**: 16-32 (depending on GPU memory)
- **Label Smoothing**: 0.1 (helps with overfitting on small datasets)
- **Warmup**: 5 epochs (for transformers)

## Advantages

1. **SignBERT-style Architecture**: Proper spatial-temporal modeling with joint-level tokens
2. **[CLS] Token**: BERT-style classification token for better sequence representation
3. **Joint-aware Embeddings**: Temporal + Joint + Hand type embeddings for rich positional information
4. **Skeleton Normalization**: Root centering and scale normalization for robust recognition
5. **Multi-modal Features**: Base + Velocity + Bone features capture different aspects of motion
6. **Transfer Learning**: Supports pretrained weights for better performance on small datasets
7. **Regularization**: Drop path and dropout prevent overfitting

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

## Key Improvements over Baseline Transformer

### ✅ SignBERT-style Features
1. **[CLS] Token**: Uses classification token instead of global average pooling
2. **Joint-level Tokens**: Each token is (frame, joint) pair for spatial-temporal attention
3. **Multi-level Positional Embeddings**: Temporal + Joint + Hand type embeddings
4. **Skeleton Normalization**: Root centering and scale normalization
5. **Multi-modal Features**: Base coordinates + Velocity + Bone vectors

### 📊 Comparison

| Feature | Baseline Transformer | SignBERT (This Implementation) |
|---------|---------------------|-------------------------------|
| Token Structure | Frame-level (T tokens) | Joint-level (T*V tokens) |
| Classification | Global Average Pooling | [CLS] Token |
| Positional Embedding | Temporal only | Temporal + Joint + Hand |
| Skeleton Normalization | ❌ | ✅ |
| Velocity Features | ❌ | ✅ (optional) |
| Bone Features | ❌ | ✅ (optional) |
| Spatial Modeling | Limited | Full spatial-temporal |

## References

- Inspired by BERT architecture for sequence modeling
- Adapted for skeleton-based sign language recognition with proper spatial-temporal modeling
- Supports pretrained weights for transfer learning
- Follows SignBERT paper principles (with pretraining tasks to be added)