# I3D: Inflated 3D ConvNet for Video Action Recognition

## Overview

**I3D (Inflated 3D ConvNet)** is a 3D convolutional neural network designed for video action recognition. The model extends the Inception architecture from 2D to 3D convolutions, enabling it to learn spatio-temporal features directly from RGB video frames. I3D can leverage pretrained weights from the Kinetics dataset for transfer learning.

## Architecture

### Input Format
The model accepts input tensor with shape: **(batch, channels, frames, height, width)**
- **batch**: Batch size
- **channels**: Number of channels (3 for RGB)
- **frames**: Number of temporal frames (default: 16)
- **height, width**: Spatial dimensions (default: 224×224)

### Architecture Pipeline

```
Input (B, C, T, H, W)
    ↓
Stem
    ├─ Conv3D 7×7×7 (stride 2×2×2) → 64 channels
    ├─ MaxPool3D 1×3×3 (stride 1×2×2)
    ├─ Conv3D 1×1×1 → 64 channels
    └─ Conv3D 3×3×3 → 192 channels
    ↓
MaxPool3D 1×3×3 (stride 1×2×2)
    ↓
Inception Modules (Mixed_3b, Mixed_3c)
    ↓
MaxPool3D 3×3×3 (stride 2×2×2)
    ↓
Inception Modules (Mixed_4b, 4c, 4d, 4e, 4f)
    ↓
MaxPool3D 2×2×2 (stride 2×2×2)
    ↓
Inception Modules (Mixed_5b, 5c)
    ↓
Adaptive Average Pooling 3D (1×1×1)
    ↓
Dropout
    ↓
Conv3D 1×1×1 → num_classes
    ↓
Spatial Squeeze
    ↓
Output (B, num_classes)
```

## Key Components

### 1. Stem Network
- Initial 3D convolutions to extract low-level spatio-temporal features
- Progressive channel expansion: 3 → 64 → 192
- Temporal and spatial downsampling

### 2. Inception Modules
Each Inception module consists of 4 parallel branches:
- **Branch 0**: 1×1×1 convolution (bottleneck)
- **Branch 1**: 1×1×1 → 3×3×3 convolution
- **Branch 2**: 1×1×1 → 3×3×3 convolution
- **Branch 3**: MaxPool3D → 1×1×1 convolution

All branches are concatenated along the channel dimension.

### 3. Unit3D
Basic building block for I3D:
- 3D convolution with configurable kernel size and stride
- Batch normalization (optional)
- Activation function (ReLU by default)
- Custom padding support (int or tuple)

### 4. MaxPool3dSamePadding
- Custom max pooling with "same" padding
- Maintains spatial/temporal dimensions when needed

### 5. Classification Head
- Adaptive average pooling to (1, 1, 1)
- Dropout for regularization
- Final 1×1×1 convolution to num_classes

## Configuration Parameters

### Model Parameters
- `num_classes` (int, default=400): Number of output classes
- `in_channels` (int, default=3): Input channels (3 for RGB)
- `dropout_keep_prob` (float, default=0.5): Dropout keep probability
- `use_pretrained` (bool, default=False): Load pretrained Kinetics weights
- `pretrained_path` (str, optional): Path to pretrained checkpoint

### Training Recommendations
- **Learning Rate**: 0.001 (standard for CNNs)
- **Optimizer**: Adam with weight decay 0.0001
- **Batch Size**: 8-16 (3D convolutions are memory-intensive)
- **Mixed Precision**: Recommended (use_amp: true)
- **Input Size**: 224×224 pixels, 16 frames

## Advantages

1. **Spatio-Temporal Learning**: 3D convolutions capture both spatial and temporal patterns simultaneously
2. **Transfer Learning**: Pretrained on Kinetics dataset (400 action classes)
3. **Inception Architecture**: Efficient multi-scale feature extraction
4. **Proven Performance**: State-of-the-art results on action recognition benchmarks

## Memory Considerations

I3D is memory-intensive due to 3D convolutions:
- **Batch Size**: Start with 8, increase if GPU memory allows
- **Frame Count**: 16 frames is a good balance (can reduce to 8 if needed)
- **Mixed Precision**: Use AMP to reduce memory usage by ~50%

## Usage Example

```python
from src.model.i3d import I3D

# Create model
model = I3D(
    num_classes=3,
    in_channels=3,
    dropout_keep_prob=0.5,
    use_pretrained=False
)

# Forward pass
# Input: (batch, channels, frames, height, width)
x = torch.randn(8, 3, 16, 224, 224)
output = model(x)  # (8, 3)
```

## Training

```bash
python main.py --config configs/i3d.yaml
```

## Pretrained Weights

I3D can be initialized with pretrained weights from Kinetics dataset:
1. Download pretrained checkpoint (e.g., from official I3D repository)
2. Set `use_pretrained: true` and `pretrained_path: "path/to/checkpoint.pth"` in config

## References

- **Paper**: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (Carreira & Zisserman, 2017)
- **Original Implementation**: [DeepMind Kinetics-I3D](https://github.com/deepmind/kinetics-i3d)
- Pretrained on Kinetics-400 dataset