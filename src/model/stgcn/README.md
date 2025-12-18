# ST-GCN: Spatial Temporal Graph Convolutional Networks

## Overview

**ST-GCN (Spatial Temporal Graph Convolutional Networks)** is a standard GCN model for action recognition and sign language, learning both **spatial relationships** (graph convolution on joints) and **temporal relationships** (temporal convolution on frames) simultaneously.

## Overall Architecture

### Input Format
The model accepts input tensor with shape: **(N, C, T, V, M)**
- **N**: Batch size
- **C**: Number of channels (default is 3: x, y, confidence)
- **T**: Number of frames in temporal sequence
- **V**: Number of joints (default 27 for Mediapipe/Sign language)
- **M**: Number of people (default is 1)

### Architecture Pipeline

```
Input (N, C, T, V, M)
    ↓
Data Batch Normalization (optional)
    ↓
Initial GCN + TCN (gcn0 + tcn0)
    ↓
9 × ST-GCN Blocks (backbone)
    ├─ Block 1-3: 64 channels
    ├─ Block 4-6: 128 channels (stride=2 at block 4)
    └─ Block 7-9: 256 channels (stride=2 at block 7)
    ↓
Spatial Pooling (over joints V)
    ↓
Temporal Pooling (over frames T)
    ↓
1D Convolution (FCN)
    ↓
Output (N, num_class)
```

## Main Components

### 1. STGCNBlock (`stgcn.py`)

Each ST-GCN block consists of 3 components:

```
Input
  ↓
[SpatialGCN] → Graph Convolution on joints
  ↓
[TemporalConv] → 1D Convolution on frames
  ↓
[Residual Connection]
  ↓
Output
```

**Parameters:**
- `in_channels`, `out_channels`: Input/output channel numbers
- `A`: Adjacency matrix (K, V, V) with K partitions
- `kernel_size`: Kernel size for temporal conv (default: 9)
- `stride`: Stride for temporal convolution
- `dropout`: Dropout rate (default: 0.5)
- `use_local_bn`: Use local batch norm per node
- `mask_learning`: Learn mask to reweight adjacency matrix

**Forward process:**
1. Spatial GCN: `x = gcn(x)` → Graph convolution on joints
2. Temporal Conv: `x = tcn(x)` → 1D convolution on frames
3. Residual: `x = x + downsample(input)` → Skip connection

### 2. SpatialGCN (`spatial_gcn.py`)

**Spatial Graph Convolution** - graph convolution layer on joint space.

**Spatial Partitioning Strategy:**
The graph is divided into **K partitions** (typically K=3):
- **Partition 0**: Root node (itself)
- **Partition 1**: Centripetal group (neighbors closer to skeleton center)
- **Partition 2**: Centrifugal group (neighbors farther from skeleton center)

**Formula:**
```
y = Σ(k=0 to K-1) Conv(x @ A[k])
```

Where:
- **A[k]**: Adjacency matrix for partition k (V, V)
- **x @ A[k]**: Matrix multiplication to aggregate neighbors
- **Conv**: 2D convolution (kernel_size, 1) on each partition
- **Σ**: Aggregate results from all partitions

**Advanced options:**
- **Mask Learning**: Learn mask to reweight adjacency matrix
  ```python
  A_learned = A * mask  # mask is learnable parameter
  ```
- **Local Batch Norm**: Each node has its own BN parameters
  - Global BN: `(N, C, T, V)` → BN2D
  - Local BN: `(N, C, T, V)` → `(N, C*V, T)` → BN1D → reshape back

**Forward process:**
1. Apply mask (if `mask_learning=True`): `A = A * mask`
2. For each partition k:
   - Reshape: `(N, C, T, V)` → `(N*T*C, V)`
   - Matrix multiplication: `xa = x @ A[k]` → `(N*T*C, V)`
   - Reshape back: `(N*T*C, V)` → `(N, C, T, V)`
   - Convolution: `conv_k(xa)`
3. Aggregate: `y = Σ conv_k(xa)`
4. Batch normalization (global or local)
5. ReLU activation

### 3. TemporalConv (`temporal_conv.py`)

**Temporal Convolution** - 1D convolution layer along temporal dimension.

**Formula:**
```
y = ReLU(BN(Conv1D(x)))
```

**Details:**
- **Conv2D**: `kernel_size=(kernel_size, 1)`, `stride=(stride, 1)`
  - Convolve along T dimension, keep V dimension unchanged
- **BatchNorm2D**: Normalize along batch
- **Dropout**: Optional dropout before convolution
- **ReLU**: Activation function

**Forward process:**
1. Dropout (if `dropout > 0`)
2. 2D Convolution: `(N, C, T, V)` → `(N, C_out, T_out, V)`
3. Batch Normalization
4. ReLU activation

### 4. STGCN Model (`stgcn.py`)

**Spatial Temporal Graph Convolutional Network** - main model.

#### Structure

1. **Data Batch Normalization** (optional)
   - Normalize input data before entering network
   - Supports multi-person: `(N, M*V*C, T)` or single-person: `(N*M, V*C, T)`

2. **Initial Layers**
   - `gcn0`: Initial spatial GCN (3 → 64 channels)
   - `tcn0`: Initial temporal conv

3. **Backbone** (9 ST-GCN Blocks)
   - Default configuration: `DEFAULT_BACKBONE`
   - Can be customized via `backbone_config`

4. **Classification Head**
   - Spatial pooling: Average pool over joints `(N, C, T, V)` → `(N, C, T, 1)`
   - Person pooling: Average over people (if M > 1)
   - Temporal pooling: Average pool over frames `(N, C, T)` → `(N, C, 1)`
   - FCN: 1D convolution `(N, C, 1)` → `(N, num_class, 1)`
   - Final pooling: `(N, num_class, 1)` → `(N, num_class)`

#### Default Backbone Configuration

```python
DEFAULT_BACKBONE = [
    (64, 64, 1),   # Block 1: 64→64, stride=1
    (64, 64, 1),   # Block 2: 64→64, stride=1
    (64, 64, 1),   # Block 3: 64→64, stride=1
    (64, 128, 2),  # Block 4: 64→128, stride=2
    (128, 128, 1), # Block 5: 128→128, stride=1
    (128, 128, 1), # Block 6: 128→128, stride=1
    (128, 256, 2), # Block 7: 128→256, stride=2
    (256, 256, 1), # Block 8: 256→256, stride=1
    (256, 256, 1), # Block 9: 256→256, stride=1
]
```

## Model Configuration

### Initialization from Config

The model is created through factory function:

```python
from src.model.gcn_model_factory import create_model

model = create_model(config)  # if config["model"]["type"] = "stgcn"
```

Or directly:

```python
from src.model.gcn_model_factory import create_stgcn_from_config

model = create_stgcn_from_config(config, num_classes=None)
```

### Configuration Parameters

In `config["model"]`:

- `in_channels`: Number of input channels (default: 3)
- `num_nodes`: Number of joints (default: 27)
- `num_person`: Number of people (default: 1)
- `window_size`: Temporal window size
- `num_class`: Number of classification classes
- `skeleton_layout`: Skeleton layout (e.g., `"mediapipe_27"`)
- `adjacency_strategy`: Adjacency matrix creation strategy (e.g., `"spatial"`)
- `use_data_bn`: Enable/disable data batch normalization (default: True)
- `mask_learning`: Learn mask to reweight adjacency (default: False)
- `use_local_bn`: Use local batch norm per node (default: False)
- `temporal_kernel_size`: Kernel size for temporal conv (default: 9)
- `dropout`: Dropout rate (default: 0.5)
- `backbone_config`: Custom backbone config (list of `(in_c, out_c, stride)`)

### Direct Initialization

```python
from src.model.stgcn import STGCN
import torch

# Create adjacency matrix A (K, V, V)
A = ...  # from SkeletonGraph or config

model = STGCN(
    in_channels=3,
    num_class=100,
    num_nodes=27,
    num_person=1,
    window_size=64,
    use_data_bn=True,
    backbone_config=None,  
    A=A,
    mask_learning=False,
    use_local_bn=False,
    temporal_kernel_size=9,
    dropout=0.5
)
```

## Architecture Details

### Forward Pass

```python
def forward(self, x):
    # x: (N, C, T, V, M)
    N, C, T, V, M = x.size()
    
    # 1. Data Batch Normalization
    if self.use_data_bn:
        if M > 1:
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M*V*C, T)
        else:
            x = x.permute(0, 4, 3, 1, 2).contiguous().view(N*M, V*C, T)
        x = self.data_bn(x)
        # Reshape back: (N*M, C, T, V)
        ...
    else:
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N*M, C, T, V)
    
    # 2. Initial Layers
    x = self.gcn0(x)  # (N*M, 64, T, V)
    x = self.tcn0(x)  # (N*M, 64, T, V)
    
    # 3. Backbone (9 blocks)
    for block in self.backbone:
        x = block(x)  # (N*M, C, T, V)
    
    # 4. Spatial Pooling
    x = F.avg_pool2d(x, kernel_size=(1, V))  # (N*M, C, T, 1)
    x = x.squeeze(-1)  # (N*M, C, T)
    
    # 5. Person Pooling (if M > 1)
    if M > 1:
        x = x.view(N, M, C, T)  # (N, M, C, T)
        x = x.mean(dim=1)  # (N, C, T)
    
    # 6. Temporal Pooling
    x = F.avg_pool1d(x, kernel_size=x.size(2))  # (N, C, 1)
    x = x.squeeze(-1)  # (N, C)
    
    # 7. Classification
    x = x.unsqueeze(-1)  # (N, C, 1)
    x = self.fcn(x)  # (N, num_class, 1)
    x = F.avg_pool1d(x, kernel_size=x.size(2))  # (N, num_class, 1)
    x = x.squeeze(-1)  # (N, num_class)
    
    return x
```

### Spatial Partitioning Strategy

ST-GCN uses **spatial partitioning** to divide neighbors of each node into groups:

1. **Root partition**: The node itself
2. **Centripetal partition**: Neighbors closer to skeleton center
3. **Centrifugal partition**: Neighbors farther from skeleton center

This helps the model learn different patterns based on the direction of connections in the skeleton.

### Mask Learning

When `mask_learning=True`, the model learns a mask to reweight the adjacency matrix:

```python
A_learned = A * mask  # mask is learnable parameter (K, V, V)
```

This allows the model to automatically adjust the importance of connections in the graph.

### Local Batch Normalization

When `use_local_bn=True`, each node has its own batch normalization parameters:

- **Global BN**: `(N, C, T, V)` → BN2D → All nodes share parameters
- **Local BN**: `(N, C, T, V)` → `(N, C*V, T)` → BN1D → Each node has its own parameters

Local BN can be useful when nodes have different distributions.

## Advantages of ST-GCN

1. **Simple and Easy to Understand**: Clear architecture, easy to debug and customize
2. **Efficient**: Effectively combines spatial and temporal convolution
3. **Flexible**: Supports many options (mask learning, local BN, custom backbone)
4. **Standard**: Implementation follows original paper, easy to compare with other research
5. **Multi-person Support**: Supports processing multiple people in the same frame

## When to Use ST-GCN

- When you have **pose sequences (keypoints)** and want to use a standard, simple, easy-to-debug GCN
- When you don't need complex hand structure like HA-GCN
- When you want to model **body structure + two hands** via adjacency matrix
- When you need to capture temporal patterns using 1D convolution
- When you want a baseline model to compare with more complex models

## File Structure

```
stgcn/
├── __init__.py              # Exports main classes
├── stgcn.py                 # STGCN, STGCNBlock - main model
├── spatial_gcn.py           # SpatialGCN, UnitGCN - graph convolution
├── temporal_conv.py         # TemporalConv, Unit2D - temporal convolution
└── README.md                # This documentation
```

## Dependencies

- `torch`: PyTorch framework
- `torch.nn.functional`: F.avg_pool2d, F.avg_pool1d
- `src.data.gcn.graph_constructor.SkeletonGraph`: To create adjacency matrix

## References

- **Original Paper**: [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- **Implementation**: Based on ST-GCN codebase with adaptations for sign language recognition

## Notes

- Adjacency matrix A must have shape `(K, V, V)` where K is the number of partitions (typically K=3)
- If A is 2D `(V, V)`, need to expand to `(1, V, V)` or stack to `(K, V, V)`
- Backbone configuration can be customized but must ensure consistency in channels and strides