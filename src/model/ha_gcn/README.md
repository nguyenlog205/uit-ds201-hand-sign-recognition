# HA-GCN: Hand-aware Graph Convolutional Network

## Overview

**HA-GCN (Hand-aware Graph Convolutional Network)** is a specialized GCN model for sign language recognition, designed to deeply exploit hand structure in pose data. The model combines body graph with hand graphs to enhance the ability to learn fine-grained features of hand gestures.

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
Data Batch Normalization
    ↓
10 × HA-GCN Blocks (layer_1 → layer_10)
    ├─ Block 1-4: 64 channels
    ├─ Block 5-7: 128 channels (stride=2 at block 5)
    └─ Block 8-10: 256 channels (stride=2 at block 8)
    ↓
Global Average Pooling (T, V)
    ↓
Fully-Connected Layer
    ↓
Output (N, num_class)
```

## Main Components

### 1. HAGCNBlock (`ha_gcn.py`)

Each HA-GCN block consists of 4 main components:

```
Input
  ↓
[HandAwareGCN] → Graph Convolution with A + SH + PH
  ↓
[STCAttention] → Spatial-Temporal-Channel Attention (optional)
  ↓
[TemporalConv] → Temporal Convolution + Adaptive DropGraph
  ↓
[Residual Connection]
  ↓
Output
```

**Parameters:**
- `in_channels`, `out_channels`: Input/output channel numbers
- `A`: Body adjacency matrix (K, V, V)
- `SH`: Structured Hand Graph (K, V, V) - fixed
- `PH`: Parameterized Hand Graph (K, V, V) - learnable
- `num_point`: Number of joints (default: 27)
- `block_size`: Block size for DropGraph (default: 41)
- `stride`: Stride for temporal convolution
- `residual`: Whether to use residual connection
- `attention`: Whether to use STC attention

### 2. HandAwareGCN (`ha_gcn.py`)

**Hand-aware Graph Convolution Layer** - a graph convolution layer that is aware of hand structure.

**Graph combination formula:**
```
A_combined = A + PA + SH × α + PH × β
```

Where:
- **A**: Body adjacency matrix (fixed, from skeleton layout)
- **PA**: Learnable body graph adjustments
- **SH**: Structured Hand Graph (fixed, based on physical hand structure)
- **PH**: Parameterized Hand Graph (learnable, initialized from SH)
- **α, β**: Learnable gating coefficients

**Forward process:**
1. Combine graphs: `A_combined = A + PA + SH × α + PH × β`
2. Apply graph convolution for each partition (K partitions)
3. Batch normalization
4. Residual connection (if needed)
5. ReLU activation

### 3. HandGraphConstructor (`hierarchy.py`)

Constructs hand graphs for HA-GCN.

#### Structured Hand Graph (SH)
- **Definition**: Fixed graph based on physical hand structure
- **Structure**:
  - Left hand: nodes 7-16 (11 nodes, wrist at 7)
  - Right hand: nodes 17-26 (11 nodes, wrist at 17)
  - Connections: Wrist → finger bases → finger joints
- **Characteristics**: Does not change during training

#### Parameterized Hand Graph (PH)
- **Definition**: Learnable graph, initialized from SH
- **Initialization**: `PH = SH + noise` (small noise to ensure learnability)
- **Characteristics**: All values are learnable parameters

**Hand connection structure:**
```
Left Hand (nodes 7-16):
  7 (wrist) → 8, 9, 10, 11, 12 (finger bases)
  8 → 9
  10 → 11
  12 → 13 → 14 → 15 → 16

Right Hand (nodes 17-26):
  17 (wrist) → 18, 19, 20, 21, 22 (finger bases)
  18 → 19
  20 → 21
  22 → 23 → 24 → 25 → 26
```

### 4. STCAttention (`attention.py`)

**Spatial-Temporal-Channel Attention Module** - 3D attention module.

#### Spatial Attention
- **Purpose**: Focus on important joints
- **Mechanism**: 
  - Average pooling along temporal dimension: (N, C, T, V) → (N, C, V)
  - 1D convolution: (N, C, V) → (N, 1, V)
  - Sigmoid activation → attention weights
  - Apply: `x = x * attention_weights + x`

#### Temporal Attention
- **Purpose**: Focus on important frames
- **Mechanism**:
  - Average pooling along spatial dimension: (N, C, T, V) → (N, C, T)
  - 1D convolution: (N, C, T) → (N, 1, T)
  - Sigmoid activation → attention weights
  - Apply: `x = x * attention_weights + x`

#### Channel Attention
- **Purpose**: Focus on important feature channels
- **Mechanism**:
  - Global average pooling: (N, C, T, V) → (N, C)
  - FC layers with reduction ratio: (N, C) → (N, C//2) → (N, C)
  - Sigmoid activation → attention weights
  - Apply: `x = x * attention_weights + x`

### 5. AdaptiveDropGraph (`adaptive_dropgraph.py`)

**Adaptive DropGraph** - intelligent regularization technique combining spatial and temporal dropout.

#### SpatialDropGraph
- **Purpose**: Drop joints based on importance
- **Mechanism**:
  1. Compute attention map: `input_abs = mean(|x|, dim=[C, T])` → (N, V)
  2. Normalize attention map
  3. Compute gamma: `γ = (1 - keep_prob) / (1 + 1.92)`
  4. Generate dropout mask using Bernoulli: `M_seed ~ Bernoulli(attention × γ)`
  5. Propagate through adjacency matrix: `M = M_seed @ A`
  6. Binarize mask and apply

#### TemporalDropGraph
- **Purpose**: Drop frames based on importance
- **Mechanism**:
  1. Compute attention map: `input_abs = mean(|x|, dim=[C, V])` → (N, T)
  2. Normalize attention map
  3. Compute gamma: `γ = (1 - keep_prob) / block_size`
  4. Generate dropout mask: `M ~ Bernoulli(attention × γ)`
  5. Max pooling to create block dropout: `Msum = MaxPool1d(M, kernel=block_size)`
  6. Apply mask

#### AdaptiveDropGraph
- **Combination**: `y = γ × SpatialDrop(x) + (1-γ) × x`
- **Then**: `y = δ × TemporalDrop(y) + (1-δ) × y`
- **γ, δ**: Learnable gating coefficients

## Model Configuration

### Initialization from Config

The model is created through factory function:

```python
from src.model.gcn_model_factory import create_model

model = create_model(config)  # if config["model"]["type"] = "ha_gcn"
```

Or directly:

```python
from src.model.gcn_model_factory import create_hagcn_from_config

model = create_hagcn_from_config(config, num_classes=None)
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
- `block_size`: Block size for DropGraph (default: 41)
- `use_attention`: Enable/disable STC attention (default: True)

### Direct Initialization

```python
from src.model.ha_gcn import HAGCN
import torch

# Create adjacency matrix A (K, V, V)
A = ...  # from SkeletonGraph or config

model = HAGCN(
    in_channels=3,
    num_class=100,
    num_nodes=27,
    num_person=1,
    window_size=64,
    A=A,
    block_size=41,
    use_attention=True
)
```

## Architecture Details

### 10 HA-GCN Blocks

| Block | Input Channels | Output Channels | Stride | Residual | Attention |
|-------|---------------|----------------|--------|----------|-----------|
| l1    | 3             | 64             | 1      | No       | Yes       |
| l2    | 64            | 64             | 1      | Yes      | Yes       |
| l3    | 64            | 64             | 1      | Yes      | Yes       |
| l4    | 64            | 64             | 1      | Yes      | Yes       |
| l5    | 64            | 128            | 2      | Yes      | Yes       |
| l6    | 128           | 128            | 1      | Yes      | Yes       |
| l7    | 128           | 128            | 1      | Yes      | Yes       |
| l8    | 128           | 256            | 2      | Yes      | Yes       |
| l9    | 256           | 256            | 1      | Yes      | Yes       |
| l10   | 256           | 256            | 1      | Yes      | Yes       |

**Note**: 
- Blocks l5 and l8 have stride=2 to reduce temporal size
- Block l1 has no residual connection
- All blocks use STC attention (if `use_attention=True`)

### Forward Pass

```python
def forward(self, x, keep_prob=0.9):
    # x: (N, C, T, V, M)
    N, C, T, V, M = x.size()
    
    # 1. Data Batch Normalization
    x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M*V*C, T)
    x = self.data_bn(x)
    x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(N*M, C, T, V)  # (N*M, C, T, V)
    
    # 2. HA-GCN Blocks
    x = self.l1(x, keep_prob=1.0)
    x = self.l2(x, keep_prob=1.0)
    x = self.l3(x, keep_prob=1.0)
    x = self.l4(x, keep_prob=1.0)
    x = self.l5(x, keep_prob=1.0)
    x = self.l6(x, keep_prob=1.0)
    x = self.l7(x, keep_prob=keep_prob)  # Start dropout
    x = self.l8(x, keep_prob=keep_prob)
    x = self.l9(x, keep_prob=keep_prob)
    x = self.l10(x, keep_prob=keep_prob)
    
    # 3. Global Average Pooling
    x = x.reshape(N, M, C_out, -1)  # (N, M, C, T*V)
    x = x.mean(dim=3).mean(dim=1)   # (N, C)
    
    # 4. Classification
    x = self.fc(x)  # (N, num_class)
    
    return x
```

## Advantages of HA-GCN

1. **Hand-aware Design**: Focuses on hand structure with SH and PH graphs
2. **Multi-graph Fusion**: Intelligently combines body graph (A) with hand graphs (SH, PH)
3. **Attention Mechanism**: STC attention helps focus on important joints, frames, and channels
4. **Adaptive Regularization**: Adaptive DropGraph reduces overfitting and increases generalization ability
5. **Learnable Hand Graph**: PH allows the model to learn optimal hand connections for the task

## When to Use HA-GCN

- When you want to **strongly exploit hand structure** (fine-grained hand pose)
- When sequences are relatively long and need attention across space/time/channels
- When DropGraph is needed for regularization (reducing overfitting)
- When you want the model to automatically learn optimal hand connections (via PH)

## File Structure

```
ha_gcn/
├── __init__.py              # Exports main classes
├── ha_gcn_model.py          # HAGCN class - main model
├── ha_gcn.py                # HAGCNBlock, HandAwareGCN, TemporalConv
├── hierarchy.py             # HandGraphConstructor - creates SH and PH
├── attention.py             # STCAttention - spatial-temporal-channel attention
├── adaptive_dropgraph.py    # AdaptiveDropGraph, SpatialDropGraph, TemporalDropGraph
└── README.md                # This documentation
```

## Dependencies

- `torch`: PyTorch framework
- `numpy`: Array processing
- `src.data.gcn.graph_constructor.SkeletonGraph`: To create body adjacency matrix

## References

The model is designed based on ideas from:
- ST-GCN (Spatial Temporal Graph Convolutional Networks)
- Hand-aware graph structures for sign language recognition
- Adaptive DropGraph regularization techniques
- Multi-dimensional attention mechanisms