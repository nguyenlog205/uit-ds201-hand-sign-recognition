# HA-GCN: Hand-aware Graph Convolutional Network

## Tổng quan

**HA-GCN (Hand-aware Graph Convolutional Network)** là mô hình GCN chuyên biệt cho nhận dạng ngôn ngữ ký hiệu, được thiết kế để khai thác sâu cấu trúc bàn tay trong dữ liệu pose. Mô hình kết hợp đồ thị cơ thể (body graph) với đồ thị bàn tay (hand graphs) để tăng cường khả năng học các đặc trưng tinh tế của cử chỉ tay.

## Kiến trúc tổng quát

### Input Format
Mô hình nhận input tensor với shape: **(N, C, T, V, M)**
- **N**: Batch size
- **C**: Số kênh (default là 3: x, y, confidence)
- **T**: Số frame trong chuỗi thời gian
- **V**: Số khớp/joints (mặc định 27 cho Mediapipe/Sign language)
- **M**: Số người (default là 1)

### Kiến trúc Pipeline

```
Input (N, C, T, V, M)
    ↓
Data Batch Normalization
    ↓
10 × HA-GCN Blocks (layer_1 → layer_10)
    ├─ Block 1-4: 64 channels
    ├─ Block 5-7: 128 channels (stride=2 ở block 5)
    └─ Block 8-10: 256 channels (stride=2 ở block 8)
    ↓
Global Average Pooling (T, V)
    ↓
Fully-Connected Layer
    ↓
Output (N, num_class)
```

## Các thành phần chính

### 1. HAGCNBlock (`ha_gcn.py`)

Mỗi block HA-GCN bao gồm 4 thành phần chính:

```
Input
  ↓
[HandAwareGCN] → Graph Convolution với A + SH + PH
  ↓
[STCAttention] → Spatial-Temporal-Channel Attention (optional)
  ↓
[TemporalConv] → Temporal Convolution + Adaptive DropGraph
  ↓
[Residual Connection]
  ↓
Output
```

**Tham số:**
- `in_channels`, `out_channels`: Số kênh đầu vào/ra
- `A`: Body adjacency matrix (K, V, V)
- `SH`: Structured Hand Graph (K, V, V) - cố định
- `PH`: Parameterized Hand Graph (K, V, V) - learnable
- `num_point`: Số joints (mặc định 27)
- `block_size`: Kích thước block cho DropGraph (mặc định 41)
- `stride`: Stride cho temporal convolution
- `residual`: Có sử dụng residual connection không
- `attention`: Có sử dụng STC attention không

### 2. HandAwareGCN (`ha_gcn.py`)

**Hand-aware Graph Convolution Layer** - lớp convolution đồ thị nhận biết bàn tay.

**Công thức kết hợp đồ thị:**
```
A_combined = A + PA + SH × α + PH × β
```

Trong đó:
- **A**: Body adjacency matrix (cố định, từ skeleton layout)
- **PA**: Learnable body graph adjustments
- **SH**: Structured Hand Graph (cố định, dựa trên cấu trúc vật lý của bàn tay)
- **PH**: Parameterized Hand Graph (learnable, khởi tạo từ SH)
- **α, β**: Learnable gating coefficients

**Quy trình forward:**
1. Kết hợp các đồ thị: `A_combined = A + PA + SH × α + PH × β`
2. Áp dụng graph convolution cho từng partition (K partitions)
3. Batch normalization
4. Residual connection (nếu cần)
5. ReLU activation

### 3. HandGraphConstructor (`hierarchy.py`)

Xây dựng đồ thị bàn tay cho HA-GCN.

#### Structured Hand Graph (SH)
- **Định nghĩa**: Đồ thị cố định dựa trên cấu trúc vật lý của bàn tay
- **Cấu trúc**:
  - Left hand: nodes 7-16 (11 nodes, wrist tại 7)
  - Right hand: nodes 17-26 (11 nodes, wrist tại 17)
  - Kết nối: Wrist → finger bases → finger joints
- **Đặc điểm**: Không thay đổi trong quá trình training

#### Parameterized Hand Graph (PH)
- **Định nghĩa**: Đồ thị learnable, khởi tạo từ SH
- **Khởi tạo**: `PH = SH + noise` (noise nhỏ để đảm bảo learnability)
- **Đặc điểm**: Tất cả các giá trị đều là tham số có thể học

**Cấu trúc kết nối bàn tay:**
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

**Spatial-Temporal-Channel Attention Module** - module attention 3 chiều.

#### Spatial Attention
- **Mục đích**: Tập trung vào các joints quan trọng
- **Cơ chế**: 
  - Average pooling theo chiều thời gian: (N, C, T, V) → (N, C, V)
  - 1D convolution: (N, C, V) → (N, 1, V)
  - Sigmoid activation → attention weights
  - Áp dụng: `x = x * attention_weights + x`

#### Temporal Attention
- **Mục đích**: Tập trung vào các frames quan trọng
- **Cơ chế**:
  - Average pooling theo chiều không gian: (N, C, T, V) → (N, C, T)
  - 1D convolution: (N, C, T) → (N, 1, T)
  - Sigmoid activation → attention weights
  - Áp dụng: `x = x * attention_weights + x`

#### Channel Attention
- **Mục đích**: Tập trung vào các feature channels quan trọng
- **Cơ chế**:
  - Global average pooling: (N, C, T, V) → (N, C)
  - FC layers với reduction ratio: (N, C) → (N, C//2) → (N, C)
  - Sigmoid activation → attention weights
  - Áp dụng: `x = x * attention_weights + x`

### 5. AdaptiveDropGraph (`adaptive_dropgraph.py`)

**Adaptive DropGraph** - kỹ thuật regularization thông minh kết hợp spatial và temporal dropout.

#### SpatialDropGraph
- **Mục đích**: Drop các joints dựa trên tầm quan trọng
- **Cơ chế**:
  1. Tính attention map: `input_abs = mean(|x|, dim=[C, T])` → (N, V)
  2. Normalize attention map
  3. Tính gamma: `γ = (1 - keep_prob) / (1 + 1.92)`
  4. Tạo dropout mask bằng Bernoulli: `M_seed ~ Bernoulli(attention × γ)`
  5. Propagate qua adjacency matrix: `M = M_seed @ A`
  6. Binarize mask và áp dụng

#### TemporalDropGraph
- **Mục đích**: Drop các frames dựa trên tầm quan trọng
- **Cơ chế**:
  1. Tính attention map: `input_abs = mean(|x|, dim=[C, V])` → (N, T)
  2. Normalize attention map
  3. Tính gamma: `γ = (1 - keep_prob) / block_size`
  4. Tạo dropout mask: `M ~ Bernoulli(attention × γ)`
  5. Max pooling để tạo block dropout: `Msum = MaxPool1d(M, kernel=block_size)`
  6. Áp dụng mask

#### AdaptiveDropGraph
- **Kết hợp**: `y = γ × SpatialDrop(x) + (1-γ) × x`
- **Sau đó**: `y = δ × TemporalDrop(y) + (1-δ) × y`
- **γ, δ**: Learnable gating coefficients

## Cấu hình mô hình

### Khởi tạo từ config

Mô hình được tạo thông qua factory function:

```python
from src.model.gcn_model_factory import create_model

model = create_model(config)  # nếu config["model"]["type"] = "ha_gcn"
```

Hoặc trực tiếp:

```python
from src.model.gcn_model_factory import create_hagcn_from_config

model = create_hagcn_from_config(config, num_classes=None)
```

### Các tham số cấu hình

Trong `config["model"]`:

- `in_channels`: Số kênh đầu vào (mặc định: 3)
- `num_nodes`: Số joints (mặc định: 27)
- `num_person`: Số người (mặc định: 1)
- `window_size`: Kích thước cửa sổ thời gian
- `num_class`: Số lớp phân loại
- `skeleton_layout`: Layout skeleton (ví dụ: `"mediapipe_27"`)
- `adjacency_strategy`: Chiến lược tạo adjacency matrix (ví dụ: `"spatial"`)
- `block_size`: Kích thước block cho DropGraph (mặc định: 41)
- `use_attention`: Bật/tắt STC attention (mặc định: True)

### Khởi tạo trực tiếp

```python
from src.model.ha_gcn import HAGCN
import torch

# Tạo adjacency matrix A (K, V, V)
A = ...  # từ SkeletonGraph hoặc config

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

## Chi tiết kiến trúc

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

**Lưu ý**: 
- Block l5 và l8 có stride=2 để giảm kích thước temporal
- Block l1 không có residual connection
- Tất cả blocks đều sử dụng STC attention (nếu `use_attention=True`)

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
    x = self.l7(x, keep_prob=keep_prob)  # Bắt đầu dropout
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

## Ưu điểm của HA-GCN

1. **Hand-aware Design**: Tập trung vào cấu trúc bàn tay với SH và PH graphs
2. **Multi-graph Fusion**: Kết hợp body graph (A) với hand graphs (SH, PH) một cách thông minh
3. **Attention Mechanism**: STC attention giúp tập trung vào joints, frames và channels quan trọng
4. **Adaptive Regularization**: Adaptive DropGraph giảm overfitting và tăng khả năng generalization
5. **Learnable Hand Graph**: PH cho phép mô hình học các kết nối tay tối ưu cho task

## Khi nào sử dụng HA-GCN

- Khi muốn **khai thác mạnh vào cấu trúc bàn tay** (fine-grained hand pose)
- Khi sequence tương đối dài và cần attention theo không gian/thời gian/kênh
- Khi cần DropGraph để regularization (giảm overfitting)
- Khi muốn mô hình tự học các kết nối tay tối ưu (qua PH)

## File Structure

```
ha_gcn/
├── __init__.py              # Exports các class chính
├── ha_gcn_model.py          # HAGCN class - model chính
├── ha_gcn.py                # HAGCNBlock, HandAwareGCN, TemporalConv
├── hierarchy.py             # HandGraphConstructor - tạo SH và PH
├── attention.py             # STCAttention - spatial-temporal-channel attention
├── adaptive_dropgraph.py    # AdaptiveDropGraph, SpatialDropGraph, TemporalDropGraph
└── README.md                # Tài liệu này
```

## Dependencies

- `torch`: PyTorch framework
- `numpy`: Xử lý arrays
- `src.data.gcn.graph_constructor.SkeletonGraph`: Để tạo body adjacency matrix

## Tham khảo

Mô hình được thiết kế dựa trên các ý tưởng từ:
- ST-GCN (Spatial Temporal Graph Convolutional Networks)
- Hand-aware graph structures cho sign language recognition
- Adaptive DropGraph regularization techniques
- Multi-dimensional attention mechanisms