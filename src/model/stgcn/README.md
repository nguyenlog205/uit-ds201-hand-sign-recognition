# ST-GCN: Spatial Temporal Graph Convolutional Networks

## Tổng quan

**ST-GCN (Spatial Temporal Graph Convolutional Networks)** là mô hình GCN chuẩn cho nhận dạng hành động và ngôn ngữ ký hiệu, học đồng thời **quan hệ không gian** (graph convolution trên joints) và **quan hệ thời gian** (temporal convolution trên frames).

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
Data Batch Normalization (optional)
    ↓
Initial GCN + TCN (gcn0 + tcn0)
    ↓
9 × ST-GCN Blocks (backbone)
    ├─ Block 1-3: 64 channels
    ├─ Block 4-6: 128 channels (stride=2 ở block 4)
    └─ Block 7-9: 256 channels (stride=2 ở block 7)
    ↓
Spatial Pooling (over joints V)
    ↓
Temporal Pooling (over frames T)
    ↓
1D Convolution (FCN)
    ↓
Output (N, num_class)
```

## Các thành phần chính

### 1. STGCNBlock (`stgcn.py`)

Mỗi block ST-GCN bao gồm 3 thành phần:

```
Input
  ↓
[SpatialGCN] → Graph Convolution trên joints
  ↓
[TemporalConv] → 1D Convolution trên frames
  ↓
[Residual Connection]
  ↓
Output
```

**Tham số:**
- `in_channels`, `out_channels`: Số kênh đầu vào/ra
- `A`: Adjacency matrix (K, V, V) với K partitions
- `kernel_size`: Kích thước kernel cho temporal conv (mặc định: 9)
- `stride`: Stride cho temporal convolution
- `dropout`: Dropout rate (mặc định: 0.5)
- `use_local_bn`: Sử dụng local batch norm per node
- `mask_learning`: Học mask để reweight adjacency matrix

**Quy trình forward:**
1. Spatial GCN: `x = gcn(x)` → Graph convolution trên joints
2. Temporal Conv: `x = tcn(x)` → 1D convolution trên frames
3. Residual: `x = x + downsample(input)` → Kết nối tắt

### 2. SpatialGCN (`spatial_gcn.py`)

**Spatial Graph Convolution** - lớp convolution đồ thị trên không gian joints.

**Spatial Partitioning Strategy:**
Đồ thị được chia thành **K partitions** (thường K=3):
- **Partition 0**: Root node (chính nó)
- **Partition 1**: Centripetal group (neighbors gần center của skeleton)
- **Partition 2**: Centrifugal group (neighbors xa center của skeleton)

**Công thức:**
```
y = Σ(k=0 to K-1) Conv(x @ A[k])
```

Trong đó:
- **A[k]**: Adjacency matrix cho partition k (V, V)
- **x @ A[k]**: Matrix multiplication để aggregate neighbors
- **Conv**: 2D convolution (kernel_size, 1) trên từng partition
- **Σ**: Tổng hợp kết quả từ tất cả partitions

**Tùy chọn nâng cao:**
- **Mask Learning**: Học mask để reweight adjacency matrix
  ```python
  A_learned = A * mask  # mask là learnable parameter
  ```
- **Local Batch Norm**: Mỗi node có BN parameters riêng
  - Global BN: `(N, C, T, V)` → BN2D
  - Local BN: `(N, C, T, V)` → `(N, C*V, T)` → BN1D → reshape back

**Quy trình forward:**
1. Áp dụng mask (nếu `mask_learning=True`): `A = A * mask`
2. Với mỗi partition k:
   - Reshape: `(N, C, T, V)` → `(N*T*C, V)`
   - Matrix multiplication: `xa = x @ A[k]` → `(N*T*C, V)`
   - Reshape back: `(N*T*C, V)` → `(N, C, T, V)`
   - Convolution: `conv_k(xa)`
3. Tổng hợp: `y = Σ conv_k(xa)`
4. Batch normalization (global hoặc local)
5. ReLU activation

### 3. TemporalConv (`temporal_conv.py`)

**Temporal Convolution** - lớp convolution 1D theo chiều thời gian.

**Công thức:**
```
y = ReLU(BN(Conv1D(x)))
```

**Chi tiết:**
- **Conv2D**: `kernel_size=(kernel_size, 1)`, `stride=(stride, 1)`
  - Convolve theo chiều T, giữ nguyên chiều V
- **BatchNorm2D**: Normalize theo batch
- **Dropout**: Optional dropout trước convolution
- **ReLU**: Activation function

**Quy trình forward:**
1. Dropout (nếu `dropout > 0`)
2. 2D Convolution: `(N, C, T, V)` → `(N, C_out, T_out, V)`
3. Batch Normalization
4. ReLU activation

### 4. STGCN Model (`stgcn.py`)

**Spatial Temporal Graph Convolutional Network** - model chính.

#### Cấu trúc

1. **Data Batch Normalization** (optional)
   - Normalize input data trước khi vào network
   - Hỗ trợ multi-person: `(N, M*V*C, T)` hoặc single-person: `(N*M, V*C, T)`

2. **Initial Layers**
   - `gcn0`: Initial spatial GCN (3 → 64 channels)
   - `tcn0`: Initial temporal conv

3. **Backbone** (9 ST-GCN Blocks)
   - Default configuration: `DEFAULT_BACKBONE`
   - Có thể custom qua `backbone_config`

4. **Classification Head**
   - Spatial pooling: Average pool over joints `(N, C, T, V)` → `(N, C, T, 1)`
   - Person pooling: Average over people (nếu M > 1)
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

## Cấu hình mô hình

### Khởi tạo từ config

Mô hình được tạo thông qua factory function:

```python
from src.model.gcn_model_factory import create_model

model = create_model(config)  # nếu config["model"]["type"] = "stgcn"
```

Hoặc trực tiếp:

```python
from src.model.gcn_model_factory import create_stgcn_from_config

model = create_stgcn_from_config(config, num_classes=None)
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
- `use_data_bn`: Bật/tắt data batch normalization (mặc định: True)
- `mask_learning`: Học mask để reweight adjacency (mặc định: False)
- `use_local_bn`: Sử dụng local batch norm per node (mặc định: False)
- `temporal_kernel_size`: Kích thước kernel cho temporal conv (mặc định: 9)
- `dropout`: Dropout rate (mặc định: 0.5)
- `backbone_config`: Custom backbone config (list of `(in_c, out_c, stride)`)

### Khởi tạo trực tiếp

```python
from src.model.stgcn import STGCN
import torch

# Tạo adjacency matrix A (K, V, V)
A = ...  # từ SkeletonGraph hoặc config

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

## Chi tiết kiến trúc

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

ST-GCN sử dụng **spatial partitioning** để chia neighbors của mỗi node thành các nhóm:

1. **Root partition**: Node chính nó
2. **Centripetal partition**: Neighbors gần center của skeleton hơn
3. **Centrifugal partition**: Neighbors xa center của skeleton hơn

Điều này giúp mô hình học được các pattern khác nhau dựa trên hướng của kết nối trong skeleton.

### Mask Learning

Khi `mask_learning=True`, mô hình học một mask để reweight adjacency matrix:

```python
A_learned = A * mask  # mask là learnable parameter (K, V, V)
```

Điều này cho phép mô hình tự động điều chỉnh tầm quan trọng của các kết nối trong đồ thị.

### Local Batch Normalization

Khi `use_local_bn=True`, mỗi node có batch normalization parameters riêng:

- **Global BN**: `(N, C, T, V)` → BN2D → Tất cả nodes share parameters
- **Local BN**: `(N, C, T, V)` → `(N, C*V, T)` → BN1D → Mỗi node có parameters riêng

Local BN có thể hữu ích khi các nodes có distribution khác nhau.

## Ưu điểm của ST-GCN

1. **Đơn giản và dễ hiểu**: Kiến trúc rõ ràng, dễ debug và customize
2. **Hiệu quả**: Kết hợp spatial và temporal convolution một cách hiệu quả
3. **Linh hoạt**: Hỗ trợ nhiều tùy chọn (mask learning, local BN, custom backbone)
4. **Chuẩn mực**: Implementation theo paper gốc, dễ so sánh với các nghiên cứu khác
5. **Multi-person support**: Hỗ trợ xử lý nhiều người trong cùng một frame

## Khi nào sử dụng ST-GCN

- Khi có **chuỗi pose (keypoints)** và muốn dùng GCN chuẩn, đơn giản, dễ debug
- Khi không cần cấu trúc bàn tay quá phức tạp như HA-GCN
- Khi muốn mô hình hóa **structure cơ thể + hai tay** qua adjacency matrix
- Khi cần bắt temporal pattern bằng convolution 1D
- Khi muốn baseline model để so sánh với các mô hình phức tạp hơn

## File Structure

```
stgcn/
├── __init__.py              # Exports các class chính
├── stgcn.py                 # STGCN, STGCNBlock - model chính
├── spatial_gcn.py           # SpatialGCN, UnitGCN - graph convolution
├── temporal_conv.py         # TemporalConv, Unit2D - temporal convolution
└── README.md                # Tài liệu này
```

## Dependencies

- `torch`: PyTorch framework
- `torch.nn.functional`: F.avg_pool2d, F.avg_pool1d
- `src.data.gcn.graph_constructor.SkeletonGraph`: Để tạo adjacency matrix

## Tham khảo

- **Paper gốc**: [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)
- **Implementation**: Dựa trên codebase ST-GCN với adaptations cho sign language recognition

## Lưu ý

- Adjacency matrix A phải có shape `(K, V, V)` với K là số partitions (thường K=3)
- Nếu A là 2D `(V, V)`, cần expand thành `(1, V, V)` hoặc stack thành `(K, V, V)`
- Backbone configuration có thể custom nhưng cần đảm bảo tính nhất quán về channels và strides