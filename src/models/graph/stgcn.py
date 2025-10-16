import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    """
    Một block ST-GCN: gồm một bước spatial graph convolution (GCN) 
    và một bước temporal convolution (TCN), với skip/residual connection.
    Input/Output tensor shape convention trong block:
      - X  có shape (N, C, T, V)
        N: batch size
        C: channels (features per node, ví dụ 3 = x,y,z)
        T: số frame theo time dimension
        V: số node (joints) trong mỗi frame
    """

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        # Lưu ma trận kề (A) tĩnh được truyền vào (kiểu torch.Tensor, shape VxV)
        # Lưu ý: trong model chính bạn đã register_buffer('A_tensor', A) — tốt hơn là dùng buffer đó.
        self.A = A

        # "GCN" được biểu diễn bằng conv2d với kernel (1, V).
        # Ý tưởng: sau khi aggregate theo A (qua einsum), ta dùng 1xV conv để mix channel.
        # Tuy nhiên kernel (1, A.size(0)) ở đây có ý nghĩa phụ thuộc cách bạn biểu diễn tensor sau einsum.
        # Conv2d nhận input (N, C_in, T, V), kernel (kT, kV).
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, A.size(0)))

        # Temporal convolution block: (temporal conv) + BN + ReLU + dropout.
        # kernel_size=(9,1) chỉ conv dọc theo trục thời gian (T), không làm thay đổi V.
        # stride=(stride,1) cho phép giảm sampling rate theo thời gian (temporal downsample).
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), stride=(stride,1), padding=(4,0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5)
        )

        # Residual connection: nếu in/out channel khác nhau, dùng conv 1x1 để biến đổi residual.
        # stride phải match với tcn stride để kích thước T align khi downsampling.
        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=(stride,1)) if residual else None

    def forward(self, X):
        # X: (N, C, T, V)
        # Lưu phần residual (skip connection)
        res = X if self.residual is None else self.residual(X)

        # --- Spatial graph convolution (aggregation theo adjacency) ---
        # torch.einsum('nctv, vw -> nctw', X, self.A)
        # Ý nghĩa: với mỗi batch n, channel c, time t:
        #   new_feature_at_node_w = sum_over_v ( X[n,c,t,v] * A[v,w] )
        # Tức là: với mỗi node w, ta aggregate feature từ các neighbor v theo A[v,w].
        # Kết quả shape vẫn (N, C, T, V) nhưng nội dung đã là aggregated features.
        X = torch.einsum('nctv, vw -> nctw', X, self.A)

        # Sau khi aggregate, áp dụng conv 2D để mix thông tin giữa các channel
        # self.gcn kỳ vọng input (N, C_in, T, V). Kết quả shape: (N, out_channels, T, ?)
        # Lưu ý: kernel width = A.size(0) có thể làm "thu hẹp" hoặc che dấu dimension V,
        # vì vậy cần kiểm tra kỹ kích thước output với dataset thực tế.
        X = self.gcn(X)

        # Temporal conv (TCN) xử lý theo trục thời gian, rồi cộng residual
        X = self.tcn(X) + res

        # Activation cuối cùng
        return F.relu(X)


class STGCN(nn.Module):
    """
    Mạng ST-GCN tối giản cho bài toán: (N, C, T, V, M) --> class logits
    - N: batch size
    - C: channels (ví dụ 3 = x,y,z)
    - T: frames length
    - V: number of joints
    - M: number of persons (thường 1 hoặc 2)
    Output: logits shape (N, num_class)
    """

    def __init__(self, num_class, num_point, num_person=1, num_channel=3):
        super().__init__()
        # Xây adjacency mặc định (dummy): self-loop + neighbor chain
        # build_adjacency trả về tensor VxV
        self.A = self.build_adjacency(num_point)
        # Đăng ký buffer để ma trận A di chuyển theo thiết bị model (cpu/gpu)
        # (khi dùng model.to(device), buffer sẽ tự động chuyển)
        self.register_buffer('A_tensor', self.A)

        # BatchNorm cho dữ liệu đầu vào: flatten M*C*V làm channels để BN 1D.
        # Cách dùng: ta reshape sau đó BN và reshape lại về (N, M*C, T, V)
        self.data_bn = nn.BatchNorm1d(num_person * num_channel * num_point)

        # Ba block ST-GCN lần lượt tăng channel (feature) và có thể giảm temporal dimension bằng stride.
        self.layer1 = STGCNBlock(num_channel, 64, self.A)        # giữ T
        self.layer2 = STGCNBlock(64, 128, self.A, stride=2)      # giảm T bằng 2
        self.layer3 = STGCNBlock(128, 256, self.A, stride=2)     # giảm T tiếp

        # Fully connected cuối để phân lớp. Input dim = số channel output cuối cùng (256).
        self.fc = nn.Linear(256, num_class)

    def build_adjacency(self, num_point):
        """
        Tạo ma trận kề đơn giản: self-loop + chuỗi neighbor. 
        Trong thực tế, bạn cần dùng adjacency dựa trên cấu trúc xương (skeleton).
        Trả về tensor float shape (V, V).
        """
        A = torch.eye(num_point)
        for i in range(num_point-1):
            A[i, i+1] = A[i+1, i] = 1
        return A

    def forward(self, X):
        """
        X input shape: (N, C, T, V, M)
        Chuyển về dạng (N, M*C, T, V) để xử lý:
        - permute(0,4,1,2,3) đưa M lên ngay sau batch
        - view -> kết hợp M và C thành 1 chiều channel (M*C)
        Sau đó áp dụng data_bn: BatchNorm1d trên dimension (M*C*V)
        """
        N, C, T, V, M = X.shape

        # 1) chuyển (N, C, T, V, M) -> (N, M, C, T, V), rồi gộp M và C
        X = X.permute(0, 4, 1, 2, 3).contiguous().view(N, M * C, T, V)

        # 2) BatchNorm: cần reshape thành (N, M*C, T*V) -> BN1d hoạt động trên channel dim
        X = self.data_bn(X.view(N, -1, T*V)).view(N, M*C, T, V)

        # 3) Áp dụng chuỗi ST-GCN block
        # LƯU Ý: Các block của bạn dựa vào self.A (tensor) — tốt nhất là dùng self.A_tensor
        # để đảm bảo ma trận kề nằm trên device đúng (cpu/gpu). Hiện tại self.A là một tensor cục bộ.
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)

        # 4) Global Average Pooling: trung bình theo time (T) và nodes (V)
        # Kết quả shape: (N, channels_out)
        X = X.mean(dim=[2,3])

        # 5) Fully connected -> logits cho mỗi class
        return self.fc(X)
