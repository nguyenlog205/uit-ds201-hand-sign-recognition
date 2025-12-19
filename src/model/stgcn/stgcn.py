"""
ST-GCN: Spatial Temporal Graph Convolutional Networks
Paper: https://arxiv.org/abs/1801.07455

Implementation for Sign Language Recognition with 27 keypoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .spatial_gcn import SpatialGCN, unit_gcn
from .temporal_conv import TemporalConv, Unit2D


# Default backbone configuration (for large datasets)
# Format: (in_channels, out_channels, stride)
DEFAULT_BACKBONE_LARGE = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
    (256, 256, 1),
]

# Lightweight backbone for small datasets (< 100 samples)
DEFAULT_BACKBONE_SMALL = [
    (64, 64, 1),
    (64, 64, 1),
    (64, 128, 2),
    (128, 128, 1),
    (128, 256, 2),
    (256, 256, 1),
]

# Default uses large backbone (can be overridden in config)
DEFAULT_BACKBONE = DEFAULT_BACKBONE_LARGE


class STGCNBlock(nn.Module):
    """
    ST-GCN Block: Spatial GCN + Temporal Conv
    
    Combines spatial graph convolution and temporal convolution
    with residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.5,
        use_local_bn: bool = False,
        mask_learning: bool = False,
    ):
        super(STGCNBlock, self).__init__()
        
        # Spatial GCN
        self.gcn = unit_gcn(
            in_channels,
            out_channels,
            A,
            use_local_bn=use_local_bn,
            mask_learning=mask_learning,
        )
        
        # Temporal Conv
        self.tcn = Unit2D(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            stride=stride,
        )
        
        # Residual connection
        if (in_channels != out_channels) or (stride != 1):
            self.downsample = Unit2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.downsample = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (N, C, T, V)
        
        Returns:
            Output tensor (N, C_out, T, V)
        """
        # Spatial GCN -> Temporal Conv
        out = self.tcn(self.gcn(x))
        
        # Residual connection
        if self.downsample is not None:
            x = self.downsample(x)
        
        return out + x


class STGCN(nn.Module):
    """
    Spatial Temporal Graph Convolutional Network
    
    Input shape: (N, C, T, V, M)
        N = batch size
        C = number of input channels (typically 3: x, y, confidence)
        T = temporal frames (sequence length)
        V = number of joints/nodes
        M = number of people (typically 1)
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_class: Number of output classes
        num_nodes: Number of joints (default: 27 for sign language)
        num_person: Number of people (default: 1)
        window_size: Temporal window size (sequence length)
        use_data_bn: If True, apply batch norm to input data
        backbone_config: Backbone configuration (list of (in_c, out_c, stride))
        A: Adjacency matrix (K, V, V) where K is number of partitions
        mask_learning: If True, learn mask to reweight adjacency matrix
        use_local_bn: If True, use local batch norm per node
        temporal_kernel_size: Temporal convolution kernel size (default: 9)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_class: int = 100,
        num_nodes: int = 27,
        num_person: int = 1,
        window_size: int = 64,
        use_data_bn: bool = True,
        backbone_config: Optional[list] = None,
        A: Optional[torch.Tensor] = None,
        mask_learning: bool = False,
        use_local_bn: bool = False,
        temporal_kernel_size: int = 9,
        dropout: float = 0.5,
    ):
        super(STGCN, self).__init__()
        
        if A is None:
            raise ValueError("Adjacency matrix A must be provided")
        
        # Store parameters
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.num_person = num_person
        self.window_size = window_size
        self.use_data_bn = use_data_bn
        
        # Register adjacency matrix
        self.register_buffer('A', A.float())
        
        # Data batch normalization
        if use_data_bn:
            if num_person > 1:
                # Different people share BN parameters
                self.data_bn = nn.BatchNorm1d(in_channels * num_nodes * num_person)
            else:
                self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        # Backbone configuration
        if backbone_config is None:
            backbone_config = DEFAULT_BACKBONE
        
        # Build backbone
        self.backbone = nn.ModuleList([
            STGCNBlock(
                in_c,
                out_c,
                self.A,
                kernel_size=temporal_kernel_size,
                stride=stride,
                dropout=dropout,
                use_local_bn=use_local_bn,
                mask_learning=mask_learning,
            )
            for in_c, out_c, stride in backbone_config
        ])
        
        # Initial GCN and TCN layers
        backbone_in_c = backbone_config[0][0]
        self.gcn0 = unit_gcn(
            in_channels,
            backbone_in_c,
            self.A,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn,
        )
        self.tcn0 = Unit2D(backbone_in_c, backbone_in_c, kernel_size=temporal_kernel_size)
        
        # Calculate output dimensions after backbone
        backbone_out_c = backbone_config[-1][1]
        backbone_out_t = window_size
        for _, _, stride in backbone_config:
            if backbone_out_t % stride == 0:
                backbone_out_t = backbone_out_t // stride
            else:
                backbone_out_t = backbone_out_t // stride + 1
        
        # Classification head
        self.person_bn = nn.BatchNorm1d(backbone_out_c)
        self.gap_size = backbone_out_t
        self.fcn = nn.Conv1d(backbone_out_c, num_class, kernel_size=1)
        
        # Initialize FCN
        nn.init.normal_(self.fcn.weight, 0, 0.01)
        if self.fcn.bias is not None:
            nn.init.constant_(self.fcn.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C, T, V, M)
                N = batch size
                C = input channels
                T = temporal frames
                V = number of joints
                M = number of people
        
        Returns:
            Output logits of shape (N, num_class)
        """
        N, C, T, V, M = x.size()
        
        # Data batch normalization
        if self.use_data_bn:
            if M > 1:
                # Reshape: (N, C, T, V, M) -> (N, M*V*C, T)
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                # Reshape: (N, C, T, V, M) -> (N*M, V*C, T)
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            
            x = self.data_bn(x)
            
            # Reshape back: (N*M, V*C, T) -> (N*M, C, T, V)
            if M > 1:
                x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                    N * M, C, T, V)
            else:
                x = x.view(N * M, V, C, T).permute(0, 2, 3, 1).contiguous()
        else:
            # Reshape: (N, C, T, V, M) -> (N*M, C, T, V)
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Initial layers
        x = self.gcn0(x)  # (N*M, C, T, V)
        x = self.tcn0(x)  # (N*M, C, T, V)
        
        # Backbone
        for block in self.backbone:
            x = block(x)  # (N*M, C, T, V)
        
        # Spatial pooling (pool over joints)
        x = F.avg_pool2d(x, kernel_size=(1, V))  # (N*M, C, T, 1)
        x = x.squeeze(-1)  # (N*M, C, T)
        
        # Person pooling (if multiple people)
        if M > 1:
            x = x.view(N, M, x.size(1), x.size(2))  # (N, M, C, T)
            x = x.mean(dim=1)  # (N, C, T)
        
        # Temporal pooling
        x = F.avg_pool1d(x, kernel_size=x.size(2))  # (N, C, 1)
        x = x.squeeze(-1)  # (N, C)
        
        # Classification
        x = x.unsqueeze(-1)  # (N, C, 1)
        x = self.fcn(x)  # (N, num_class, 1)
        x = F.avg_pool1d(x, kernel_size=x.size(2))  # (N, num_class, 1)
        x = x.squeeze(-1)  # (N, num_class)
        
        return x
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get learned attention/mask weights if mask_learning is enabled
        
        Returns:
            Mask weights of shape (K, V, V) or None
        """
        if hasattr(self, 'gcn0') and hasattr(self.gcn0.gcn, 'mask'):
            return self.gcn0.gcn.mask
        return None


def create_stgcn_model(
    num_class: int,
    num_nodes: int = 27,
    window_size: int = 64,
    in_channels: int = 3,
    A: Optional[torch.Tensor] = None,
    **kwargs
) -> STGCN:
    """
    Factory function to create ST-GCN model
    
    Args:
        num_class: Number of output classes
        num_nodes: Number of joints (default: 27 for sign language)
        window_size: Temporal window size
        in_channels: Number of input channels
        A: Adjacency matrix (K, V, V)
        **kwargs: Additional arguments for STGCN
    
    Returns:
        STGCN model instance
    """
    if A is None:
        raise ValueError("Adjacency matrix A must be provided")
    
    model = STGCN(
        in_channels=in_channels,
        num_class=num_class,
        num_nodes=num_nodes,
        window_size=window_size,
        A=A,
        **kwargs
    )
    
    return model