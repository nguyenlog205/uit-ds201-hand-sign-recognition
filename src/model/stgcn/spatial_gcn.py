"""
Spatial Graph Convolution
Spatial graph convolution layers for ST-GCN with partitioning strategy
Paper: https://arxiv.org/abs/1801.07455
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


def conv_init(module):
    """Initialize convolution weights using He initialization"""
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))
    if module.bias is not None:
        module.bias.data.zero_()


class SpatialGCN(nn.Module):
    """
    Spatial Graph Convolution Unit
    
    Implements graph convolution with spatial partitioning strategy:
    - Root node: The node itself
    - Centripetal group: Neighbors closer to skeleton center
    - Centrifugal group: Neighbors farther from skeleton center
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        A: Adjacency matrix (K, V, V) where K is number of partitions
        kernel_size: Kernel size for convolution (default: 1)
        stride: Stride for convolution (default: 1)
        use_local_bn: If True, use local batch norm per node
        mask_learning: If True, learn mask to reweight adjacency matrix
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        kernel_size: int = 1,
        stride: int = 1,
        use_local_bn: bool = False,
        mask_learning: bool = False,
    ):
        super(SpatialGCN, self).__init__()
        
        # Number of nodes
        self.V = A.size()[-1]
        
        # Adjacency matrices (K, V, V) where K is number of partitions
        # K=3 for spatial partitioning: [self, centripetal, centrifugal]
        self.register_buffer('A', A.clone().detach().view(-1, self.V, self.V))
        
        # Number of partitions
        self.num_A = self.A.size(0)
        
        # Number of channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Options
        self.use_local_bn = use_local_bn
        self.mask_learning = mask_learning
        
        # Convolution layers for each partition
        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)
            ) for _ in range(self.num_A)
        ])
        
        # Learnable mask for adjacency matrix
        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()))
        
        # Batch normalization
        if use_local_bn:
            # Local BN: each node has its own BN parameters
            self.bn = nn.BatchNorm1d(out_channels * self.V)
        else:
            # Global BN: all nodes share BN parameters
            self.bn = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        # Initialize convolutions
        for conv in self.conv_list:
            conv_init(conv)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C, T, V)
                N = batch size
                C = number of channels
                T = temporal frames
                V = number of nodes/joints
        
        Returns:
            Output tensor of shape (N, C_out, T, V)
        """
        N, C, T, V = x.size()
        
        # Get adjacency matrices
        A = self.A
        
        # Apply learnable mask if enabled
        if self.mask_learning:
            A = A * self.mask
        
        # Graph convolution for each partition
        y = None
        for i, a in enumerate(A):
            # Matrix multiplication: x @ A
            # x: (N, C, T, V) -> (N*C*T, V) @ (V, V) -> (N*C*T, V) -> (N, C, T, V)
            x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(N * T * C, V)  # (N*T*C, V)
            xa_reshaped = x_reshaped.mm(a)  # (N*T*C, V) @ (V, V) -> (N*T*C, V)
            xa = xa_reshaped.view(N, T, C, V).permute(0, 2, 1, 3).contiguous()  # (N, C, T, V)
            
            # Apply convolution
            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)
        
        # Batch normalization
        if self.use_local_bn:
            # Reshape for local BN: (N, C, T, V) -> (N, C*V, T)
            y = y.permute(0, 1, 3, 2).contiguous().view(N, self.out_channels * V, T)
            y = self.bn(y)
            # Reshape back: (N, C*V, T) -> (N, C, V, T) -> (N, C, T, V)
            y = y.view(N, self.out_channels, V, T).permute(0, 1, 3, 2)
        else:
            y = self.bn(y)
        
        # Non-linearity
        y = self.relu(y)
        
        return y


class UnitGCN(nn.Module):
    """
    Basic Graph Convolution Unit (alias for SpatialGCN)
    Maintains compatibility with original ST-GCN code
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        kernel_size: int = 1,
        stride: int = 1,
        use_local_bn: bool = False,
        mask_learning: bool = False,
    ):
        super(UnitGCN, self).__init__()
        self.gcn = SpatialGCN(
            in_channels=in_channels,
            out_channels=out_channels,
            A=A,
            kernel_size=kernel_size,
            stride=stride,
            use_local_bn=use_local_bn,
            mask_learning=mask_learning,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gcn(x)


# Alias for backward compatibility
unit_gcn = UnitGCN