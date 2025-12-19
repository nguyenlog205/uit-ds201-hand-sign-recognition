"""
HA-GCN: Hand-aware Graph Convolutional Network
Main model implementation with HA-GC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from typing import Optional, Tuple

from .attention import STCAttention
from .adaptive_dropgraph import AdaptiveDropGraph


def conv_init(module):
    """Initialize convolution weights"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def bn_init(bn, scale):
    """Initialize batch normalization"""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    """Initialize convolution branch"""
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class TemporalConv(nn.Module):
    """Temporal Convolution with Adaptive DropGraph"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        num_point: int = 27,
        block_size: int = 41,
    ):
        super(TemporalConv, self).__init__()
        
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.adrop = AdaptiveDropGraph(num_point=num_point, block_size=block_size)
        
        conv_init(self.conv)
        bn_init(self.bn, 1)
    
    def forward(self, x: torch.Tensor, keep_prob: float, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (N, C, T, V)
            keep_prob: Keep probability for dropout
            A: Adjacency matrix
        
        Returns:
            Output (N, C_out, T, V)
        """
        x = self.bn(self.conv(x))
        x = self.adrop(x, keep_prob, A)
        return x


class HandAwareGCN(nn.Module):
    """
    Hand-aware Graph Convolution Layer
    
    Combines body graph (A) with hand graphs (SH and PH)
    Formula: A + PA + SH * alpha + PH * beta
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        SH: torch.Tensor,
        PH: torch.Tensor,
        num_subset: int = 3,
        freeze_graph: bool = False,
    ):
        """
        Initialize HA-GC Layer
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            A: Body adjacency matrix (K, V, V)
            SH: Structured Hand Graph (K, V, V)
            PH: Parameterized Hand Graph (K, V, V) - learnable
            num_subset: Number of spatial partitions
        """
        super(HandAwareGCN, self).__init__()
        
        # Register body graph (fixed)
        self.register_buffer('A', A.float())
        
        # Register structured hand graph (fixed)
        self.register_buffer('SH', SH.float())
        
        # Parameterized hand graph (learnable or frozen)
        if freeze_graph:
            self.register_buffer('PH', PH.float())  # Frozen
        else:
            self.PH = nn.Parameter(PH.float())  # Learnable
        
        # Learnable body graph adjustments (can be frozen for small datasets)
        self.PA = nn.Parameter(A.float())
        
        # Learnable gating coefficients
        self.alpha = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
        
        self.num_subset = num_subset
        
        # Convolution layers for each partition
        self.conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(self.num_subset)
        ])
        
        # Residual connection
        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = lambda x: x
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv[i], self.num_subset)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (N, C, T, V)
        
        Returns:
            Output (N, C_out, T, V)
        """
        # Get graphs on same device as input
        A = self.A.to(x.device)
        SH = self.SH.to(x.device)
        PH = self.PH.to(x.device)
        
        # Combine graphs: A + PA + SH * alpha + PH * beta
        # A: (K, V, V), PA: (K, V, V), SH: (K, V, V), PH: (K, V, V)
        A_combined = A + self.PA + SH * self.alpha + PH * self.beta
        
        # Graph convolution for each partition
        y = None
        for i in range(self.num_subset):
            f = self.conv[i](x)  # (N, C_out, T, V)
            N, C, T, V = f.size()
            
            # Matrix multiplication: f @ A_combined[i]
            # (N, C, T, V) -> (N, C*T, V) @ (V, V) -> (N, C*T, V) -> (N, C, T, V)
            z = torch.matmul(
                f.view(N, C * T, V),
                A_combined[i]
            ).view(N, C, T, V)
            
            if y is None:
                y = z
            else:
                y = y + z
        
        # Batch normalization
        y = self.bn(y)
        
        # Residual connection
        y = y + self.res(x)
        
        return self.relu(y)


class HAGCNBlock(nn.Module):
    """
    HA-GCN Block: HA-GC Layer + STC Attention + Temporal Conv + Residual
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        SH: torch.Tensor,
        PH: torch.Tensor,
        num_point: int = 27,
        block_size: int = 41,
        stride: int = 1,
        residual: bool = True,
        attention: bool = True,
        freeze_graph: bool = False,
    ):
        super(HAGCNBlock, self).__init__()
        
        # HA-GC Layer
        self.gcn = HandAwareGCN(in_channels, out_channels, A, SH, PH, freeze_graph=freeze_graph)
        
        # STC Attention
        self.attention = attention
        if attention:
            self.stc_attention = STCAttention(
                in_channels=out_channels,
                num_nodes=num_point,
            )
        
        # Temporal Convolution
        self.tcn = TemporalConv(
            out_channels,
            out_channels,
            kernel_size=9,
            stride=stride,
            num_point=num_point,
            block_size=block_size,
        )
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            # Need to handle both channel and temporal dimension changes
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(stride, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        # Store A for temporal conv
        if A.dim() == 3:
            self.register_buffer('A_sum', torch.sum(A.float(), dim=0))
        else:
            self.register_buffer('A_sum', A.float())
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor, keep_prob: float = 0.9) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input (N, C, T, V)
            keep_prob: Keep probability for dropout
        
        Returns:
            Output (N, C_out, T, V)
        """
        # HA-GC Layer
        y = self.gcn(x)  # (N, C_out, T, V)
        
        # STC Attention
        if self.attention:
            y = self.stc_attention(y)  # (N, C_out, T, V)
        
        # Temporal Convolution with Adaptive DropGraph
        y = self.tcn(y, keep_prob, self.A_sum)  # (N, C_out, T, V)
        
        # Residual connection
        x_skip = self.residual(x)
        y = y + x_skip
        
        return self.relu(y)