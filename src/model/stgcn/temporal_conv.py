"""
Temporal Convolution Module
Temporal convolution layers for ST-GCN
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


class TemporalConv(nn.Module):
    """
    Temporal Convolution Unit (TCN)
    
    Applies 1D convolution along temporal dimension
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Temporal kernel size (default: 9)
        stride: Temporal stride (default: 1)
        dropout: Dropout rate (default: 0.0)
        bias: Whether to use bias (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super(TemporalConv, self).__init__()
        
        pad = int((kernel_size - 1) / 2)
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            bias=bias
        )
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        conv_init(self.conv)
    
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
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Unit2D(nn.Module):
    """
    Basic 2D Convolution Unit (alias for TemporalConv)
    Maintains compatibility with original ST-GCN code
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super(Unit2D, self).__init__()
        self.tcn = TemporalConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tcn(x)