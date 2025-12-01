"""
STC Attention Module for HA-GCN
Spatial, Temporal, and Channel Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_init(module):
    """Initialize convolution weights"""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class STCAttention(nn.Module):
    """
    Spatial-Temporal-Channel Attention Module
    
    Applies attention in three dimensions:
    1. Spatial attention: Focus on important joints
    2. Temporal attention: Focus on important frames
    3. Channel attention: Focus on important feature channels
    """
    
    def __init__(self, in_channels: int, num_nodes: int = 27, temporal_kernel_size: int = 9):
        """
        Initialize STC Attention
        
        Args:
            in_channels: Number of input channels
            num_nodes: Number of joints/nodes
            temporal_kernel_size: Temporal convolution kernel size
        """
        super(STCAttention, self).__init__()
        
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        
        # Spatial attention
        # Use 1D conv over joints
        ker_jpt = num_nodes - 1 if num_nodes % 2 == 0 else num_nodes
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(in_channels, 1, kernel_size=ker_jpt, padding=pad)
        
        # Temporal attention
        # Use 1D conv over time
        pad_t = (temporal_kernel_size - 1) // 2
        self.conv_ta = nn.Conv1d(in_channels, 1, kernel_size=temporal_kernel_size, padding=pad_t)
        
        # Channel attention
        # Use fully connected layers with reduction ratio
        reduction_ratio = 2
        self.fc1c = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2c = nn.Linear(in_channels // reduction_ratio, in_channels)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        # Initialize
        conv_init(self.conv_sa)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply STC attention
        
        Args:
            x: Input tensor (N, C, T, V)
                N = batch size
                C = channels
                T = temporal frames
                V = number of nodes/joints
        
        Returns:
            Attended features (N, C, T, V)
        """
        # Spatial attention
        # Average over temporal dimension: (N, C, T, V) -> (N, C, V)
        se = x.mean(dim=2)  # (N, C, V)
        # Apply 1D conv: (N, C, V) -> (N, 1, V)
        se1 = self.sigmoid(self.conv_sa(se))  # (N, 1, V)
        # Apply attention: (N, C, T, V) * (N, 1, 1, V) + (N, C, T, V)
        x = x * se1.unsqueeze(2) + x  # (N, C, T, V)
        
        # Temporal attention
        # Average over spatial dimension: (N, C, T, V) -> (N, C, T)
        se = x.mean(dim=3)  # (N, C, T)
        # Apply 1D conv: (N, C, T) -> (N, 1, T)
        se1 = self.sigmoid(self.conv_ta(se))  # (N, 1, T)
        # Apply attention: (N, C, T, V) * (N, 1, T, 1) + (N, C, T, V)
        x = x * se1.unsqueeze(-1) + x  # (N, C, T, V)
        
        # Channel attention
        # Average over both spatial and temporal: (N, C, T, V) -> (N, C)
        se = x.mean(dim=3).mean(dim=2)  # (N, C)
        # Apply FC layers: (N, C) -> (N, C//r) -> (N, C)
        se1 = self.relu(self.fc1c(se))  # (N, C//r)
        se2 = self.sigmoid(self.fc2c(se1))  # (N, C)
        # Apply attention: (N, C, T, V) * (N, C, 1, 1) + (N, C, T, V)
        x = x * se2.unsqueeze(-1).unsqueeze(-1) + x  # (N, C, T, V)
        
        return x