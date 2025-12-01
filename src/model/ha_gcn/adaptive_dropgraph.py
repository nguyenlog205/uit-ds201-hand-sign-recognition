"""
Adaptive DropGraph Module for HA-GCN
Spatial and Temporal Dropout with Adaptive Gating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional


class SpatialDropGraph(nn.Module):
    """
    Spatial DropGraph - Drops joints/nodes based on adjacency matrix
    """
    
    def __init__(self, num_point: int = 27, block_size: int = 7):
        super(SpatialDropGraph, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
        self.num_point = num_point
    
    def forward(self, x: torch.Tensor, keep_prob: float, A: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial dropout on joints
        
        Args:
            x: Input tensor (N, C, T, V)
            keep_prob: Keep probability (1.0 = no dropout)
            A: Adjacency matrix (V, V) or (K, V, V)
        
        Returns:
            Dropped features (N, C, T, V)
        """
        self.keep_prob = keep_prob
        
        # No dropout during inference or if keep_prob == 1
        if not self.training or self.keep_prob == 1.0:
            return x
        
        N, C, T, V = x.size()
        
        # Get adjacency matrix (use first partition if stacked)
        if A.dim() == 3:
            A = A[0]  # (V, V)
        
        # Compute attention map: average over time and channels
        # (N, C, T, V) -> (N, C, V) -> (N, V)
        input_abs = torch.mean(torch.mean(torch.abs(x), dim=2), dim=1).detach()  # (N, V)
        
        # Normalize attention map
        input_abs = input_abs / torch.sum(input_abs, dim=1, keepdim=True) * input_abs.numel()
        
        # Compute gamma (drop probability)
        if self.num_point == 25:
            gamma = (1.0 - self.keep_prob) / (1 + 1.92)
        elif self.num_point == 20:
            gamma = (1.0 - self.keep_prob) / (1 + 1.9)
        else:  # 27 or other
            gamma = (1.0 - self.keep_prob) / (1 + 1.92)
        
        # Generate dropout mask using Bernoulli distribution
        M_seed = torch.bernoulli(
            torch.clamp(input_abs * gamma, max=1.0)
        ).to(device=x.device, dtype=x.dtype)  # (N, V)
        
        # Apply adjacency matrix to propagate dropout
        # M_seed: (N, V), A: (V, V) -> M: (N, V)
        M = torch.matmul(M_seed, A)  # (N, V)
        
        # Binarize mask
        M[M > 0.001] = 1.0
        M[M < 0.5] = 0.0
        
        # Create keep mask: (N, 1, 1, V)
        mask = (1 - M).view(N, 1, 1, self.num_point)
        
        # Apply mask and normalize
        return x * mask / (mask.sum() + 1e-8) * mask.numel()


class TemporalDropGraph(nn.Module):
    """
    Temporal DropGraph - Drops frames based on temporal importance
    """
    
    def __init__(self, block_size: int = 41):
        super(TemporalDropGraph, self).__init__()
        self.keep_prob = 0.0
        self.block_size = block_size
    
    def forward(self, x: torch.Tensor, keep_prob: float) -> torch.Tensor:
        """
        Apply temporal dropout on frames
        
        Args:
            x: Input tensor (N, C, T, V)
            keep_prob: Keep probability (1.0 = no dropout)
        
        Returns:
            Dropped features (N, C, T, V)
        """
        self.keep_prob = keep_prob
        
        # No dropout during inference or if keep_prob == 1
        if not self.training or self.keep_prob == 1.0:
            return x
        
        N, C, T, V = x.size()
        
        # Compute attention map: average over joints and channels
        # (N, C, T, V) -> (N, C, T) -> (N, T)
        input_abs = torch.mean(torch.mean(torch.abs(x), dim=3), dim=1).detach()  # (N, T)
        
        # Normalize attention map
        input_abs = (input_abs / torch.sum(input_abs, dim=1, keepdim=True) * input_abs.numel()).view(N, 1, T)  # (N, 1, T)
        
        # Compute gamma
        gamma = (1.0 - self.keep_prob) / self.block_size
        
        # Reshape for temporal convolution: (N, C, T, V) -> (N, C*V, T)
        x_reshaped = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)  # (N, C*V, T)
        
        # Generate dropout mask
        M = torch.bernoulli(
            torch.clamp(input_abs * gamma, max=1.0)
        ).repeat(1, C * V, 1)  # (N, C*V, T)
        
        # Apply max pooling to create block dropout
        Msum = F.max_pool1d(
            M,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )  # (N, C*V, T)
        
        # Create keep mask
        mask = (1 - Msum).to(device=x.device, dtype=x.dtype)  # (N, C*V, T)
        
        # Apply mask and reshape back
        output = (x_reshaped * mask * mask.numel() / (mask.sum() + 1e-8)).view(N, C, V, T).permute(0, 1, 3, 2)
        
        return output


class AdaptiveDropGraph(nn.Module):
    """
    Adaptive DropGraph with learnable gating mechanism
    
    Combines spatial and temporal dropout with adaptive coefficients
    """
    
    def __init__(self, num_point: int = 27, block_size: int = 41):
        """
        Initialize Adaptive DropGraph
        
        Args:
            num_point: Number of joints/nodes
            block_size: Temporal block size for dropout
        """
        super(AdaptiveDropGraph, self).__init__()
        
        self.dropS = SpatialDropGraph(num_point=num_point)
        self.dropT = TemporalDropGraph(block_size=block_size)
        
        # Learnable gating coefficients
        self.gamma = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor([random.random()], dtype=torch.float32))
    
    def forward(self, x: torch.Tensor, keep_prob: float, A: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive dropout
        
        Args:
            x: Input tensor (N, C, T, V)
            keep_prob: Keep probability
            A: Adjacency matrix
        
        Returns:
            Dropped features (N, C, T, V)
        """
        # Spatial dropout with adaptive gating
        a1 = self.gamma * self.dropS(x, keep_prob, A)
        y = a1 + (1 - self.gamma) * x
        
        # Temporal dropout with adaptive gating
        a2 = self.delta * self.dropT(y, keep_prob)
        y = a2 + (1 - self.delta) * y
        
        return y