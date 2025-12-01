"""
HA-GCN Model: Hand-aware Graph Convolutional Network
Main model class for sign language recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional

from .ha_gcn import HAGCNBlock
from .hierarchy import HandGraphConstructor, create_hand_graphs_for_spatial_partitioning


def bn_init(bn, scale):
    """Initialize batch normalization"""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class HAGCN(nn.Module):
    """
    Hand-aware Graph Convolutional Network
    
    Architecture:
    - Input BN
    - 10 HA-GCN Blocks
    - Global Average Pooling
    - Classification head
    
    Args:
        in_channels: Number of input channels (default: 3)
        num_class: Number of output classes
        num_nodes: Number of joints (default: 27 for sign language)
        num_person: Number of people (default: 1)
        window_size: Temporal window size
        A: Body adjacency matrix (K, V, V)
        block_size: Block size for Adaptive DropGraph (default: 41)
        use_attention: Whether to use STC attention (default: True)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_class: int = 100,
        num_nodes: int = 27,
        num_person: int = 1,
        window_size: int = 64,
        A: Optional[torch.Tensor] = None,
        block_size: int = 41,
        use_attention: bool = True,
    ):
        super(HAGCN, self).__init__()
        
        if A is None:
            raise ValueError("Adjacency matrix A must be provided")
        
        self.num_class = num_class
        self.num_nodes = num_nodes
        self.num_person = num_person
        self.window_size = window_size
        
        # Register body adjacency matrix
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        self.register_buffer('A', A.float())
        
        # Create hand graphs
        hand_constructor = HandGraphConstructor(num_nodes=num_nodes)
        SH_base, PH_base = hand_constructor.get_hand_graphs()
        
        # Stack for spatial partitions if needed
        if A.dim() == 3:
            num_partitions = A.size(0)
            SH = np.stack([SH_base] * num_partitions, axis=0)
            PH = np.stack([PH_base] * num_partitions, axis=0)
        else:
            SH = np.expand_dims(SH_base, axis=0)
            PH = np.expand_dims(PH_base, axis=0)
        
        SH = torch.from_numpy(SH).float()
        PH = torch.from_numpy(PH).float()
        
        self.register_buffer('SH', SH)
        self.PH = nn.Parameter(PH)  # Learnable
        
        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)
        bn_init(self.data_bn, 1)
        
        # HA-GCN Blocks (10 blocks as in paper)
        self.l1 = HAGCNBlock(
            in_channels, 64, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=False, attention=use_attention
        )
        self.l2 = HAGCNBlock(
            64, 64, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l3 = HAGCNBlock(
            64, 64, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l4 = HAGCNBlock(
            64, 64, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l5 = HAGCNBlock(
            64, 128, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            stride=2, residual=True, attention=use_attention
        )
        self.l6 = HAGCNBlock(
            128, 128, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l7 = HAGCNBlock(
            128, 128, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l8 = HAGCNBlock(
            128, 256, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            stride=2, residual=True, attention=use_attention
        )
        self.l9 = HAGCNBlock(
            256, 256, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        self.l10 = HAGCNBlock(
            256, 256, self.A, self.SH, self.PH,
            num_point=num_nodes, block_size=block_size,
            residual=True, attention=use_attention
        )
        
        # Classification head
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor, keep_prob: float = 0.9) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (N, C, T, V, M)
                N = batch size
                C = channels
                T = temporal frames
                V = number of joints
                M = number of people
            keep_prob: Keep probability for dropout
        
        Returns:
            Output logits (N, num_class)
        """
        N, C, T, V, M = x.size()
        
        # Data batch normalization
        # (N, C, T, V, M) -> (N, M*V*C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        # (N, M*V*C, T) -> (N*M, C, T, V)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        # HA-GCN Blocks
        x = self.l1(x, keep_prob=1.0)
        x = self.l2(x, keep_prob=1.0)
        x = self.l3(x, keep_prob=1.0)
        x = self.l4(x, keep_prob=1.0)
        x = self.l5(x, keep_prob=1.0)
        x = self.l6(x, keep_prob=1.0)
        x = self.l7(x, keep_prob=keep_prob)
        x = self.l8(x, keep_prob=keep_prob)
        x = self.l9(x, keep_prob=keep_prob)
        x = self.l10(x, keep_prob=keep_prob)
        
        # Global Average Pooling
        # (N*M, C, T, V) -> (N, M, C, T*V) -> (N, M, C) -> (N, C)
        N_M, C_out, T_out, V_out = x.size()
        x = x.reshape(N, M, C_out, -1)  # (N, M, C, T*V)
        x = x.mean(dim=3).mean(dim=1)  # (N, C)
        
        # Classification
        x = self.fc(x)  # (N, num_class)
        
        return x


def create_hagcn_model(
    num_class: int,
    num_nodes: int = 27,
    window_size: int = 64,
    in_channels: int = 3,
    A: Optional[torch.Tensor] = None,
    **kwargs
) -> HAGCN:
    """
    Factory function to create HA-GCN model
    
    Args:
        num_class: Number of output classes
        num_nodes: Number of joints
        window_size: Temporal window size
        in_channels: Number of input channels
        A: Adjacency matrix (K, V, V)
        **kwargs: Additional arguments for HAGCN
    
    Returns:
        HAGCN model instance
    """
    if A is None:
        raise ValueError("Adjacency matrix A must be provided")
    
    model = HAGCN(
        in_channels=in_channels,
        num_class=num_class,
        num_nodes=num_nodes,
        window_size=window_size,
        A=A,
        **kwargs
    )
    
    return model