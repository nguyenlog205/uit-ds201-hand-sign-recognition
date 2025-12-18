"""
PoseFormer: Transformer-based model for Sign Language Recognition
Base-code from: https://github.com/zczcwh/PoseFormer.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from typing import Optional


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with self-attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PoseFormer(nn.Module):
    """
    PoseFormer: Transformer-based model for Sign Language Recognition
    
    Architecture:
    1. Spatial Transformer: Processes each frame independently to extract spatial features from joints
    2. Temporal Transformer: Processes the temporal sequence of spatial features
    3. Classification Head: Maps to sign language classes
    
    Based on the reference implementation from:
    "PoseFormer: A Simple Baseline for 3D Human Pose Estimation" (Zheng et al., 2021)
    Adapted for sign language recognition (classification task).
    """
    
    def __init__(
        self,
        num_frame: int = 64,
        num_joints: int = 27,
        in_chans: int = 3,
        embed_dim_ratio: int = 32,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        num_class: int = 100,
        norm_layer: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_frame (int): Number of frames in temporal sequence
            num_joints (int): Number of joints
            in_chans (int): Number of input channels (typically 3: x, y, confidence)
            embed_dim_ratio (int): Embedding dimension ratio for spatial features
            depth (int): Depth of transformer (number of blocks)
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): Enable bias for qkv if True
            qk_scale (float): Override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): Dropout rate
            attn_drop_rate (float): Attention dropout rate
            drop_path_rate (float): Stochastic depth rate
            num_class (int): Number of output classes for classification
            norm_layer: Normalization layer
        """
        super().__init__()
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  # temporal embed_dim is num_joints * spatial embedding dim ratio
        
        self.num_frame = num_frame
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.embed_dim_ratio = embed_dim_ratio
        
        # Spatial patch embedding
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        
        # Temporal positional embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Spatial transformer blocks (process each frame independently)
        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        # Temporal transformer blocks (process temporal sequence)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        
        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        
        # Classification head for sign language recognition
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_class),
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize spatial positional embedding
        nn.init.trunc_normal_(self.Spatial_pos_embed, std=0.02)
        # Initialize temporal positional embedding
        nn.init.trunc_normal_(self.Temporal_pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def Spatial_forward_features(self, x):
        """
        Process spatial features for each frame independently.
        
        Args:
            x: Input tensor of shape (batch, channels, frames, joints)
        
        Returns:
            Output tensor of shape (batch, frames, embed_dim)
        """
        b, c, f, p = x.shape  # batch, channels, frames, joints
        
        # Reshape: (batch, channels, frames, joints) -> (batch*frames, joints, channels)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, frames, joints, channels)
        x = x.view(b * f, p, c)  # (batch*frames, joints, channels)
        
        # Spatial embedding
        x = self.Spatial_patch_to_embedding(x)  # (batch*frames, joints, embed_dim_ratio)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        
        # Spatial transformer blocks
        for blk in self.Spatial_blocks:
            x = blk(x)
        
        x = self.Spatial_norm(x)
        
        # Reshape: (batch*frames, joints, embed_dim_ratio) -> (batch, frames, embed_dim)
        x = x.view(b, f, p * self.embed_dim_ratio)
        
        return x
    
    def forward_features(self, x):
        """
        Process temporal features.
        
        Args:
            x: Input tensor of shape (batch, frames, embed_dim)
        
        Returns:
            Output tensor of shape (batch, embed_dim)
        """
        b = x.shape[0]
        
        # Add temporal positional embedding
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        
        # Temporal transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.Temporal_norm(x)
        
        x = x.mean(dim=1)  # (batch, frames, embed_dim) -> (batch, embed_dim)
        
        return x
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (N, C, T, V, M)
                N = batch size
                C = input channels (typically 3: x, y, confidence)
                T = temporal frames
                V = number of joints
                M = number of people
            mask: Optional attention mask
        
        Returns:
            Output logits of shape (N, num_class)
        """
        N, C, T, V, M = x.size()
        
        # Reshape for single person: (N, C, T, V, M) -> (N*M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Spatial forward: (N*M, C, T, V) -> (N*M, T, embed_dim)
        x = self.Spatial_forward_features(x)
        
        # Temporal forward: (N*M, T, embed_dim) -> (N*M, embed_dim)
        x = self.forward_features(x)
        
        # Classification head: (N*M, embed_dim) -> (N*M, num_class)
        x = self.head(x)
        
        # Reshape back if multiple people: (N*M, num_class) -> (N, M, num_class) -> (N, num_class)
        if M > 1:
            x = x.view(N, M, -1).mean(dim=1)
        
        return x


def create_poseformer_model(
    num_class: int,
    num_joints: int = 27,
    num_frames: int = 64,
    in_channels: int = 3,
    embed_dim_ratio: int = 32,
    depth: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 2.0,
    **kwargs
) -> PoseFormer:
    """
    Factory function to create PoseFormer model
    
    Args:
        num_class: Number of output classes
        num_joints: Number of joints (default: 27 for sign language)
        num_frames: Temporal window size
        in_channels: Number of input channels
        embed_dim_ratio: Embedding dimension ratio for spatial features
        depth: Number of transformer layers
        num_heads: Number of attention heads
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        **kwargs: Additional arguments for PoseFormer
    
    Returns:
        PoseFormer model instance
    """
    model = PoseFormer(
        num_frame=num_frames,
        num_joints=num_joints,
        in_chans=in_channels,
        embed_dim_ratio=embed_dim_ratio,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_class=num_class,
        **kwargs
    )
    
    return model