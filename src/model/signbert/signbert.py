"""
SignBERT: Pretrained Transformer Model for Sign Language Recognition
Based on BERT architecture adapted for skeleton/keypoint sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
from functools import partial


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
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer"""
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


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SignBERT(nn.Module):
    """
    SignBERT: Pretrained Transformer for Sign Language Recognition
    
    Architecture:
    1. Input embedding (keypoint features + positional encoding)
    2. Transformer encoder blocks
    3. Classification head
    
    Input: Keypoint sequences (batch, frames, joints, coords) or (batch, frames, features)
    Output: Classification logits
    """
    
    def __init__(
        self,
        num_joints: int = 27,
        num_coords: int = 3,
        num_frames: int = 64,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        num_classes: int = 100,
        use_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        """
        Args:
            num_joints: Number of joints/keypoints
            num_coords: Number of coordinates per joint (x, y, z or x, y, confidence)
            num_frames: Number of frames in sequence
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            qkv_bias: Whether to use bias in QKV projection
            qk_scale: Scale factor for attention (None = 1/sqrt(head_dim))
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            drop_path_rate: Drop path rate
            norm_layer: Normalization layer
            num_classes: Number of output classes
            use_pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained checkpoint
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Input projection: (frames, joints, coords) -> (frames, embed_dim)
        # Flatten joints and coords: num_joints * num_coords -> embed_dim
        input_dim = num_joints * num_coords
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained weights if specified
        if use_pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _init_weights(self):
        """Initialize weights"""
        # Positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Input projection
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)
        
        # Classification head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, pretrained_path: str):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out classification head if dimensions don't match
            filtered_dict = {}
            for k, v in state_dict.items():
                if 'head' in k:
                    # Skip head if num_classes doesn't match
                    if 'head.5.weight' in k or 'head.5.bias' in k:
                        continue
                filtered_dict[k] = v
            
            self.load_state_dict(filtered_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, frames, joints, coords) or (batch, frames, features)
        
        Returns:
            Classification logits of shape (batch, num_classes)
        """
        B = x.shape[0]
        
        # Handle different input formats
        if x.dim() == 4:
            # (batch, frames, joints, coords)
            T, V, C = x.shape[1], x.shape[2], x.shape[3]
            x = x.view(B, T, V * C)  # (batch, frames, joints*coords)
        elif x.dim() == 3:
            # (batch, frames, features) - already flattened
            pass
        else:
            raise ValueError(f"Expected input shape (B, T, V, C) or (B, T, F), got {x.shape}")
        
        # Project to embedding dimension
        x = self.input_proj(x)  # (batch, frames, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Global average pooling over temporal dimension
        x = x.mean(dim=1)  # (batch, embed_dim)
        
        # Classification head
        x = self.head(x)  # (batch, num_classes)
        
        return x


def create_signbert_model(
    num_classes: int,
    num_joints: int = 27,
    num_coords: int = 3,
    num_frames: int = 64,
    embed_dim: int = 256,
    depth: int = 6,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.1,
    use_pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> SignBERT:
    """
    Factory function to create SignBERT model
    
    Args:
        num_classes: Number of output classes
        num_joints: Number of joints/keypoints
        num_coords: Number of coordinates per joint
        num_frames: Number of frames in sequence
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        drop_rate: Dropout rate
        use_pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained checkpoint
        **kwargs: Additional arguments
    
    Returns:
        SignBERT model instance
    """
    model = SignBERT(
        num_joints=num_joints,
        num_coords=num_coords,
        num_frames=num_frames,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        num_classes=num_classes,
        use_pretrained=use_pretrained,
        pretrained_path=pretrained_path,
        **kwargs
    )
    
    return model