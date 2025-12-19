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
    
    Architecture (SignBERT-style):
    1. Skeleton normalization (root centering, bone encoding, velocity)
    2. Input embedding (keypoint features + temporal + joint + hand positional encoding)
    3. [CLS] token (BERT-style)
    4. Transformer encoder blocks with spatial-temporal attention
    5. Classification head using [CLS] token
    
    Input: Keypoint sequences (batch, frames, joints, coords)
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
        use_cls_token: bool = True,
        normalize_skeleton: bool = True,
        use_velocity: bool = True,
        use_bone: bool = True,
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
            use_cls_token: Whether to use [CLS] token (BERT-style)
            normalize_skeleton: Whether to normalize skeleton (root centering)
            use_velocity: Whether to include velocity features (Δx)
            use_bone: Whether to include bone vector features
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.normalize_skeleton = normalize_skeleton
        self.use_velocity = use_velocity
        self.use_bone = use_bone
        
        # Calculate input dimension with optional features
        # All features keep joint dimension, concatenate along channel dimension
        # Base: coords per joint
        # + Velocity: coords per joint (if enabled)
        # + Bone: coords per joint (if enabled)
        base_dim = num_coords
        velocity_dim = num_coords if use_velocity else 0
        bone_dim = num_coords if use_bone else 0
        input_dim = base_dim + velocity_dim + bone_dim
        
        # Input projection: (frames, features) -> (frames, embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        # [CLS] token (BERT-style)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)
        
        # Positional embeddings
        # 1. Temporal positional embedding (frame position)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # 2. Joint positional embedding (joint position in skeleton)
        self.joint_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        
        # 3. Hand-aware embedding (left/right hand, body)
        # For MediaPipe 27: 0=neck, 1-2=shoulders, 3-4=elbows, 5-15=left_hand, 16-26=right_hand
        self.hand_type_embed = nn.Embedding(3, embed_dim)  # 0=body, 1=left_hand, 2=right_hand
        
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
        
        # Classification head (uses [CLS] token if enabled, otherwise GAP)
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
        # Positional embeddings
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_pos_embed, std=0.02)
        
        # Hand type embedding
        nn.init.normal_(self.hand_type_embed.weight, std=0.02)
        
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
    
    def _normalize_skeleton(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize skeleton: root centering, scale normalization
        
        Args:
            x: (B, T, V, C) skeleton sequence
        
        Returns:
            Normalized skeleton (B, T, V, C)
        """
        if not self.normalize_skeleton:
            return x
        
        # Root centering: subtract root joint (neck, index 0)
        root = x[:, :, 0:1, :]  # (B, T, 1, C)
        x = x - root
        
        # Scale normalization: normalize by shoulder width
        # Shoulders are typically at indices 1 (left) and 2 (right)
        if x.shape[2] > 2:
            left_shoulder = x[:, :, 1:2, :2]  # (B, T, 1, 2) - only x, y
            right_shoulder = x[:, :, 2:3, :2]  # (B, T, 1, 2)
            shoulder_width = torch.norm(left_shoulder - right_shoulder, dim=-1, keepdim=True)  # (B, T, 1, 1)
            shoulder_width = torch.clamp(shoulder_width, min=1e-6)  # Avoid division by zero
            scale = shoulder_width.median(dim=1, keepdim=True)[0]  # (B, 1, 1, 1)
            x = x / (scale * 1.2)  # 1.2 padding factor
        
        return x
    
    def _get_skeleton_parents(self) -> list:
        """
        Get parent indices for each joint based on skeleton topology
        
        Returns:
            parents: List of parent indices (-1 for root)
        """
        # MediaPipe 27 skeleton structure
        # Based on edges from SkeletonGraph
        # 0: Neck (root, no parent)
        # 1-2: Shoulders (parent: 0)
        # 3-4: Elbows (parents: 1, 2)
        # 5: Left Wrist (parent: 3)
        # 6-15: Left Hand fingers (parent: 5 for most, or previous in chain)
        # 16: Right Wrist (parent: 4)
        # 17-26: Right Hand fingers (parent: 16 for most, or previous in chain)
        
        if self.num_joints == 27:
            parents = [-1,  # 0: Neck (root)
                       0,   # 1: Left Shoulder
                       0,   # 2: Right Shoulder
                       1,   # 3: Left Elbow
                       2,   # 4: Right Elbow
                       3,   # 5: Left Wrist
                       5,   # 6: Left Thumb MCP
                       6,   # 7: Left Thumb Tip
                       5,   # 8: Left Index MCP
                       8,   # 9: Left Index Tip
                       5,   # 10: Left Middle MCP
                       10,  # 11: Left Middle Tip
                       5,   # 12: Left Ring MCP
                       12,  # 13: Left Ring Tip
                       5,   # 14: Left Pinky MCP
                       14,  # 15: Left Pinky Tip
                       4,   # 16: Right Wrist
                       16,  # 17: Right Thumb MCP
                       17,  # 18: Right Thumb Tip
                       16,  # 19: Right Index MCP
                       19,  # 20: Right Index Tip
                       16,  # 21: Right Middle MCP
                       21,  # 22: Right Middle Tip
                       16,  # 23: Right Ring MCP
                       23,  # 24: Right Ring Tip
                       16,  # 25: Right Pinky MCP
                       25]  # 26: Right Pinky Tip
        else:
            # Default: no parent relationships (all -1)
            parents = [-1] * self.num_joints
        
        return parents
    
    def _compute_bone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute bone vector features (parent-child relationships)
        
        Args:
            x: (B, T, V, C) skeleton sequence
        
        Returns:
            Bone features (B, T, V, C) - zero for root joint
        """
        B, T, V, C = x.shape
        parents = self._get_skeleton_parents()
        
        # Initialize bone features
        bones = torch.zeros_like(x)  # (B, T, V, C)
        
        # Compute bone vectors: child - parent
        for j in range(V):
            parent_idx = parents[j]
            if parent_idx >= 0:
                bones[:, :, j, :] = x[:, :, j, :] - x[:, :, parent_idx, :]
            # Root joint (parent_idx == -1) remains zero
        
        return bones
    
    def _compute_velocity_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity features (temporal differences)
        
        Args:
            x: (B, T, V, C) skeleton sequence
        
        Returns:
            Velocity features (B, T, V, C)
        """
        # Compute velocity: difference between consecutive frames
        velocity = torch.zeros_like(x)
        velocity[:, 1:, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
        return velocity
    
    def _get_hand_type_indices(self) -> torch.Tensor:
        """
        Get hand type indices for each joint
        0 = body, 1 = left_hand, 2 = right_hand
        
        Returns:
            (V,) tensor with hand type for each joint
        """
        hand_types = torch.zeros(self.num_joints, dtype=torch.long)
        
        # MediaPipe 27 layout:
        # 0: neck (body)
        # 1-2: shoulders (body)
        # 3-4: elbows (body)
        # 5-15: left hand (11 points)
        # 16-26: right hand (11 points)
        
        if self.num_joints == 27:
            hand_types[0:5] = 0  # body (neck, shoulders, elbows)
            hand_types[5:16] = 1  # left hand
            hand_types[16:27] = 2  # right hand
        else:
            # Default: all body
            hand_types[:] = 0
        
        return hand_types
    
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
            x: Input tensor of shape (batch, frames, joints, coords)
        
        Returns:
            Classification logits of shape (batch, num_classes)
        """
        B = x.shape[0]
        
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (B, T, V, C), got {x.shape}")
        
        T, V, C = x.shape[1], x.shape[2], x.shape[3]
        
        # 1. Skeleton normalization
        x = self._normalize_skeleton(x)  # (B, T, V, C)
        
        # 2. Extract features: base + velocity + bone
        # CRITICAL: Keep joint dimension (V) when concatenating features
        features_list = [x]  # Base features (B, T, V, C)
        
        if self.use_velocity:
            velocity = self._compute_velocity_features(x)  # (B, T, V, C)
            features_list.append(velocity)
        
        if self.use_bone:
            bones = self._compute_bone_features(x)  # (B, T, V, C)
            features_list.append(bones)
        
        # Concatenate features along channel dimension (keep joint dimension)
        x_features = torch.cat(features_list, dim=-1)  # (B, T, V, C')
        # C' = C (base) + C (velocity if enabled) + C (bone if enabled)
        
        # 3. Restructure to joint-level tokens: (B, T, V, C') -> (B, T*V, C')
        # Each token represents a (frame, joint) pair for proper spatial-temporal modeling
        x_features = x_features.view(B, T * V, -1)  # (B, T*V, C')
        
        # Project to embedding dimension
        x = self.input_proj(x_features)  # (B, T*V, embed_dim)
        
        # 4. Add positional embeddings
        # Temporal positional embedding: repeat for each joint in frame
        temporal_embed = self.temporal_pos_embed[:, :T, :]  # (1, T, embed_dim)
        temporal_embed = temporal_embed.repeat(1, V, 1)  # (1, T*V, embed_dim)
        x = x + temporal_embed
        
        # Joint positional embedding: repeat for each frame
        joint_embed = self.joint_pos_embed  # (1, V, embed_dim)
        joint_embed = joint_embed.repeat(1, T, 1)  # (1, T*V, embed_dim)
        x = x + joint_embed
        
        # Hand type embedding
        hand_types = self._get_hand_type_indices().to(x.device)  # (V,)
        hand_embed = self.hand_type_embed(hand_types)  # (V, embed_dim)
        hand_embed = hand_embed.repeat(T, 1)  # (T*V, embed_dim)
        hand_embed = hand_embed.unsqueeze(0).expand(B, -1, -1)  # (B, T*V, embed_dim)
        x = x + hand_embed
        
        x = self.pos_drop(x)
        
        # 5. Add [CLS] token (BERT-style)
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            # Add CLS positional embedding
            if hasattr(self, 'cls_pos_embed'):
                cls_tokens = cls_tokens + self.cls_pos_embed
            x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+T*V, embed_dim)
        
        # 6. Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # 7. Extract representation
        if self.use_cls_token:
            # Use [CLS] token (first token) - BERT-style
            x = x[:, 0]  # (B, embed_dim)
        else:
            # Global average pooling over all tokens
            x = x.mean(dim=1)  # (B, embed_dim)
        
        # 8. Classification head
        x = self.head(x)  # (B, num_classes)
        
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
    use_cls_token: bool = True,
    normalize_skeleton: bool = True,
    use_velocity: bool = True,
    use_bone: bool = True,
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
        use_cls_token=use_cls_token,
        normalize_skeleton=normalize_skeleton,
        use_velocity=use_velocity,
        use_bone=use_bone,
        **kwargs
    )
    
    return model