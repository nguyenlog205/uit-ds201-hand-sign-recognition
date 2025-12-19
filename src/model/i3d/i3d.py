"""
Fixed I3D: Inflated 3D ConvNet for Video Action Recognition
Based on Inception architecture inflated to 3D convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MaxPool3dSamePadding(nn.MaxPool3d):
    """MaxPool3d with same padding"""
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)
    
    def forward(self, x):
        # Compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        
        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f
        
        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    """Basic 3D convolution unit with batch normalization"""
    
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding=0,
        activation_fn=F.relu,
        use_batch_norm=True,
        use_bias=False,
        name='unit3d',
    ):
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,  # We handle padding manually
            bias=self._use_bias
        )
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)
    
    def forward(self, x):
        # Handle padding
        if self.padding != 0:
            if isinstance(self.padding, tuple):
                if len(self.padding) == 3:
                    # (pad_t, pad_h, pad_w) -> expand to 6-tuple
                    pad_t, pad_h, pad_w = self.padding
                    pad = (pad_w, pad_w, pad_h, pad_h, pad_t, pad_t)
                elif len(self.padding) == 6:
                    pad = self.padding
                else:
                    raise ValueError(f"Invalid padding tuple length: {len(self.padding)}")
                x = F.pad(x, pad)
            elif isinstance(self.padding, int):
                # Single int: apply to all dimensions
                pad = (self.padding, self.padding, 
                       self.padding, self.padding,
                       self.padding, self.padding)
                x = F.pad(x, pad)
        
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    """Inception module for I3D"""
    
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()
        
        # Branch 0: 1x1x1 conv
        self.b0 = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[0],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name+'/Branch_0/Conv3d_0a_1x1'
        )
        
        # Branch 1: 1x1x1 -> 3x3x3
        self.b1a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[1],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name+'/Branch_1/Conv3d_0a_1x1'
        )
        self.b1b = Unit3D(
            in_channels=out_channels[1],
            output_channels=out_channels[2],
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name+'/Branch_1/Conv3d_0b_3x3'
        )
        
        # Branch 2: 1x1x1 -> 3x3x3
        self.b2a = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[3],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name+'/Branch_2/Conv3d_0a_1x1'
        )
        self.b2b = Unit3D(
            in_channels=out_channels[3],
            output_channels=out_channels[4],
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name+'/Branch_2/Conv3d_0b_3x3'
        )
        
        # Branch 3: MaxPool -> 1x1x1
        self.b3a = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(1, 1, 1),
            padding=0
        )
        self.b3b = Unit3D(
            in_channels=in_channels,
            output_channels=out_channels[5],
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name+'/Branch_3/Conv3d_0b_1x1'
        )
        
        self.name = name
    
    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class I3D(nn.Module):
    """
    I3D: Inflated 3D ConvNet for Video Action Recognition
    
    Architecture:
    1. Stem (conv + maxpool)
    2. Inception modules
    3. Classification head
    
    Input: RGB video frames (batch, channels, frames, height, width)
    Output: Classification logits
    
    Expected input shape: (B, 3, T, H, W)
    - T should be at least 16 frames (better 32 or 64)
    - H, W should be at least 224x224
    """
    
    def __init__(
        self,
        num_classes: int = 400,
        in_channels: int = 3,
        spatial_squeeze: bool = True,
        final_endpoint: str = 'Logits',
        name: str = 'inception_i3d',
        dropout_prob: float = 0.5,  # Changed from dropout_keep_prob
        use_pretrained: bool = False,
        pretrained_path: Optional[str] = None,
    ):
        """
        Args:
            num_classes: Number of output classes
            in_channels: Number of input channels (3 for RGB)
            spatial_squeeze: Whether to squeeze spatial dimensions
            final_endpoint: Final endpoint name
            name: Model name
            dropout_prob: Dropout probability (0.5 = drop 50%)
            use_pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained checkpoint
        """
        super(I3D, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.spatial_squeeze = spatial_squeeze
        self.final_endpoint = final_endpoint
        
        # Stem: Conv3d_1a_7x7
        self.conv3d_1a_7x7 = Unit3D(
            in_channels=in_channels,
            output_channels=64,
            kernel_shape=[7, 7, 7],
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            name=name+'/Conv3d_1a_7x7'
        )
        
        # MaxPool3d_2a_3x3
        self.maxPool3d_2a_3x3 = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding=0
        )
        
        # Conv3d_2b_1x1
        self.conv3d_2b_1x1 = Unit3D(
            in_channels=64,
            output_channels=64,
            kernel_shape=[1, 1, 1],
            padding=0,
            name=name+'/Conv3d_2b_1x1'
        )
        
        # Conv3d_2c_3x3
        self.conv3d_2c_3x3 = Unit3D(
            in_channels=64,
            output_channels=192,
            kernel_shape=[3, 3, 3],
            padding=1,
            name=name+'/Conv3d_2c_3x3'
        )
        
        # MaxPool3d_3a_3x3
        self.maxPool3d_3a_3x3 = MaxPool3dSamePadding(
            kernel_size=[1, 3, 3],
            stride=(1, 2, 2),
            padding=0
        )
        
        # Mixed_3b and Mixed_3c
        self.mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], name+'/Mixed_3b')
        self.mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], name+'/Mixed_3c')
        
        # MaxPool3d_4a_3x3
        self.maxPool3d_4a_3x3 = MaxPool3dSamePadding(
            kernel_size=[3, 3, 3],
            stride=(2, 2, 2),
            padding=0
        )
        
        # Mixed_4b to Mixed_4f
        self.mixed_4b = InceptionModule(480, [192, 96, 208, 16, 48, 64], name+'/Mixed_4b')
        self.mixed_4c = InceptionModule(512, [160, 112, 224, 24, 64, 64], name+'/Mixed_4c')
        self.mixed_4d = InceptionModule(512, [128, 128, 256, 24, 64, 64], name+'/Mixed_4d')
        self.mixed_4e = InceptionModule(512, [112, 144, 288, 32, 64, 64], name+'/Mixed_4e')
        self.mixed_4f = InceptionModule(528, [256, 160, 320, 32, 128, 128], name+'/Mixed_4f')
        
        # MaxPool3d_5a_2x2
        self.maxPool3d_5a_2x2 = MaxPool3dSamePadding(
            kernel_size=[2, 2, 2],
            stride=(2, 2, 2),
            padding=0
        )
        
        # Mixed_5b and Mixed_5c
        self.mixed_5b = InceptionModule(832, [256, 160, 320, 32, 128, 128], name+'/Mixed_5b')
        self.mixed_5c = InceptionModule(832, [384, 192, 384, 48, 128, 128], name+'/Mixed_5c')
        
        # Classification head
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.logits = Unit3D(
            in_channels=1024,
            output_channels=num_classes,
            kernel_shape=[1, 1, 1],
            padding=0,
            activation_fn=None,
            use_batch_norm=False,
            use_bias=True,
            name='logits'
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if specified
        if use_pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)
    
    def _initialize_weights(self):
        """Initialize weights with Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained(self, pretrained_path: str):
        """Load pretrained weights from Kinetics"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out classification head if num_classes doesn't match
            filtered_dict = {}
            for k, v in state_dict.items():
                if 'logits' in k:
                    # Skip logits if num_classes doesn't match
                    if v.shape[0] != self.num_classes:
                        print(f"Skipping {k} due to shape mismatch: {v.shape[0]} vs {self.num_classes}")
                        continue
                filtered_dict[k] = v
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
            
            print(f"✓ Loaded pretrained weights from {pretrained_path}")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)} (including logits layer)")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")
                
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)
               Expected: (B, 3, T>=16, H>=224, W>=224)
        
        Returns:
            Classification logits of shape (batch, num_classes)
        """
        # Validate input shape
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B, C, T, H, W), got {x.dim()}D")
        
        B, C, T, H, W = x.shape
        
        if T < 16:
            print(f"⚠ Warning: Input has only {T} frames (recommended: >=16)")
        if H < 224 or W < 224:
            print(f"⚠ Warning: Input resolution {H}x{W} is low (recommended: >=224x224)")
        
        # Stem
        x = self.conv3d_1a_7x7(x)          # -> (B, 64, T/2, H/2, W/2)
        x = self.maxPool3d_2a_3x3(x)       # -> (B, 64, T/2, H/4, W/4)
        x = self.conv3d_2b_1x1(x)          # -> (B, 64, T/2, H/4, W/4)
        x = self.conv3d_2c_3x3(x)          # -> (B, 192, T/2, H/4, W/4)
        x = self.maxPool3d_3a_3x3(x)       # -> (B, 192, T/2, H/8, W/8)
        
        # Inception modules
        x = self.mixed_3b(x)               # -> (B, 256, T/2, H/8, W/8)
        x = self.mixed_3c(x)               # -> (B, 480, T/2, H/8, W/8)
        x = self.maxPool3d_4a_3x3(x)       # -> (B, 480, T/4, H/16, W/16)
        
        x = self.mixed_4b(x)               # -> (B, 512, T/4, H/16, W/16)
        x = self.mixed_4c(x)               # -> (B, 512, T/4, H/16, W/16)
        x = self.mixed_4d(x)               # -> (B, 512, T/4, H/16, W/16)
        x = self.mixed_4e(x)               # -> (B, 528, T/4, H/16, W/16)
        x = self.mixed_4f(x)               # -> (B, 832, T/4, H/16, W/16)
        x = self.maxPool3d_5a_2x2(x)       # -> (B, 832, T/8, H/32, W/32)
        
        x = self.mixed_5b(x)               # -> (B, 832, T/8, H/32, W/32)
        x = self.mixed_5c(x)               # -> (B, 1024, T/8, H/32, W/32)
        
        # Classification head
        x = self.avg_pool(x)               # -> (B, 1024, 1, 1, 1)
        x = self.dropout(x)
        x = self.logits(x)                 # -> (B, num_classes, 1, 1, 1)
        
        if self.spatial_squeeze:
            x = x.squeeze(4).squeeze(3).squeeze(2)  # -> (B, num_classes)
        
        return x
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head
        
        Args:
            x: Input tensor (B, C, T, H, W)
        
        Returns:
            Features tensor (B, 1024)
        """
        # Stem
        x = self.conv3d_1a_7x7(x)
        x = self.maxPool3d_2a_3x3(x)
        x = self.conv3d_2b_1x1(x)
        x = self.conv3d_2c_3x3(x)
        x = self.maxPool3d_3a_3x3(x)
        
        # Inception modules
        x = self.mixed_3b(x)
        x = self.mixed_3c(x)
        x = self.maxPool3d_4a_3x3(x)
        
        x = self.mixed_4b(x)
        x = self.mixed_4c(x)
        x = self.mixed_4d(x)
        x = self.mixed_4e(x)
        x = self.mixed_4f(x)
        x = self.maxPool3d_5a_2x2(x)
        
        x = self.mixed_5b(x)
        x = self.mixed_5c(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.squeeze(4).squeeze(3).squeeze(2)  # (B, 1024)
        
        return x


def create_i3d_model(
    num_classes: int,
    in_channels: int = 3,
    dropout_prob: float = 0.5,
    use_pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs
) -> I3D:
    """
    Factory function to create I3D model
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for RGB)
        dropout_prob: Dropout probability (0.5 = drop 50%)
        use_pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained checkpoint
        **kwargs: Additional arguments
    
    Returns:
        I3D model instance
    
    Example:
        >>> model = create_i3d_model(num_classes=10, dropout_prob=0.5)
        >>> x = torch.randn(2, 3, 32, 224, 224)  # (B, C, T, H, W)
        >>> logits = model(x)  # (2, 10)
    """
    model = I3D(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_prob=dropout_prob,
        use_pretrained=use_pretrained,
        pretrained_path=pretrained_path,
        **kwargs
    )
    
    return model