"""
VideoMAE: Video Masked Autoencoder for Vietnamese Sign Language Recognition
Based on pretrained VideoMAE model from Hugging Face Transformers
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig


class VideoMAEForSignLanguage(nn.Module):
    """
    VideoMAE model adapted for Sign Language Recognition
    
    Architecture:
    1. Pretrained VideoMAE backbone (from Hugging Face)
    2. Classification head for sign language classes
    
    Input: Video frames (list of images or tensor)
    Output: Classification logits
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/VideoMAEv2-Base",
        num_class: int = 100,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            model_name: Hugging Face model name or path
            num_class: Number of output classes for classification
            freeze_backbone: Whether to freeze the pretrained backbone
            dropout: Dropout rate for classification head
            hidden_dim: Hidden dimension for classification head (default: uses model's hidden size)
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_class = num_class
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained VideoMAE model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            config=config, 
            trust_remote_code=True
        )
        
        # Get hidden dimension from config
        if hidden_dim is None:
            # Try multiple ways to get hidden size from config
            if hasattr(config, 'hidden_size'):
                hidden_dim = config.hidden_size
            elif hasattr(config, 'hidden_dim'):
                hidden_dim = config.hidden_dim
            elif hasattr(config, 'encoder') and hasattr(config.encoder, 'hidden_size'):
                hidden_dim = config.encoder.hidden_size
            else:
                # Try to infer from model output
                # VideoMAE Base typically has 768, Large has 1024
                if 'base' in model_name.lower():
                    hidden_dim = 768
                elif 'large' in model_name.lower():
                    hidden_dim = 1024
                else:
                    hidden_dim = 768  # Default fallback
        
        self.hidden_dim = hidden_dim
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_class),
        )
        
        # Initialize classification head
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classification head weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            pixel_values: Input video frames
                Shape: (batch_size, num_frames, num_channels, height, width)
                or (batch_size, num_channels, num_frames, height, width)
            head_mask: Optional attention mask
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict or tuple
        
        Returns:
            Classification logits of shape (batch_size, num_class)
        """
        # Ensure correct input format: (B, C, T, H, W)
        if pixel_values.dim() == 5:
            if pixel_values.shape[1] != 3:  # If (B, T, C, H, W), permute to (B, C, T, H, W)
                pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        
        # Forward through backbone
        # VideoMAEv2 may have different forward signature, so we only pass pixel_values
        # and handle the output format dynamically
        try:
            # Try with minimal arguments first
            outputs = self.backbone(pixel_values=pixel_values)
        except TypeError:
            # If that fails, try with just positional argument
            outputs = self.backbone(pixel_values)
        
        # Get pooled output (usually the [CLS] token or mean pooling)
        if hasattr(outputs, 'last_hidden_state'):
            # Mean pooling over sequence dimension
            hidden_states = outputs.last_hidden_state
            # hidden_states shape: (batch_size, num_patches, hidden_dim)
            pooled_output = hidden_states.mean(dim=1)  # (batch_size, hidden_dim)
        elif hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        elif hasattr(outputs, 'logits'):
            return outputs.logits
        else:
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            
            if hidden_states.dim() == 3:
                # (batch_size, seq_len, hidden_dim)
                pooled_output = hidden_states.mean(dim=1)
            elif hidden_states.dim() == 2:
                # Already pooled: (batch_size, hidden_dim)
                pooled_output = hidden_states
            else:
                # Unexpected shape, try to flatten
                batch_size = hidden_states.size(0)
                hidden_states = hidden_states.view(batch_size, -1)
                if hidden_states.size(1) == self.hidden_dim:
                    pooled_output = hidden_states
                else:
                    # Use mean as fallback
                    pooled_output = hidden_states.mean(dim=1, keepdim=True)
                    # Expand to expected hidden_dim if needed
                    if pooled_output.size(1) != self.hidden_dim:
                        # Project to hidden_dim
                        if not hasattr(self, '_fallback_proj'):
                            self._fallback_proj = nn.Linear(pooled_output.size(1), self.hidden_dim).to(pooled_output.device)
                        pooled_output = self._fallback_proj(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_backbone(self):
        """Get the pretrained backbone model"""
        return self.backbone
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def freeze_backbone_layers(self, num_layers_to_freeze: int):
        """
        Freeze the first N layers of the backbone
        
        Args:
            num_layers_to_freeze: Number of layers to freeze from the beginning
        """
        if hasattr(self.backbone, 'encoder'):
            encoder = self.backbone.encoder
            if hasattr(encoder, 'layer'):
                layers = encoder.layer
                for i in range(min(num_layers_to_freeze, len(layers))):
                    for param in layers[i].parameters():
                        param.requires_grad = False


class VideoMAEProcessor:
    """
    Wrapper for VideoMAE Image Processor
    """
    
    def __init__(self, model_name: str = "OpenGVLab/VideoMAEv2-Base"):
        """
        Args:
            model_name: Hugging Face model name or path
        """
        self.model_name = model_name
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
    
    def __call__(self, video, return_tensors: str = "pt", **kwargs):
        """
        Process video frames
        
        Args:
            video: List of frames (each frame is a numpy array or PIL Image)
                  or tensor of shape (T, C, H, W) or (B, T, C, H, W)
            return_tensors: Return format ("pt" for PyTorch)
            **kwargs: Additional arguments for processor
        
        Returns:
            Processed inputs with pixel_values
        """
        inputs = self.processor(video, return_tensors=return_tensors, **kwargs)
        
        # Convert to (B, C, T, H, W) format if needed
        if 'pixel_values' in inputs:
            pixel_values = inputs['pixel_values']
            if pixel_values.dim() == 5:
                # (B, T, C, H, W) -> (B, C, T, H, W)
                if pixel_values.shape[1] != 3:
                    inputs['pixel_values'] = pixel_values.permute(0, 2, 1, 3, 4)
        
        return inputs


def create_videomae_model(
    num_class: int,
    model_name: str = "OpenGVLab/VideoMAEv2-Base",
    freeze_backbone: bool = False,
    dropout: float = 0.1,
    **kwargs
) -> VideoMAEForSignLanguage:
    """
    Factory function to create VideoMAE model for sign language recognition
    
    Args:
        num_class: Number of output classes
        model_name: Hugging Face model name or path
        freeze_backbone: Whether to freeze the pretrained backbone
        dropout: Dropout rate for classification head
        **kwargs: Additional arguments for VideoMAEForSignLanguage
    
    Returns:
        VideoMAEForSignLanguage model instance
    """
    model = VideoMAEForSignLanguage(
        model_name=model_name,
        num_class=num_class,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        **kwargs
    )
    
    return model