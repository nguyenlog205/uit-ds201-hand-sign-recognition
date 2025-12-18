"""
Configuration management using Pydantic for type validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, List
import yaml
from pathlib import Path


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    type: Literal["stgcn", "ha_gcn", "poseformer", "videomae", "bi_lstm", "resnet_lstm", "signbert", "i3d"] = Field(
        ..., description="Model type"
    )
    
    # Common parameters
    num_class: Optional[int] = Field(None, description="Number of classes (auto-detected if None)")
    in_channels: int = Field(3, description="Input channels (3 for xyz coordinates)")
    num_nodes: int = Field(27, description="Number of skeleton nodes/keypoints")
    num_person: int = Field(1, description="Number of persons")
    window_size: int = Field(64, description="Temporal window size (number of frames)")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    
    # Graph-based models (ST-GCN, HA-GCN)
    skeleton_layout: Optional[str] = Field("mediapipe_27", description="Skeleton layout type")
    adjacency_strategy: Optional[str] = Field("spatial", description="Graph adjacency strategy")
    use_data_bn: Optional[bool] = Field(True, description="Use data batch normalization")
    mask_learning: Optional[bool] = Field(False, description="Enable mask learning")
    use_local_bn: Optional[bool] = Field(False, description="Use local batch normalization")
    temporal_kernel_size: Optional[int] = Field(9, description="Temporal convolution kernel size")
    backbone_config: Optional[List[List[int]]] = Field(None, description="Backbone configuration")
    
    # HA-GCN specific
    block_size: Optional[int] = Field(41, description="Block size for HA-GCN")
    use_attention: Optional[bool] = Field(True, description="Use attention in HA-GCN")
    
    # PoseFormer specific
    d_model: Optional[int] = Field(256, description="Transformer model dimension")
    nhead: Optional[int] = Field(8, description="Number of attention heads")
    num_layers: Optional[int] = Field(6, description="Number of transformer layers")
    dim_feedforward: Optional[int] = Field(1024, description="Feed-forward dimension")
    use_spatial_pooling: Optional[bool] = Field(True, description="Use spatial pooling")
    
    # VideoMAE specific
    model_name: Optional[str] = Field("OpenGVLab/VideoMAEv2-Base", description="VideoMAE model name")
    freeze_backbone: Optional[bool] = Field(False, description="Freeze VideoMAE backbone")
    hidden_dim: Optional[int] = Field(None, description="Hidden dimension for classifier")
    
    # Bi-LSTM specific
    num_joints: Optional[int] = Field(27, description="Number of joints")
    num_dims: Optional[int] = Field(3, description="Number of dimensions (x, y, z)")
    lstm_hidden_size: Optional[int] = Field(512, description="LSTM hidden size")
    lstm_num_layers: Optional[int] = Field(2, description="Number of LSTM layers")
    lstm_dropout_rate: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="LSTM dropout rate")
    
    # ResNet-LSTM specific
    resnet_dropout_rate: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="ResNet dropout rate")
    
    # SignBERT specific
    num_coords: Optional[int] = Field(3, description="Number of coordinates per joint (x, y, z)")
    embed_dim: Optional[int] = Field(256, description="Embedding dimension for SignBERT")
    depth: Optional[int] = Field(6, description="Number of transformer blocks")
    num_heads: Optional[int] = Field(8, description="Number of attention heads")
    mlp_ratio: Optional[float] = Field(4.0, description="MLP hidden dimension ratio")
    drop_rate: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Dropout rate for SignBERT")
    use_pretrained: Optional[bool] = Field(False, description="Use pretrained weights")
    pretrained_path: Optional[str] = Field(None, description="Path to pretrained checkpoint")
    
    # I3D specific
    dropout_keep_prob: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Dropout keep probability for I3D")


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    batch_size: int = Field(32, gt=0, description="Batch size")
    num_epochs: int = Field(50, gt=0, description="Number of epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    optimizer: Literal["adam", "sgd", "adamw"] = Field("adam", description="Optimizer type")
    weight_decay: float = Field(1e-4, ge=0.0, description="Weight decay (L2 regularization)")
    
    # Learning rate scheduler
    scheduler: Optional[Literal["cosine", "step", "plateau", "warmup_cosine"]] = Field(
        "cosine", description="Learning rate scheduler type"
    )
    scheduler_params: Optional[Dict] = Field(
        None, description="Additional scheduler parameters"
    )
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = Field(5.0, gt=0.0, description="Gradient clipping threshold")
    
    # Warmup (for transformers)
    warmup_steps: Optional[int] = Field(None, description="Warmup steps for scheduler")
    warmup_epochs: Optional[int] = Field(0, description="Warmup epochs")
    
    # Regularization
    label_smoothing: float = Field(0.0, ge=0.0, le=1.0, description="Label smoothing factor")
    dropout: Optional[float] = Field(None, ge=0.0, le=1.0, description="Additional dropout (overrides model dropout if set)")
    
    # Evaluation and checkpointing
    eval_every: int = Field(1, gt=0, description="Evaluate every N epochs")
    save_every: int = Field(5, gt=0, description="Save checkpoint every N epochs")
    save_best: bool = Field(True, description="Save best model based on validation metric")
    early_stopping_patience: Optional[int] = Field(None, description="Early stopping patience (None = disabled)")
    
    # Mixed precision training
    use_amp: bool = Field(False, description="Use automatic mixed precision")


class DataConfig(BaseModel):
    """Data paths and processing configuration"""
    # Data paths
    raw_path: str = Field(..., description="Path to raw video data")
    metadata: str = Field(..., description="Path to metadata directory")
    metadata_val: Optional[str] = Field(None, description="Path to validation metadata directory (if separate from train)")
    skeleton_path: Optional[str] = Field(None, description="Path to pre-extracted skeleton data")
    
    # Train/Val split
    train_val_split: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Train/validation split ratio (only used if metadata_val is None)")
    random_split: bool = Field(True, description="Use random split for train/val (if metadata_val is None)")
    
    # Dataset type
    dataset_type: Literal["rgb", "skeleton", "skeleton_keypoint", "skeleton_graph", "videomae"] = Field(
        "skeleton_keypoint", description="Dataset type"
    )
    
    # Processing parameters
    num_frames: Optional[int] = Field(None, description="Number of frames to sample (None = use all)")
    target_size: Optional[List[int]] = Field([224, 224], description="Target image size [H, W]")
    use_imagenet_norm: bool = Field(False, description="Use ImageNet normalization")
    normalize: bool = Field(True, description="Normalize skeleton coordinates")
    
    # Graph construction (for graph-based models)
    skeleton_layout: Optional[str] = Field("mediapipe_27", description="Skeleton layout")
    graph_strategy: Optional[str] = Field("spatial", description="Graph construction strategy")
    
    # Augmentation
    augmentations: Optional[List[Dict]] = Field(None, description="Data augmentation configurations")
    
    # VideoMAE specific
    processor_name: Optional[str] = Field(None, description="VideoMAE processor name")
    
    # FPS (for temporal sampling)
    fps: Optional[float] = Field(30.0, description="Video FPS")


class Config(BaseModel):
    """
    Main configuration class for hand sign recognition models.
    
    Usage:
        config = Config.from_yaml('configs/stgcn.yaml')
        print(config.model.type)
        print(config.training.batch_size)
        print(config.data.raw_path)
    """
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str = Field("cuda", description="Device: cuda or cpu")
    seed: Optional[int] = Field(42, description="Random seed")
    output_dir: str = Field("./logs", description="Output directory for logs and checkpoints")
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'Config':
        """
        Load configuration from YAML file with Pydantic validation.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object with validated fields
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValidationError: If configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_yaml(self, yaml_path: str | Path):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
    
    def get(self, key: str, default=None):
        """
        Get nested configuration value using dot notation.
        
        Args:
            key: Dot-separated key (e.g., 'model.type', 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow extra fields
        validate_assignment = True  # Validate on assignment