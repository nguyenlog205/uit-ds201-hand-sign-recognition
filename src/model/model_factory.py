"""
Model Factory
Factory pattern for creating models from config
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from ..data.gcn.graph_constructor import SkeletonGraph
from .stgcn import STGCN, create_stgcn_model
from .ha_gcn import HAGCN, create_hagcn_model
from .poseformer import PoseFormer, create_poseformer_model
from .VideoMAE import VideoMAEForSignLanguage, create_videomae_model
from .bi_lstm.bi_lstm import BiLSTM
from .resnet_lstm.model import RGB_based_model
from .signbert import SignBERT, create_signbert_model
from .i3d import I3D, create_i3d_model


def create_model(config: Dict[str, Any], num_classes: Optional[int] = None) -> nn.Module:
    """
    Create a GCN model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes (overrides config if provided)
    
    Returns:
        PyTorch model instance
    """
    model_type = config.get('model', {}).get('type', 'stgcn').lower()
    
    if model_type == 'stgcn':
        return create_stgcn_from_config(config, num_classes)
    elif model_type == 'ha_gcn':
        return create_hagcn_from_config(config, num_classes)
    elif model_type == 'poseformer':
        return create_poseformer_from_config(config, num_classes)
    elif model_type == 'videomae':
        return create_videomae_from_config(config, num_classes)
    elif model_type == 'bi_lstm':
        return create_bilstm_from_config(config, num_classes)
    elif model_type == 'resnet_lstm':
        return create_resnet_lstm_from_config(config, num_classes)
    elif model_type == 'signbert':
        return create_signbert_from_config(config, num_classes)
    elif model_type == 'i3d':
        return create_i3d_from_config(config, num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_stgcn_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> STGCN:
    """
    Create ST-GCN model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        STGCN model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    in_channels = model_config.get('in_channels', 3)
    num_nodes = model_config.get('num_nodes', 27)
    num_person = model_config.get('num_person', 1)
    window_size = model_config.get('window_size', 64)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Graph configuration
    skeleton_layout = model_config.get('skeleton_layout', 'mediapipe_27')
    adjacency_strategy = model_config.get('adjacency_strategy', 'spatial')
    
    # Create skeleton graph and get adjacency matrix
    skeleton = SkeletonGraph(layout=skeleton_layout)
    A = skeleton.get_adjacency_matrix(strategy=adjacency_strategy)
    
    # Convert to tensor
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).float()
    
    # Ensure correct shape: (K, V, V) where K is number of partitions
    if A.dim() == 2:
        # If 2D, add partition dimension (uniform strategy)
        A = A.unsqueeze(0)  # (1, V, V)
    elif A.dim() == 3:
        # Already 3D: (K, V, V)
        pass
    else:
        raise ValueError(f"Invalid adjacency matrix shape: {A.shape}")
    
    # Model options
    use_data_bn = model_config.get('use_data_bn', True)
    mask_learning = model_config.get('mask_learning', False)
    use_local_bn = model_config.get('use_local_bn', False)
    temporal_kernel_size = model_config.get('temporal_kernel_size', 9)
    dropout = model_config.get('dropout', 0.5)
    
    # Backbone configuration
    backbone_config = model_config.get('backbone_config', None)
    if backbone_config is not None:
        # Convert list of lists to list of tuples
        backbone_config = [tuple(c) for c in backbone_config]
    
    # Create model
    model = STGCN(
        in_channels=in_channels,
        num_class=num_class,
        num_nodes=num_nodes,
        num_person=num_person,
        window_size=window_size,
        use_data_bn=use_data_bn,
        backbone_config=backbone_config,
        A=A,
        mask_learning=mask_learning,
        use_local_bn=use_local_bn,
        temporal_kernel_size=temporal_kernel_size,
        dropout=dropout,
    )
    
    return model


def create_hagcn_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> HAGCN:
    """
    Create HA-GCN model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        HAGCN model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    in_channels = model_config.get('in_channels', 3)
    num_nodes = model_config.get('num_nodes', 27)
    num_person = model_config.get('num_person', 1)
    window_size = model_config.get('window_size', 64)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Graph configuration
    skeleton_layout = model_config.get('skeleton_layout', 'mediapipe_27')
    adjacency_strategy = model_config.get('adjacency_strategy', 'spatial')
    
    # Create skeleton graph and get adjacency matrix
    skeleton = SkeletonGraph(layout=skeleton_layout)
    A = skeleton.get_adjacency_matrix(strategy=adjacency_strategy)
    
    # Convert to tensor
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A).float()
    
    # Ensure correct shape: (K, V, V) where K is number of partitions
    if A.dim() == 2:
        # If 2D, add partition dimension (uniform strategy)
        A = A.unsqueeze(0)  # (1, V, V)
    elif A.dim() == 3:
        # Already 3D: (K, V, V)
        pass
    else:
        raise ValueError(f"Invalid adjacency matrix shape: {A.shape}")
    
    # Model options
    block_size = model_config.get('block_size', 41)
    use_attention = model_config.get('use_attention', True)
    
    # Create model
    model = HAGCN(
        in_channels=in_channels,
        num_class=num_class,
        num_nodes=num_nodes,
        num_person=num_person,
        window_size=window_size,
        A=A,
        block_size=block_size,
        use_attention=use_attention,
    )
    
    return model


def create_poseformer_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> PoseFormer:
    """
    Create PoseFormer model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        PoseFormer model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters with correct names for PoseFormer
    in_channels = model_config.get('in_channels', 3)
    num_joints = model_config.get('num_nodes', model_config.get('num_joints', 27))
    num_frames = model_config.get('window_size', model_config.get('num_frames', 64))
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Transformer-specific parameters (map to PoseFormer parameter names)
    d_model_value = model_config.get('d_model', None)
    embed_dim_ratio_default = 32  # Safe default for skeleton data
    
    if 'embed_dim_ratio' in model_config:
        embed_dim_ratio = model_config['embed_dim_ratio']
    elif d_model_value is not None:
        # If d_model is too large (>64), it's likely meant for a different purpose
        # Use safe default instead to prevent parameter explosion
        if d_model_value > 64:
            print(f"Warning: d_model={d_model_value} is too large for embed_dim_ratio. Using safe default {embed_dim_ratio_default} instead.")
            embed_dim_ratio = embed_dim_ratio_default
        else:
            embed_dim_ratio = d_model_value
    else:
        embed_dim_ratio = embed_dim_ratio_default
    
    depth = model_config.get('depth', model_config.get('num_layers', 4))
    num_heads = model_config.get('num_heads', model_config.get('nhead', 8))
    mlp_ratio_raw = model_config.get('mlp_ratio', model_config.get('dim_feedforward', 2.0))
    if isinstance(mlp_ratio_raw, (int, float)) and mlp_ratio_raw > 10:
        mlp_ratio = mlp_ratio_raw / 864.0
        print(f"Warning: dim_feedforward={mlp_ratio_raw} appears to be absolute dimension. Converting to mlp_ratio={mlp_ratio:.2f}")
    else:
        mlp_ratio = float(mlp_ratio_raw)
    drop_rate = model_config.get('drop_rate', model_config.get('dropout', 0.1))
    
    # Additional optional parameters
    qkv_bias = model_config.get('qkv_bias', True)
    attn_drop_rate = model_config.get('attn_drop_rate', 0.1)
    drop_path_rate = model_config.get('drop_path_rate', 0.1)
    
    # Create model using factory function
    model = create_poseformer_model(
        num_class=num_class,
        num_joints=num_joints,
        num_frames=num_frames,
        in_channels=in_channels,
        embed_dim_ratio=embed_dim_ratio,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        qkv_bias=qkv_bias,
    )
    
    return model


def create_videomae_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> VideoMAEForSignLanguage:
    """
    Create VideoMAE model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        VideoMAEForSignLanguage model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    model_name = model_config.get('model_name', 'OpenGVLab/VideoMAEv2-Base')
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Model options
    freeze_backbone = model_config.get('freeze_backbone', False)
    dropout = model_config.get('dropout', 0.1)
    hidden_dim = model_config.get('hidden_dim', None)
    
    # Create model
    model = VideoMAEForSignLanguage(
        model_name=model_name,
        num_class=num_class,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        hidden_dim=hidden_dim,
    )
    
    return model


def create_bilstm_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> BiLSTM:
    """
    Create Bi-LSTM model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        BiLSTM model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    num_joints = model_config.get('num_joints', 27)
    num_dims = model_config.get('num_dims', 3)
    lstm_hidden_size = model_config.get('lstm_hidden_size', 512)
    lstm_num_layers = model_config.get('lstm_num_layers', 2)
    lstm_dropout_rate = model_config.get('lstm_dropout_rate', 0.5)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Create model
    model = BiLSTM(
        num_joints=num_joints,
        num_dims=num_dims,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout_rate=lstm_dropout_rate,
        num_classes=num_class,
    )
    
    return model


def create_resnet_lstm_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> RGB_based_model:
    """
    Create ResNet-LSTM model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes 
    
    Returns:
        RGB_based_model model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    resnet_dropout_rate = model_config.get('resnet_dropout_rate', 0.5)
    lstm_hidden_size = model_config.get('lstm_hidden_size', 512)
    lstm_num_layers = model_config.get('lstm_num_layers', 1)
    lstm_dropout_rate = model_config.get('lstm_dropout_rate', 0.5)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Create model
    model = RGB_based_model(
        resnet_dropout_rate=resnet_dropout_rate,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout_rate=lstm_dropout_rate,
        num_classes=num_class,
    )
    
    return model


def create_signbert_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> SignBERT:
    """
    Create SignBERT model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes
    
    Returns:
        SignBERT model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    num_joints = model_config.get('num_joints', 27)
    num_coords = model_config.get('num_coords', 3)
    num_frames = model_config.get('num_frames', 64)
    embed_dim = model_config.get('embed_dim', 256)
    depth = model_config.get('depth', 6)
    num_heads = model_config.get('num_heads', 8)
    mlp_ratio = model_config.get('mlp_ratio', 4.0)
    drop_rate = model_config.get('drop_rate', 0.1)
    use_pretrained = model_config.get('use_pretrained', False)
    pretrained_path = model_config.get('pretrained_path', None)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 100)
    
    # Create model
    model = create_signbert_model(
        num_classes=num_class,
        num_joints=num_joints,
        num_coords=num_coords,
        num_frames=num_frames,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        use_pretrained=use_pretrained,
        pretrained_path=pretrained_path,
    )
    
    return model


def create_i3d_from_config(
    config: Dict[str, Any],
    num_classes: Optional[int] = None
) -> I3D:
    """
    Create I3D model from configuration
    
    Args:
        config: Model configuration dictionary
        num_classes: Number of output classes
    
    Returns:
        I3D model instance
    """
    model_config = config.get('model', {})
    
    # Extract parameters
    in_channels = model_config.get('in_channels', 3)
    dropout_keep_prob = model_config.get('dropout_keep_prob', 0.5)
    use_pretrained = model_config.get('use_pretrained', False)
    pretrained_path = model_config.get('pretrained_path', None)
    
    # Number of classes
    if num_classes is not None:
        num_class = num_classes
    else:
        num_class = model_config.get('num_class', 400)
    
    # Create model
    model = create_i3d_model(
        num_classes=num_class,
        in_channels=in_channels,
        dropout_keep_prob=dropout_keep_prob,
        use_pretrained=use_pretrained,
        pretrained_path=pretrained_path,
    )
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_type': type(model).__name__,
    }