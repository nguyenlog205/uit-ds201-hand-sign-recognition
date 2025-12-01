"""
Model Factory
Factory pattern for creating GCN models from config
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

from ..data.graph_constructor import SkeletonGraph
from .stgcn import STGCN, create_stgcn_model
from .ha_gcn import HAGCN, create_hagcn_model
from .base_gcn import BaseGCN


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
    skeleton_layout = model_config.get('skeleton_layout', 'sign_language_27')
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
    skeleton_layout = model_config.get('skeleton_layout', 'sign_language_27')
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