"""
Data Augmentation Module
Augmentation techniques for graph data (temporal, spatial, noise) and RGB video
"""

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Dict, List, Optional, Union


def apply_augmentations(
    poses: np.ndarray,
    augmentation_configs: List[Dict]
) -> np.ndarray:
    """
    Apply a list of augmentations to pose sequence
    
    Args:
        poses: (T, V, C) pose sequence
        augmentation_configs: List of augmentation configurations
        
    Returns:
        Augmented poses
    """
    augmented = poses.copy()
    
    for aug_config in augmentation_configs:
        aug_type = aug_config.get('type')
        
        if aug_type == 'temporal_jitter':
            augmented = temporal_jitter(
                augmented,
                max_jitter=aug_config.get('max_jitter', 3)
            )
        elif aug_type == 'spatial_transform':
            augmented = spatial_transform(
                augmented,
                translation=aug_config.get('translation', 0.1),
                rotation=aug_config.get('rotation', 0),
                scale=aug_config.get('scale', 1.0),
            )
        elif aug_type == 'noise_injection':
            augmented = noise_injection(
                augmented,
                std=aug_config.get('std', 0.01)
            )
        elif aug_type == 'random_flip':
            augmented = random_flip(
                augmented,
                prob=aug_config.get('prob', 0.5),
                flip_pairs=aug_config.get('flip_pairs', [])
            )
        elif aug_type == 'temporal_scale':
            augmented = temporal_scale(
                augmented,
                scale_range=aug_config.get('scale_range', [0.9, 1.1])
            )
        else:
            print(f"Warning: Unknown augmentation type: {aug_type}")
    
    return augmented


def temporal_jitter(poses: np.ndarray, max_jitter: int = 3) -> np.ndarray:
    """
    Apply temporal jittering (frame skipping/duplication)
    
    Args:
        poses: (T, V, C) pose sequence
        max_jitter: Maximum number of frames to jitter
        
    Returns:
        Jittered poses
    """
    T, V, C = poses.shape
    
    if max_jitter == 0:
        return poses
    
    # Random jitter for each frame
    jittered = []
    for t in range(T):
        jitter = np.random.randint(-max_jitter, max_jitter + 1)
        t_jittered = np.clip(t + jitter, 0, T - 1)
        jittered.append(poses[t_jittered])
    
    return np.array(jittered)


def spatial_transform(
    poses: np.ndarray,
    translation: float = 0.1,
    rotation: float = 0,
    scale: Union[float, List[float]] = 1.0,
) -> np.ndarray:
    """
    Apply spatial transformations (translation, rotation, scale)
    
    Args:
        poses: (T, V, C) pose sequence
        translation: Maximum translation ratio
        rotation: Rotation angle in degrees
        scale: Scale factor or [min, max] range
        
    Returns:
        Transformed poses
    """
    T, V, C = poses.shape
    transformed = poses.copy()
    
    # Extract coordinates (x, y)
    coords = transformed[..., :2]  # (T, V, 2)
    
    # Translation
    if translation > 0:
        tx = np.random.uniform(-translation, translation)
        ty = np.random.uniform(-translation, translation)
        coords = coords + np.array([tx, ty])
    
    # Rotation
    if rotation > 0:
        angle = np.random.uniform(-rotation, rotation) * np.pi / 180
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        coords = coords @ rotation_matrix.T
    
    # Scale
    if isinstance(scale, (list, tuple)):
        scale_factor = np.random.uniform(scale[0], scale[1])
    else:
        scale_factor = scale
    
    if scale_factor != 1.0:
        coords = coords * scale_factor
    
    transformed[..., :2] = coords
    
    return transformed


def noise_injection(poses: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Inject Gaussian noise to coordinates
    
    Args:
        poses: (T, V, C) pose sequence
        std: Standard deviation of noise
        
    Returns:
        Noisy poses
    """
    noise = np.random.normal(0, std, poses[..., :2].shape)
    noisy = poses.copy()
    noisy[..., :2] = noisy[..., :2] + noise
    return noisy


def random_flip(
    poses: np.ndarray,
    prob: float = 0.5,
    flip_pairs: Optional[List[List[int]]] = None
) -> np.ndarray:
    """
    Randomly flip poses horizontally (left-right)
    
    Args:
        poses: (T, V, C) pose sequence
        prob: Probability of flipping
        flip_pairs: List of [left_idx, right_idx] pairs to swap
        
    Returns:
        Flipped poses
    """
    if np.random.rand() > prob:
        return poses
    
    flipped = poses.copy()
    coords = flipped[..., :2]  # (T, V, 2)
    
    # Flip x-coordinates (mirror horizontally)
    coords[..., 0] = -coords[..., 0]
    
    # Swap left-right pairs if provided
    if flip_pairs:
        for left_idx, right_idx in flip_pairs:
            coords[:, [left_idx, right_idx], :] = coords[:, [right_idx, left_idx], :]
    
    flipped[..., :2] = coords
    
    return flipped


def temporal_scale(
    poses: np.ndarray,
    scale_range: List[float] = [0.9, 1.1]
) -> np.ndarray:
    """
    Scale temporal dimension (speed up/slow down)
    
    Args:
        poses: (T, V, C) pose sequence
        scale_range: [min_scale, max_scale] range
        
    Returns:
        Temporally scaled poses
    """
    T, V, C = poses.shape
    
    scale = np.random.uniform(scale_range[0], scale_range[1])
    new_length = int(T * scale)
    
    if new_length == T:
        return poses
    
    # Resample using linear interpolation
    old_indices = np.arange(T)
    new_indices = np.linspace(0, T - 1, new_length)
    
    scaled = np.zeros((new_length, V, C))
    for v in range(V):
        for c in range(C):
            scaled[:, v, c] = np.interp(new_indices, old_indices, poses[:, v, c])
    
    # Pad or crop to original length
    if new_length < T:
        # Pad with last frame
        padding = np.repeat(scaled[-1:], T - new_length, axis=0)
        scaled = np.concatenate([scaled, padding], axis=0)
    else:
        # Crop
        scaled = scaled[:T]
    
    return scaled
    

def apply_rgb_augmentations(
    video_tensor: torch.Tensor,
    augmentation_configs: List[Dict],
    is_training: bool = True
) -> torch.Tensor:
    """
    Apply augmentations to RGB video tensor (C, T, H, W)
    
    Args:
        video_tensor: Video tensor of shape (C, T, H, W)
        augmentation_configs: List of augmentation configurations
        is_training: Whether to apply augmentations (only applied if True)
        
    Returns:
        Augmented video tensor
    """
    if not is_training or len(augmentation_configs) == 0:
        return video_tensor
    
    # Convert to (T, C, H, W) for easier frame-wise processing
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W)
    
    for aug_config in augmentation_configs:
        aug_type = aug_config.get('type', '')
        
        if aug_type == 'random_crop':
            crop_size = aug_config.get('size', 224)
            # Random crop for each frame (or same crop for all frames)
            if np.random.rand() > 0.5:
                h, w = video_tensor.shape[2], video_tensor.shape[3]
                top = np.random.randint(0, max(1, h - crop_size))
                left = np.random.randint(0, max(1, w - crop_size))
                video_tensor = F.crop(video_tensor, top, left, crop_size, crop_size)
        
        elif aug_type == 'center_crop':
            crop_size = aug_config.get('size', 224)
            video_tensor = F.center_crop(video_tensor, crop_size)
        
        elif aug_type == 'random_horizontal_flip':
            if np.random.rand() > 0.5:
                video_tensor = F.hflip(video_tensor)
        
        elif aug_type == 'color_jitter':
            brightness = aug_config.get('brightness', 0.1)
            contrast = aug_config.get('contrast', 0.1)
            saturation = aug_config.get('saturation', 0.1)
            hue = aug_config.get('hue', 0.05)
            jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
            # Apply to each frame
            for t in range(video_tensor.shape[0]):
                video_tensor[t] = jitter(video_tensor[t])
    
    # Convert back to (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    return video_tensor


if __name__ == "__main__":
    # Example usage for skeleton augmentation
    T, V, C = 64, 25, 3
    poses = np.random.randn(T, V, C)
    
    augmentation_configs = [
        {'type': 'temporal_jitter', 'max_jitter': 3},
        {'type': 'spatial_transform', 'translation': 0.1, 'rotation': 15, 'scale': [0.9, 1.1]},
        {'type': 'noise_injection', 'std': 0.01},
    ]
    
    augmented = apply_augmentations(poses, augmentation_configs)
    print(f"Original shape: {poses.shape}")
    print(f"Augmented shape: {augmented.shape}")
    
    # Example usage for RGB augmentation
    video_tensor = torch.rand(3, 16, 224, 224)  # (C, T, H, W)
    rgb_aug_configs = [
        {'type': 'random_crop', 'size': 224},
        {'type': 'random_horizontal_flip'},
        {'type': 'color_jitter', 'brightness': 0.1, 'contrast': 0.1},
    ]
    augmented_video = apply_rgb_augmentations(video_tensor, rgb_aug_configs, is_training=True)
    print(f"RGB Video shape: {augmented_video.shape}")