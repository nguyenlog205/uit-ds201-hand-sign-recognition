"""
Preprocessed dataset adapter for skeleton_keypoint models 
Converts pre-processed graph data to keypoint format
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import Optional, Dict, List
import random

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.data.augmentation import apply_augmentations


class PreprocessedKeypointDataset(Dataset):
    """
    Dataset adapter to load pre-processed graph data and convert to keypoint format
    for Bi-LSTM and PoseFormer models
    """
    
    def __init__(
        self,
        json_file: str,
        data_root: str,
        num_frames: int = 64,
        is_training: bool = True,
        duplicate_factor: int = 10,
    ):
        """
        Args:
            json_file: Path to train.json, val.json, or test.json
            data_root: Root folder containing .npz files
            num_frames: Number of frames to sample
            is_training: Whether in training mode
            duplicate_factor: Number of times to duplicate data (only for training)
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.is_training = is_training
        self.duplicate_factor = duplicate_factor if is_training else 1
        
        with open(json_file, 'r') as f:
            self.file_list = json.load(f)
        
        if is_training:
            self.augmentation_configs = [
                {'type': 'temporal_jitter', 'max_jitter': 5},
                {'type': 'spatial_transform', 
                 'translation': 0.15,
                 'rotation': 20,
                 'scale': [0.85, 1.15]},
                {'type': 'noise_injection', 'std': 0.02},
                {'type': 'temporal_scale', 'scale_range': [0.8, 1.2]},
            ]
        else:
            self.augmentation_configs = []
        
        if is_training and duplicate_factor > 1:
            self.file_list = self.file_list * duplicate_factor
        
        print(f"PreprocessedKeypointDataset initialized: Mode={'TRAINING' if is_training else 'VALIDATION'}, "
              f"Samples={len(self.file_list) // duplicate_factor if is_training else len(self.file_list)} -> {len(self.file_list)}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        item = self.file_list[idx]
        file_path = self.data_root / item['file_path']
        data = np.load(file_path, allow_pickle=True)
        
        x = data['x']
        label = int(data['label']) if 'label' in data else item['label']
        x = torch.from_numpy(x).float()
        if self.num_frames and x.shape[0] != self.num_frames:
            T = x.shape[0]
            if T > self.num_frames:
                if self.is_training:
                    start_idx = random.randint(0, T - self.num_frames)
                    x = x[start_idx:start_idx + self.num_frames]
                else:
                    start_idx = (T - self.num_frames) // 2
                    x = x[start_idx:start_idx + self.num_frames]
            elif T < self.num_frames:
                padding_size = self.num_frames - T
                last_frame = x[-1:].repeat(padding_size, 1, 1)
                x = torch.cat([x, last_frame], dim=0)
        
        if self.is_training and len(self.augmentation_configs) > 0:
            x_np = x.numpy()
            x_np = apply_augmentations(x_np, self.augmentation_configs)
            x = torch.from_numpy(x_np).float()
        
        label = torch.tensor(label, dtype=torch.long)
        return x, label


class PreprocessedGraphDataset(Dataset):
    """
    Dataset to load from pre-processed .npz files
    Includes strong data augmentation to increase the number of samples
    """
    
    def __init__(
        self,
        json_file: str,
        data_root: str,
        num_frames: int = 64,
        is_training: bool = True,
        augmentation_configs: Optional[List[Dict]] = None,
        duplicate_factor: int = 10,
    ):
        """
        Args:
            json_file: Path to train.json or val.json
            data_root: Root folder containing .npz files
            num_frames: Number of frames to sample
            is_training: Whether in training mode (only augment during training)
            augmentation_configs: List of augmentation configs
            duplicate_factor: Number of times to duplicate data (only applied during training)
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.is_training = is_training
        self.duplicate_factor = duplicate_factor if is_training else 1
        
        with open(json_file, 'r') as f:
            self.file_list = json.load(f)
        
        if augmentation_configs is None and is_training:
            self.augmentation_configs = [
                {'type': 'temporal_jitter', 'max_jitter': 5},
                {'type': 'spatial_transform', 
                 'translation': 0.15,
                 'rotation': 20,
                 'scale': [0.85, 1.15]},
                {'type': 'noise_injection', 'std': 0.02},
                {'type': 'temporal_scale', 'scale_range': [0.8, 1.2]},
            ]
        else:
            self.augmentation_configs = augmentation_configs or []
        
        if is_training and duplicate_factor > 1:
            self.file_list = self.file_list * duplicate_factor
        
        print(f"Dataset initialized: Mode={'TRAINING' if is_training else 'VALIDATION'}, "
              f"Samples={len(self.file_list) // duplicate_factor if is_training else len(self.file_list)} -> {len(self.file_list)}, "
              f"Augmentations={len(self.augmentation_configs) if is_training else 0}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        item = self.file_list[idx]
        file_path = self.data_root / item['file_path']
        data = np.load(file_path, allow_pickle=True)
        
        x = data['x']
        label = int(data['label']) if 'label' in data else item['label']
        x = torch.from_numpy(x).float()
        if self.num_frames and x.shape[0] != self.num_frames:
            T = x.shape[0]
            if T > self.num_frames:
                if self.is_training:
                    start_idx = random.randint(0, T - self.num_frames)
                    x = x[start_idx:start_idx + self.num_frames]
                else:
                    start_idx = (T - self.num_frames) // 2
                    x = x[start_idx:start_idx + self.num_frames]
            elif T < self.num_frames:
                padding_size = self.num_frames - T
                last_frame = x[-1:].repeat(padding_size, 1, 1)
                x = torch.cat([x, last_frame], dim=0)
        
        if self.is_training and len(self.augmentation_configs) > 0:
            x_np = x.numpy()
            x_np = apply_augmentations(x_np, self.augmentation_configs)
            x = torch.from_numpy(x_np).float()
        
        label = torch.tensor(label, dtype=torch.long)
        return {
            'x': x,
            'y': label,
            'label': label
        }


def create_preprocessed_datasets(
    data_root: str = "DATA/Processed_Pose",
    num_frames: int = 64,
    train_duplicate_factor: int = 10,
    augmentation_configs: Optional[List[Dict]] = None,
    include_test: bool = False,
):
    """
    Create train, val and (optional) test datasets from pre-processed files
    
    Args:
        data_root: Root folder containing train.json, val.json, test.json and .npz files
        num_frames: Number of frames
        train_duplicate_factor: Number of times to duplicate train data
        augmentation_configs: Custom augmentation configs
        include_test: Whether to create test dataset
    
    Returns:
        train_dataset, val_dataset, (test_dataset if include_test=True)
    """
    data_root = Path(data_root)
    
    train_json = data_root / "train.json"
    val_json = data_root / "val.json"
    test_json = data_root / "test.json"
    
    if not train_json.exists():
        raise FileNotFoundError(
            f"Train JSON not found: {train_json}.\n"
            f"Please run: python -m src.data.preprocess_data --raw_path DATA/Segmented --output_path {data_root}"
        )
    
    if not val_json.exists():
        raise FileNotFoundError(
            f"Val JSON not found: {val_json}.\n"
            f"Please run: python -m src.data.preprocess_data --raw_path DATA/Segmented --output_path {data_root}"
        )
    
    train_dataset = PreprocessedGraphDataset(
        json_file=str(train_json),
        data_root=str(data_root),
        num_frames=num_frames,
        is_training=True,
        augmentation_configs=augmentation_configs,
        duplicate_factor=train_duplicate_factor,
    )
    
    val_dataset = PreprocessedGraphDataset(
        json_file=str(val_json),
        data_root=str(data_root),
        num_frames=num_frames,
        is_training=False,
        augmentation_configs=None,
        duplicate_factor=1,
    )
    
    if include_test:
        if not test_json.exists():
            raise FileNotFoundError(
                f"Test JSON not found: {test_json}.\n"
                f"Please run: python -m src.data.preprocess_data --raw_path DATA/Segmented --output_path {data_root}"
            )
        
        test_dataset = PreprocessedGraphDataset(
            json_file=str(test_json),
            data_root=str(data_root),
            num_frames=num_frames,
            is_training=False,
            augmentation_configs=None,
            duplicate_factor=1,
        )
        
        return train_dataset, val_dataset, test_dataset
    
    return train_dataset, val_dataset
