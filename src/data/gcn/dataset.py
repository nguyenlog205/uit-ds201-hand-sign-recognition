"""
GCN Dataset Module
PyTorch Dataset classes for graph-based sign language recognition
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

import sys
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data.gcn.graph_constructor import GraphConstructor  
from data.augmentation import apply_augmentations 


class GraphDataset(Dataset):
    """
    PyTorch Dataset for graph-based sign language recognition
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_frames: int = 64,
        temporal_stride: int = 1,
        random_temporal_crop: bool = False,
        augmentation: Optional[List[Dict]] = None,
        normalize: bool = True,
        skeleton_layout: str = "mediapipe_27",
        graph_strategy: str = "uniform",
    ):
        """
        Initialize dataset

        Args:
            data_dir: Directory containing processed graph data
            split: Dataset split ("train", "val", "test")
            num_frames: Number of frames to sample
            temporal_stride: Stride for temporal sampling
            random_temporal_crop: Whether to randomly crop temporal sequence
            augmentation: List of augmentation configurations
            normalize: Whether to normalize coordinates
            skeleton_layout: Skeleton layout type
            graph_strategy: Graph adjacency strategy
        """

        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.random_temporal_crop = random_temporal_crop
        self.augmentation = augmentation or []
        self.normalize = normalize

        self.graph_constructor = GraphConstructor(
            skeleton_layout=skeleton_layout,
            normalize=normalize,
        )
        self.graph_strategy = graph_strategy

        self.samples = self._load_samples()
        self.class_names = self._load_class_names()

    def _load_samples(self) -> List[Dict]:
        """
        Load list of samples from annotation file or directory
        """
        annotation_file = self.data_dir / f"{self.split}.json"
        if annotation_file.exists():
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            return annotations

        split_dir = self.data_dir / self.split
        if split_dir.exists():
            samples = []
            for npz_file in sorted(split_dir.glob("*.npz")):
                label = self._infer_label(npz_file)
                samples.append(
                    {
                        "file_path": str(npz_file),
                        "label": label,
                    }
                )
            return samples

        raise FileNotFoundError(
            f"Could not find data for split '{self.split}' in {self.data_dir}"
        )

    def _infer_label(self, file_path: Path) -> int:
        """
        Infer label from file path
        """
        parent_name = file_path.parent.name
        if parent_name.isdigit():
            return int(parent_name)

        filename = file_path.stem
        parts = filename.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)

        return 0

    def _load_class_names(self) -> Optional[List[str]]:
        """
        Load class names from metadata
        """
        metadata_file = self.data_dir.parent / "metadata" / "class_names.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset
        """
        sample = self.samples[idx]

        if sample["file_path"].endswith(".npz"):
            data = np.load(sample["file_path"], allow_pickle=True)
            poses_dict = {
                k: data[k]
                for k in ("body", "left_hand", "right_hand")
                if k in data
            }

            if "body" not in poses_dict:
                poses_dict["body"] = data["poses"]

            T = poses_dict["body"].shape[0]
            if "left_hand" not in poses_dict:
                poses_dict["left_hand"] = np.zeros((T, 21, 3), dtype=np.float32)
            if "right_hand" not in poses_dict:
                poses_dict["right_hand"] = np.zeros((T, 21, 3), dtype=np.float32)

            poses = poses_dict["body"]
            masks = data.get("body_mask", None)
        else:
            raise ValueError(f"Unknown file format: {sample['file_path']}")

        poses = self._temporal_sampling(poses)
        if masks is not None:
            masks = self._temporal_sampling(masks.astype(float)).astype(bool)

        if self.split == "train" and len(self.augmentation) > 0:
            poses = self._apply_augmentations(poses)

        graph_data = self.graph_constructor.construct_graph_from_poses(
            poses_dict if "left_hand" in poses_dict else {"body": poses},
            strategy=self.graph_strategy,
            masks=masks,
        )

        graph_data["y"] = torch.tensor(sample["label"], dtype=torch.long)

        return graph_data

    def _temporal_sampling(self, poses: np.ndarray) -> np.ndarray:
        """
        Sample temporal frames from pose sequence
        """
        T_orig = poses.shape[0]
        is_2d = len(poses.shape) == 2

        if is_2d:
            V = poses.shape[1]
        else:
            V, C = poses.shape[1], poses.shape[2]

        if T_orig <= self.num_frames:
            if is_2d:
                padding = np.zeros((self.num_frames - T_orig, V))
            else:
                padding = np.zeros((self.num_frames - T_orig, V, C))
            poses = np.concatenate([poses, padding], axis=0)
            return poses

        if self.random_temporal_crop and self.split == "train":
            max_start = max(0, T_orig - self.num_frames)
            start_idx = np.random.randint(0, max_start + 1)
        else:
            start_idx = 0

        indices = np.linspace(
            start_idx,
            start_idx + self.num_frames - 1,
            self.num_frames,
            dtype=int,
        )

        return poses[indices]

    def _apply_augmentations(self, poses: np.ndarray) -> np.ndarray:
        """
        Apply data augmentations to poses
        """
        return apply_augmentations(poses, self.augmentation)

    def get_class_names(self) -> Optional[List[str]]:
        return self.class_names

    def get_num_classes(self) -> int:
        if self.class_names:
            return len(self.class_names)
        labels = [s["label"] for s in self.samples]
        return max(labels) + 1 if labels else 0


class GraphCollateFn:
    """
    Custom collate function for graph batches
    """

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of graph samples
        """
        x = torch.stack([item["x"] for item in batch])  # (B, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)  # (B, C, T, V, 1)
        y = torch.stack([item["y"] for item in batch])  # (B,)

        edge_index = batch[0]["edge_index"]
        adj_matrix = batch[0]["adj_matrix"]

        edge_attr = torch.stack([item["edge_attr"] for item in batch])

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "adj_matrix": adj_matrix,
            "y": y,
        }


if __name__ == "__main__":
    dataset = GraphDataset(
        data_dir="dataset/processed/graphs",
        split="train",
        num_frames=64,
        random_temporal_crop=True,
        augmentation=[
            {"type": "temporal_jitter", "max_jitter": 3},
            {"type": "spatial_transform", "translation": 0.1},
        ],
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Node features shape: {sample['x'].shape}")
    print(f"Label: {sample['y']}")