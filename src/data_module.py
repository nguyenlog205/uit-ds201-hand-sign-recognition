from torch.utils.data import Dataset
import numpy as np
import glob
import os
import torchvision
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import json
from typing import Optional, List, Dict, Any, Tuple
import sys
from pathlib import Path
import cv2

# Add src to path for imports
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data.pose_extractor import MediaPipePoseExtractor
from data.gcn.graph_constructor import GraphConstructor
from data.augmentation import apply_rgb_augmentations

class RGBDataset(Dataset):
    def __init__(
        self, 
        config_path: dict,
        use_imagenet_norm: bool = False,
        target_size: Optional[Tuple[int, int]] = None,
        augmentations: Optional[List[Dict[str, Any]]] = None,
        is_training: bool = False,
        num_frames: Optional[int] = None
    ):
        """
        RGB Dataset for RGB-based models (ResNet50 + LSTM)
        """
        self.config_path = config_path
        self.raw_path = self.config_path['raw_path']
        self.metadata_dir = self.config_path['metadata']
        self.use_imagenet_norm = use_imagenet_norm
        self.target_size = target_size or (224, 224)  # Default for ResNet50
        self.augmentations = augmentations or []
        self.is_training = is_training
        self.num_frames = num_frames
        
        # Find all .txt files in metadata_dir and subfolders
        search_pattern = os.path.join(self.metadata_dir, "**", "*.txt")
        self.list_of_sample_paths = glob.glob(search_pattern, recursive=True)
        self.list_of_sample_paths.sort()
        
        # ImageNet normalization
        if use_imagenet_norm:
            self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
            self.imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.list_of_sample_paths)

    def _read_metadata(self, idx):
        """
        Hàm phụ trợ để đọc file metadata, dùng chung cho cả class con
        """
        txt_path = self.list_of_sample_paths[idx]
        with open(txt_path, 'r') as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                content = f.read()
                metadata = eval(content)
        return metadata
    
    
    def __getitem__(self, idx):
        metadata = self._read_metadata(idx)

        video_filename = metadata['origin']
        full_video_path = os.path.join(self.raw_path, video_filename)

        begining = float(metadata['begining'])
        ending = float(metadata['ending'])
        
        label = torch.tensor(metadata['label'], dtype=torch.long)

        video_tensor, audio_tensor, info = torchvision.io.read_video(
            filename=full_video_path,
            start_pts=begining,
            end_pts=ending,
            pts_unit='sec'
        )
        
        # Permute: (T, H, W, C) -> (C, T, H, W) chuẩn PyTorch
        if video_tensor.shape[0] > 0: # Check nếu không bị rỗng
            video_tensor = video_tensor.permute(3, 0, 1, 2).float()  # (C, T, H, W)
            C, T, H, W = video_tensor.shape
            
            # Resize frames to target size FIRST (before temporal sampling)
            if self.target_size:
                video_tensor = F.resize(video_tensor, self.target_size, antialias=True)
            
            # Temporal sampling/padding - ALWAYS normalize to same number of frames
            # Use num_frames if specified, otherwise use default (16 for ResNet-LSTM)
            target_frames = self.num_frames if self.num_frames is not None else 16
            
            if T > target_frames:
                # Sample frames uniformly
                indices = np.linspace(0, T - 1, target_frames, dtype=int)
                video_tensor = video_tensor[:, indices, :, :]
            elif T < target_frames:
                # Pad with last frame
                padding_size = target_frames - T
                # Repeat the entire video sequence
                num_repeats = (padding_size // T) + 1
                repeated_video = video_tensor.repeat(1, num_repeats, 1, 1)
                # Take only the needed frames
                padding = repeated_video[:, :padding_size, :, :]
                video_tensor = torch.cat([video_tensor, padding], dim=1)
            
            # Normalize về [0, 1]
            video_tensor = video_tensor / 255.0
            
            # Apply augmentations (if training)
            video_tensor = apply_rgb_augmentations(video_tensor, self.augmentations, self.is_training)
            
            # ImageNet normalization (for ResNet50)
            if self.use_imagenet_norm:
                video_tensor = (video_tensor - self.imagenet_mean) / self.imagenet_std
        else:
            # Return zero tensor if video is empty
            target_frames = self.num_frames if self.num_frames is not None else 16
            video_tensor = torch.zeros((3, target_frames, *self.target_size), dtype=torch.float32)

        return video_tensor, label

class SkeletonDataset(RGBDataset):
    def __init__(self, config_path: dict):
        super().__init__(config_path)
        self.skeleton_path = self.config_path.get('skeleton_path', './dataset/skeletons')
        self.fps = self.config_path.get('fps', 30)

    def __getitem__(self, idx):
        metadata = self._read_metadata(idx)
        video_name = metadata['origin']
        skeleton_name = os.path.splitext(video_name)[0] + ".npy"
        full_skeleton_path = os.path.join(self.skeleton_path, skeleton_name)

        if os.path.exists(full_skeleton_path):
            skeleton_data = np.load(full_skeleton_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy skeleton: {full_skeleton_path}")

        start_frame = int(metadata['begining'] * self.fps)
        end_frame = int(metadata['ending'] * self.fps)

        total_frames = skeleton_data.shape[0]
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        skeleton_segment = skeleton_data[start_frame:end_frame]
        
        skeleton_tensor = torch.from_numpy(skeleton_segment).float()
        
        label = torch.tensor(metadata['label'], dtype=torch.long)

        return skeleton_tensor, label


class SkeletonKeypointDataset(RGBDataset):
    """
    Dataset cho Bi-LSTM và PoseFormer - chỉ cần skeleton keypoints, không cần graph
    Tích hợp MediaPipe để extract keypoints từ video
    """
    def __init__(
        self,
        config_path: dict,
        pose_extractor: Optional[MediaPipePoseExtractor] = None,
        num_frames: Optional[int] = None,
        skeleton_layout: str = "mediapipe_27",
        normalize: bool = True,
    ):
        """
        Args:
            config_path: Configuration dictionary
            pose_extractor: MediaPipePoseExtractor instance (nếu None sẽ tạo mới)
            num_frames: Số frames cần sample (None = lấy tất cả)
            skeleton_layout: Layout của skeleton (mediapipe_27)
            normalize: Có normalize keypoints không
        """
        super().__init__(config_path)
        self.pose_extractor = pose_extractor or MediaPipePoseExtractor()
        self.num_frames = num_frames
        self.skeleton_layout = skeleton_layout
        self.normalize = normalize
        
        # Sử dụng GraphConstructor để convert MediaPipe format → 27 points
        self.graph_constructor = GraphConstructor(
            skeleton_layout=skeleton_layout,
            normalize=normalize,
        )

    def __getitem__(self, idx):
        metadata = self._read_metadata(idx)
        
        video_filename = metadata['origin']
        full_video_path = os.path.join(self.raw_path, video_filename)
        
        begining = float(metadata['begining'])
        ending = float(metadata['ending'])
        
        label = torch.tensor(metadata['label'], dtype=torch.long)
        
        # Extract keypoints từ video segment
        cap = cv2.VideoCapture(full_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Extract từ video với time range
        # Tạm thời extract toàn bộ video rồi cắt segment sau
        poses_dict = self.pose_extractor.extract_from_video(full_video_path)
        
        # Cắt theo thời gian
        start_frame = int(begining * fps)
        end_frame = int(ending * fps)
        
        body_seq = poses_dict['body'][start_frame:end_frame]
        l_hand_seq = poses_dict['left_hand'][start_frame:end_frame]
        r_hand_seq = poses_dict['right_hand'][start_frame:end_frame]
        
        # Convert sang 27-point format
        poses_dict_segment = {
            'body': body_seq,
            'left_hand': l_hand_seq,
            'right_hand': r_hand_seq,
        }
        
        # Sử dụng GraphConstructor để convert format (không tạo graph, chỉ convert format)
        final_poses = self.graph_constructor._convert_mediapipe_to_mediapipe27(poses_dict_segment)
        
        # Temporal sampling nếu cần
        if self.num_frames and final_poses.shape[0] > self.num_frames:
            indices = np.linspace(0, final_poses.shape[0] - 1, self.num_frames, dtype=int)
            final_poses = final_poses[indices]
        elif self.num_frames and final_poses.shape[0] < self.num_frames:
            # Padding
            padding = np.zeros((self.num_frames - final_poses.shape[0], final_poses.shape[1], final_poses.shape[2]))
            final_poses = np.concatenate([final_poses, padding], axis=0)
        
        skeleton_tensor = torch.from_numpy(final_poses).float()  # (T, 27, 3)
        
        return skeleton_tensor, label


class SkeletonGraphDataset(Dataset):
    """
    Dataset for ST-GCN and HA-GCN - requires graph structure
    Supports both:
    1. Loading from pre-processed .npz files (fast, recommended)
    2. On-the-fly extraction from videos (fallback, slower)
    """
    def __init__(
        self,
        config_path: Optional[dict] = None,
        data_root: Optional[str] = None,
        metadata_file: Optional[str] = None,
        samples: Optional[List[Dict]] = None,
        pose_extractor: Optional[MediaPipePoseExtractor] = None,
        graph_constructor: Optional[GraphConstructor] = None,
        num_frames: Optional[int] = None,
        skeleton_layout: str = "mediapipe_27",
        graph_strategy: str = "spatial",
        normalize: bool = True,
        is_training: bool = False,
    ):
        """
        Args:
            config_path: Configuration dict (for on-the-fly extraction)
            data_root: Root folder containing .npz files (for pre-processed data)
            metadata_file: Path to train.json/val.json (for pre-processed data)
            samples: List of samples dicts (alternative to metadata_file)
            pose_extractor: MediaPipePoseExtractor (only for on-the-fly)
            graph_constructor: GraphConstructor instance
            num_frames: Number of frames to sample
            skeleton_layout: Skeleton layout
            graph_strategy: Graph construction strategy
            normalize: Whether to normalize keypoints
            is_training: Whether in training mode
        """
        self.num_frames = num_frames or 64
        self.graph_strategy = graph_strategy
        self.is_training = is_training
        
        # Determine mode: pre-processed (.npz) or on-the-fly (video)
        if data_root is not None and (metadata_file is not None or samples is not None):
            # Pre-processed mode: load from .npz files
            self.mode = "preprocessed"
            self.data_root = Path(data_root)
            if samples is not None:
                self.samples = samples
            else:
                with open(metadata_file, 'r') as f:
                    self.samples = json.load(f)
        elif config_path is not None:
            # On-the-fly mode: extract from videos
            self.mode = "on_the_fly"
            self.config_path = config_path
            self.raw_path = config_path['raw_path']
            self.metadata_dir = config_path['metadata']
            search_pattern = os.path.join(self.metadata_dir, "**", "*.txt")
            self.list_of_sample_paths = glob.glob(search_pattern, recursive=True)
            self.list_of_sample_paths.sort()
            self.pose_extractor = pose_extractor or MediaPipePoseExtractor()
        else:
            raise ValueError("Must provide either (data_root + metadata_file/samples) or config_path")
        
        # Initialize Graph Constructor
        self.graph_constructor = graph_constructor or GraphConstructor(
            skeleton_layout=skeleton_layout,
            normalize=normalize,
        )

    def __len__(self):
        if self.mode == "preprocessed":
            return len(self.samples)
        else:
            return len(self.list_of_sample_paths)

    def _read_metadata(self, idx):
        """Read metadata for on-the-fly mode"""
        txt_path = self.list_of_sample_paths[idx]
        with open(txt_path, 'r') as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                content = f.read()
                metadata = eval(content)
        return metadata

    def _temporal_process(self, poses_dict):
        """
        Temporal sampling/padding: normalize to num_frames
        Returns processed poses_dict with consistent temporal length
        """
        T = poses_dict['body'].shape[0]
        
        if T > self.num_frames:
            # Case 1: Longer than target -> Sample
            if self.is_training:
                # Random crop during training
                start = np.random.randint(0, T - self.num_frames)
                indices = np.arange(start, start + self.num_frames)
            else:
                # Uniform sampling during validation/test
                indices = np.linspace(0, T - 1, self.num_frames, dtype=int)
        elif T < self.num_frames:
            # Case 2: Shorter than target -> Loop padding (repeat entire sequence)
            pad_len = self.num_frames - T
            # Repeat the entire sequence to fill remaining frames
            num_repeats = (pad_len // T) + 1
            indices = None  # Mark for special handling
        else:
            # Case 3: Exact length
            indices = np.arange(T)
        
        # Apply indices or padding
        new_dict = {}
        for key, val in poses_dict.items():
            if indices is not None:
                new_dict[key] = val[indices]
            else:
                # Loop padding: repeat entire sequence
                num_repeats = (pad_len // T) + 1
                repeated = np.tile(val, (num_repeats, 1, 1))
                new_dict[key] = np.concatenate([val, repeated[:pad_len]], axis=0)
        
        return new_dict

    def __getitem__(self, idx):
        if self.mode == "preprocessed":
            # Load from pre-processed .npz file (fast)
            item = self.samples[idx]
            file_path = self.data_root / item['file_path']
            label = torch.tensor(item['label'], dtype=torch.long)
            
            # Load .npz file (fast, pre-processed)
            data = np.load(file_path, allow_pickle=True)
            
            # Extract pose data
            # Note: 'x' in pre-processed data is (T, V, C) where V=27 (mediapipe_27 format)
            # This is the combined keypoints (body + hands merged into 27 points)
            if 'x' in data:
                # Pre-processed graph format: x is (T, 27, 3) keypoints
                x = data['x']  # (T, V, C) where V=27, C=3
                T = x.shape[0]
                # Reconstruct poses_dict from x (x is already in mediapipe_27 format)
                # For graph construction, we can use x directly or split if needed
                # Since x is already in the format needed by GraphConstructor, we use it as body
                poses_dict = {
                    'body': x,  # (T, 27, 3) - already in correct format
                    'left_hand': data.get('left_hand', np.zeros((T, 21, 3), dtype=np.float32)),
                    'right_hand': data.get('right_hand', np.zeros((T, 21, 3), dtype=np.float32))
                }
            elif 'body' in data:
                # Direct pose format (if saved separately)
                poses_dict = {
                    'body': data['body'],
                    'left_hand': data.get('left_hand', np.zeros((data['body'].shape[0], 21, 3), dtype=np.float32)),
                    'right_hand': data.get('right_hand', np.zeros((data['body'].shape[0], 21, 3), dtype=np.float32))
                }
            else:
                raise ValueError(f"Invalid .npz format in {file_path}. Expected 'x' or 'body' key.")
        else:
            # On-the-fly extraction from video (fallback, slower)
            metadata = self._read_metadata(idx)
            video_filename = metadata['origin']
            full_video_path = os.path.join(self.raw_path, video_filename)
            begining = float(metadata['begining'])
            ending = float(metadata['ending'])
            label = torch.tensor(metadata['label'], dtype=torch.long)
            
            # Extract keypoints from video
            poses_dict = self.pose_extractor.extract_from_video(full_video_path)
            
            # Cut by time
            cap = cv2.VideoCapture(full_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            start_frame = int(begining * fps)
            end_frame = int(ending * fps) if ending > 0 else len(poses_dict['body'])
            
            poses_dict = {
                'body': poses_dict['body'][start_frame:end_frame],
                'left_hand': poses_dict['left_hand'][start_frame:end_frame],
                'right_hand': poses_dict['right_hand'][start_frame:end_frame],
            }
        
        # Temporal processing (sampling/padding)
        poses_dict = self._temporal_process(poses_dict)
        
        # Construct graph
        graph_data = self.graph_constructor.construct_graph_from_poses(
            poses_dict,
            strategy=self.graph_strategy,
            masks=None,
        )
        
        graph_data['y'] = label
        
        return graph_data


class VideoMAEDataset(RGBDataset):
    """
    Dataset cho VideoMAE model - tích hợp VideoMAEProcessor
    Pipeline: Video -> frames -> processor -> model
    """
    def __init__(
        self,
        config_path: dict,
        processor,
        num_frames: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            config_path: Configuration dictionary
            processor: VideoMAEProcessor instance
            num_frames: Số frames cần sample
            target_size: Target size cho frames (processor sẽ tự resize nếu cần)
        """
        super().__init__(config_path)
        self.processor = processor
        self.num_frames = num_frames
        self.target_size = target_size

    def __getitem__(self, idx):
        metadata = self._read_metadata(idx)
        
        video_filename = metadata['origin']
        full_video_path = os.path.join(self.raw_path, video_filename)
        
        begining = float(metadata['begining'])
        ending = float(metadata['ending'])
        
        label = torch.tensor(metadata['label'], dtype=torch.long)
        
        # Load video segment
        video_tensor, audio_tensor, info = torchvision.io.read_video(
            filename=full_video_path,
            start_pts=begining,
            end_pts=ending,
            pts_unit='sec'
        )
        
        if video_tensor.shape[0] == 0:
            # Return empty tensor nếu không có frames
            return torch.zeros((3, 16, 224, 224)), label
        
        # Convert (T, H, W, C) -> list of PIL Images
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t].numpy()  # (H, W, C)
            frame = (frame * 255).astype(np.uint8)  # Convert to uint8
            frame_pil = Image.fromarray(frame)
            frames.append(frame_pil)
        
        # Temporal sampling nếu cần
        if self.num_frames and len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif self.num_frames and len(frames) < self.num_frames:
            # Repeat last frame
            last_frame = frames[-1] if frames else Image.new('RGB', (224, 224))
            frames.extend([last_frame] * (self.num_frames - len(frames)))
        
        # Process với VideoMAEProcessor - chuyển đổi format video -> frames -> processor -> model
        inputs = self.processor(frames, return_tensors="pt")
        
        # Extract pixel_values
        pixel_values = inputs['pixel_values']  # (1, C, T, H, W) hoặc (C, T, H, W)
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)  # Remove batch dimension
        
        return pixel_values, label