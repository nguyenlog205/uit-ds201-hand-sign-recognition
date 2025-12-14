from torch.utils.data import Dataset
import numpy as np
import glob
import os
import torchvision
import torch
import json

class RGBDataset(Dataset):
    def __init__(self, config_path: dict):
        self.config_path = config_path
        self.raw_path = self.config_path['raw_path']
        self.metadata_dir = self.config_path['metadata']
        
        search_pattern = os.path.join(self.metadata_dir, "*.txt")
        self.list_of_sample_paths = glob.glob(search_pattern)
        self.list_of_sample_paths.sort() 

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
            video_tensor = video_tensor.permute(3, 0, 1, 2)
            # Normalize về [0, 1] và float32
            video_tensor = video_tensor.float() / 255.0

        return video_tensor, label

class SkeletonDataset(RGBDataset):
    def __init__(self, config_path: dict):
        super().__init__(config_path)
        # Skeleton thường được lưu ở folder riêng, vd: dataset/skeletons/
        # Config cần thêm key 'skeleton_path'
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