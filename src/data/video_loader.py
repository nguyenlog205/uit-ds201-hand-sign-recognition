"""
Video Loader Module
Load and preprocess video files
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm


class VideoLoader:
    """
    Video loader for reading video files
    """

    def __init__(
        self, 
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        fps: Optional[float] = None):
        """
        Initialize video loader
        
        Args:
            target_size: Target (width, height) for resizing (None = keep original)
            normalize: Whether to normalize frames to [0, 1]
            fps: Target FPS for resampling (None = keep original)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.fps = fps

    def load_video(
        self, 
        video_path: str,
        max_frames: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Load video frames
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load
            start_frame: Start frame index
            end_frame: End frame index (None = to end)
            
        Returns:
            Tuple of (frames, metadata):
                - frames: (T, H, W, 3) numpy array
                - metadata: Dictionary with video info
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine frame range
        if end_frame is None:
            end_frame = total_frames
        end_frame = min(end_frame, total_frames)

        if max_frames is not None:
            end_frame = min(end_frame, start_frame + max_frames)
        
        # Calculate frame indices to read
        if self.fps is not None and self.fps != fps:
            # Resample to target FPS
            frame_indices = self._calculate_resample_indices(
                start_frame, end_frame, fps, self.fps
            )
        else:
            frame_indices = list(range(start_frame, end_frame))

        # Read frames
        frames = []
        current_idx = 0

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        with tqdm(total=len(frame_indices), desc="Loading video") as pbar:
            for target_idx in frame_indices:
                if target_idx != current_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                    current_idx = target_idx

                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if target size is provided
                if self.target_size is not None:
                    frame = cv2.resize(frame, self.target_size)

                # Normalize if specified
                if self.normalize:
                    frame = frame / 255.0

                frames.append(frame)
                current_idx += 1
                pbar.update(1)
        
        cap.release()

        frames = np.array(frames)
        metadata = {
            'width': frames.shape[2] if len(frames.shape) > 2 else width,
            'height': frames.shape[1] if len(frames) > 0 else height,
            'fps': fps if self.fps is None else self.fps,
            'original_fps': fps,
            'total_frames': total_frames,
            'loaded_frames': len(frames),
        }

        return frames, metadata

    def _calculate_resample_indices(
        self,
        start: int,
        end: int,
        source_fps: float,
        target_fps: float,
    ) -> List[int]:
        """
        Calculate frame indices for FPS resampling
        
        Args:
            start: Start frame index
            end: End frame index
            source_fps: Source video FPS
            target_fps: Target FPS
            
        Returns:
            List of frame indices to read
        """
        ratio = source_fps / target_fps
        num_frames = int((end - start) * ratio)
        
        indices = []
        for i in range(num_frames):
            frame_idx = int(start + i / ratio)
            if frame_idx < end:
                indices.append(frame_idx)
        
        return indices

    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video information without loading frames
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        
        return info

if __name__ == "__main__":
    # Example usage
    loader = VideoLoader(
        target_size=(640, 480),
        normalize=False,
        fps=30,
    )
    
    # Use raw string or forward slashes for Windows paths
    video_path = r"C:\E old\DS201\Project\VSL-GCN-Research\sample_video\01_Co-Hien_1-100_1-2-3_0108___center_device02_signer01_center_ord1_61.mp4"
    # Or use Path for cross-platform compatibility:
    # from pathlib import Path
    # video_path = str(Path("C:/E old/DS201/Project/VSL-GCN-Research/sample_video/01_Co-Hien_1-100_1-2-3_0108___center_device02_signer01_center_ord1_61.mp4"))
    
    frames, metadata = loader.load_video(video_path, max_frames=100)
    
    print(f"Loaded {len(frames)} frames")
    print(f"Frame shape: {frames.shape}")
    print(f"Metadata: {metadata}")