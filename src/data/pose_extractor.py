"""
Pose Extractor Module
Extract skeletal keypoints from video using MediaPipe Holistic
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
from tqdm import tqdm
import mediapipe as mp
from scipy.interpolate import interp1d


class MediaPipePoseExtractor:
    """
    MediaPipe Holistic-based pose extractor
    Extracts body, hand, and face keypoints from video with interpolation and masking
    """
    
    def __init__(
        self,
        use_holistic: bool = True,
        enable_interpolation: bool = True,
        body_model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe extractor
        
        Args:
            use_holistic: Use Holistic model (recommended)
            enable_interpolation: Enable interpolation for missing keypoints
            body_model_complexity: Body model complexity (0, 1, or 2)
            min_detection_confidence: Minimum detection confidence
            min_tracking_confidence: Minimum tracking confidence
        """
        self.use_holistic = use_holistic
        self.enable_interpolation = enable_interpolation
        
        if use_holistic:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                model_complexity=body_model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                static_image_mode=False,
            )
        
        self.num_body = 33
        self.num_hand = 21
        # Aliases for compatibility
        self.num_body_keypoints = 33
        self.num_hand_keypoints = 21
        self.num_face_keypoints = 70
    
    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract pose from a single frame
        
        Args:
            frame: BGR frame (H, W, 3)
            
        Returns:
            Dictionary containing:
                - 'body': (num_body_keypoints, 3) body keypoints (x, y, confidence)
                - 'left_hand': (21, 3) left hand keypoints (x, y, confidence)
                - 'right_hand': (21, 3) right hand keypoints (x, y, confidence)
                - 'face': (70, 3) face keypoints (x, y, confidence)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        results = {}
        
        if self.use_holistic:
            # Use Holistic model 
            holistic_results = self.holistic.process(rgb_frame)
            
            # Extract body keypoints
            if holistic_results.pose_landmarks:
                body_kp = np.zeros((self.num_body_keypoints, 3))
                for idx, landmark in enumerate(holistic_results.pose_landmarks.landmark):
                    if idx < self.num_body_keypoints:
                        body_kp[idx] = [landmark.x * w, landmark.y * h, landmark.visibility]
                results['body'] = body_kp
            else:
                results['body'] = np.zeros((self.num_body_keypoints, 3))
            
            # Extract hand keypoints (always enabled with holistic)
            self.hand = True
            if True:  # Always extract hands with holistic
                left_hand = np.zeros((self.num_hand_keypoints, 3))
                right_hand = np.zeros((self.num_hand_keypoints, 3))
                
                if holistic_results.left_hand_landmarks:
                    for idx, landmark in enumerate(holistic_results.left_hand_landmarks.landmark):
                        left_hand[idx] = [landmark.x * w, landmark.y * h, 1.0]  
                
                if holistic_results.right_hand_landmarks:
                    for idx, landmark in enumerate(holistic_results.right_hand_landmarks.landmark):
                        right_hand[idx] = [landmark.x * w, landmark.y * h, 1.0]
                
                results['left_hand'] = left_hand
                results['right_hand'] = right_hand
            else:
                results['left_hand'] = np.zeros((self.num_hand_keypoints, 3))
                results['right_hand'] = np.zeros((self.num_hand_keypoints, 3))
            
            # Extract face keypoints (always enabled with holistic)
            self.face = True
            if True:  # Always extract face with holistic
                if holistic_results.face_landmarks:
                    face_landmarks = holistic_results.face_landmarks
                    # Select 70 keypoints from 468 for compatibility
                    selected_indices = list(range(70))
                    face_kp = np.zeros((self.num_face_keypoints, 3))
                    for i, idx in enumerate(selected_indices):
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            face_kp[i] = [landmark.x * w, landmark.y * h, 1.0]
                    results['face'] = face_kp
                else:
                    results['face'] = np.zeros((self.num_face_keypoints, 3))
            else:
                results['face'] = np.zeros((self.num_face_keypoints, 3))
        else:
            raise NotImplementedError("Separate models not implemented. Use use_holistic=True")
        
        return results
    
    def _interpolate(self, data):
        if data.size == 0: return data
        T, V, C = data.shape
        data = data.copy()
        mask = (data[..., 0] != 0) | (data[..., 1] != 0)
        
        for v in range(V):
            valid_idx = np.where(mask[:, v])[0]
            if len(valid_idx) > 1 and len(valid_idx) < T:
                for c in range(3):
                    f = interp1d(valid_idx, data[valid_idx, v, c], 
                               kind='linear', bounds_error=False, fill_value="extrapolate")
                    missing_idx = np.where(~mask[:, v])[0]
                    data[missing_idx, v, c] = f(missing_idx)
        return data

    def _get_empty_result(self, w=0, h=0):
        return {
            'body': np.zeros((0, 33, 3)),
            'left_hand': np.zeros((0, 21, 3)),
            'right_hand': np.zeros((0, 21, 3)),
            'video_info': {'width': w, 'height': h}
        }
    
    def extract_from_video(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERROR: Could not open video at: {video_path}")
            return self._get_empty_result()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if max_frames: 
            total_frames = min(total_frames, max_frames)
        
        body_seq = []
        l_hand_seq = []
        r_hand_seq = []
        
        pbar = tqdm(total=total_frames, desc="Extracting Pose + Hands")
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if max_frames and processed_count >= max_frames: break
            
            # MediaPipe process
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            # 1. Extract Body
            body_kp = np.zeros((33, 3))
            if results.pose_landmarks:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    body_kp[i] = [lm.x * width, lm.y * height, lm.visibility]
            body_seq.append(body_kp)
            
            # 2. Extract Left Hand
            lh_kp = np.zeros((21, 3))
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    lh_kp[i] = [lm.x * width, lm.y * height, 1.0]
            l_hand_seq.append(lh_kp)

            # 3. Extract Right Hand
            rh_kp = np.zeros((21, 3))
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    rh_kp[i] = [lm.x * width, lm.y * height, 1.0]
            r_hand_seq.append(rh_kp)
            
            processed_count += 1
            pbar.update(1)
            
        cap.release()
        
        # Handle empty result
        if len(body_seq) == 0:
            return self._get_empty_result(width, height)

        # Convert to Numpy
        body_seq = np.array(body_seq)
        l_hand_seq = np.array(l_hand_seq)
        r_hand_seq = np.array(r_hand_seq)
        
        if self.enable_interpolation:
            body_seq = self._interpolate(body_seq)
            l_hand_seq = self._interpolate(l_hand_seq)
            r_hand_seq = self._interpolate(r_hand_seq)
        
        return {
            'body': body_seq,
            'left_hand': l_hand_seq,
            'right_hand': r_hand_seq,
            'video_info': {'width': width, 'height': height}
        }
    
    def _save_keypoints(
        self,
        keypoints: Dict[str, np.ndarray],
        output_path: str,
        save_format: str = "npy"
    ):
        """
        Save extracted keypoints to file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == "npy":
            np.savez_compressed(
                output_path,
                **keypoints
            )
        elif save_format == "json":
            json_data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in keypoints.items()}
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            raise ValueError(f"Unknown save format: {save_format}")
    
    @staticmethod
    def load_keypoints(file_path: str) -> Dict[str, np.ndarray]:
        """
        Load keypoints from file
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.npz':
            data = np.load(file_path, allow_pickle=True)
            return {k: data[k] for k in data.keys()}
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {k: np.array(v) if isinstance(v, list) else v 
                   for k, v in data.items()}
        else:
            raise ValueError(f"Unknown file format: {file_path.suffix}")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()