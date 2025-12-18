"""
Script to pre-process pose extraction offline
Extract pose from all videos and save to .npz files
Automatically split train/val/test and create train.json, val.json, test.json
"""

import sys
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import json
import glob
import random
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pose_extractor import MediaPipePoseExtractor
from src.data.gcn.graph_constructor import GraphConstructor

def preprocess_all_videos(
    raw_path: str = "DATA/Segmented",
    output_path: str = "DATA/Processed_Pose",
    num_frames: int = 64,
    skeleton_layout: str = "mediapipe_27",
    normalize: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Pre-process all videos: extract pose and save to .npz files
    Automatically split train/val/test and create train.json, val.json, test.json
    
    Args:
        raw_path: Folder containing segmented videos (Segmented/)
        output_path: Folder to save extracted .npz files
        num_frames: Number of frames to sample
        skeleton_layout: Skeleton layout
        normalize: Whether to normalize
        train_ratio: Train split ratio (default: 0.7 = 70%)
        val_ratio: Validation split ratio (default: 0.15 = 15%)
        test_ratio: Test split ratio (default: 0.15 = 15%)
        seed: Random seed for train/val/test split
    """
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Initializing MediaPipe extractor...")
    pose_extractor = MediaPipePoseExtractor()
    graph_constructor = GraphConstructor(
        skeleton_layout=skeleton_layout,
        normalize=normalize
    )
    
    # Find all metadata files (.txt)
    metadata_pattern = os.path.join(raw_path, "**", "*.txt")
    metadata_files = glob.glob(metadata_pattern, recursive=True)
    metadata_files.sort()
    
    print(f"\nFound {len(metadata_files)} metadata files")
    print(f"Input: {raw_path}")
    print(f"Output: {output_path}")
    
    # Group by class to ensure even distribution
    metadata_by_class = defaultdict(list)
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r') as f:
                try:
                    metadata = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    content = f.read()
                    metadata = eval(content)
            
            label = metadata['label']
            metadata_by_class[label].append({
                'metadata_file': metadata_file,
                'metadata': metadata
            })
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
            continue
    
    print(f"\nClass distribution:")
    for label, items in sorted(metadata_by_class.items()):
        print(f"   Class {label}: {len(items)} samples")
    
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: Ratios sum to {total_ratio:.3f}, not 1.0. Normalizing...")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # Split train/val/test by class to ensure each class has enough samples in all 3 sets
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    random.seed(seed)
    np.random.seed(seed)
    
    for label, items in metadata_by_class.items():
        random.shuffle(items)
        total = len(items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_metadata.extend(items[:train_end])
        val_metadata.extend(items[train_end:val_end])
        test_metadata.extend(items[val_end:])
    
    random.shuffle(train_metadata)
    random.shuffle(val_metadata)
    random.shuffle(test_metadata)
    
    total_samples = len(train_metadata) + len(val_metadata) + len(test_metadata)
    print(f"\nSplit distribution: Train={len(train_metadata)} ({len(train_metadata)/total_samples*100:.1f}%), "
          f"Val={len(val_metadata)} ({len(val_metadata)/total_samples*100:.1f}%), "
          f"Test={len(test_metadata)} ({len(test_metadata)/total_samples*100:.1f}%)")
    
    # Process each file
    def process_metadata_list(metadata_list, split_name):
        """Process a list of metadata and save to corresponding folder"""
        split_output = output_path / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        skip_count = 0
        error_count = 0
        processed_files = []
        
        for item in tqdm(metadata_list, desc=f"Processing {split_name}"):
            try:
                metadata_file = item['metadata_file']
                metadata = item['metadata']
                
                video_filename = metadata['origin']
                full_video_path = raw_path / video_filename
                
                if not full_video_path.exists():
                    print(f"Video not found: {full_video_path}")
                    error_count += 1
                    continue
                
                begining = float(metadata['begining'])
                ending = float(metadata['ending'])
                label = metadata['label']
                
                # Output file name: based on original video name + hash to avoid duplicates
                video_stem = Path(video_filename).stem
                metadata_stem = Path(metadata_file).stem
                output_file = split_output / f"{metadata_stem}_{video_stem}.npz"
                
                # Skip if file already exists
                if output_file.exists():
                    skip_count += 1
                    processed_files.append({
                        'file_path': str(output_file.relative_to(output_path)),
                        'label': label,
                        'video_path': str(full_video_path),
                        'metadata_file': str(metadata_file)
                    })
                    continue
                
                # Extract pose from video
                poses_dict = pose_extractor.extract_from_video(str(full_video_path))
                
                # Cut by time
                import cv2
                cap = cv2.VideoCapture(str(full_video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                start_frame = int(begining * fps)
                end_frame = int(ending * fps) if ending > 0 else len(poses_dict['body'])
                
                poses_dict_segment = {
                    'body': poses_dict['body'][start_frame:end_frame],
                    'left_hand': poses_dict['left_hand'][start_frame:end_frame],
                    'right_hand': poses_dict['right_hand'][start_frame:end_frame],
                }
                
                # Temporal sampling if needed
                if num_frames:
                    T = poses_dict_segment['body'].shape[0]
                    if T > num_frames:
                        indices = np.linspace(0, T - 1, num_frames, dtype=int)
                        for key in poses_dict_segment:
                            poses_dict_segment[key] = poses_dict_segment[key][indices]
                    elif T < num_frames:
                        # Padding
                        padding_size = num_frames - T
                        for key in poses_dict_segment:
                            padding = np.zeros((padding_size, poses_dict_segment[key].shape[1], poses_dict_segment[key].shape[2]))
                            poses_dict_segment[key] = np.concatenate([poses_dict_segment[key], padding], axis=0)
                
                # Construct graph
                graph_data = graph_constructor.construct_graph_from_poses(
                    poses_dict_segment,
                    strategy="spatial",
                    masks=None,
                )
                
                # Save graph data + metadata
                save_dict = {
                    'x': graph_data['x'],  # (T, V, C) - node features
                    'edge_index': graph_data.get('edge_index'),  # Graph edges (if available)
                    'label': label,
                    'video_path': str(full_video_path),
                    'metadata': metadata
                }
                
                np.savez_compressed(output_file, **save_dict)
                
                processed_files.append({
                    'file_path': str(output_file.relative_to(output_path)),
                    'label': label,
                    'video_path': str(full_video_path),
                    'metadata_file': str(metadata_file)
                })
                
                success_count += 1
                
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
                error_count += 1
                continue
        
        return processed_files, success_count, skip_count, error_count
    
    print("\n" + "="*60)
    print("Processing TRAIN set...")
    train_files, train_success, train_skip, train_error = process_metadata_list(train_metadata, "train")
    
    print("\n" + "="*60)
    print("Processing VAL set...")
    val_files, val_success, val_skip, val_error = process_metadata_list(val_metadata, "val")
    
    print("\n" + "="*60)
    print("Processing TEST set...")
    test_files, test_success, test_skip, test_error = process_metadata_list(test_metadata, "test")
    
    # Save train.json, val.json and test.json
    train_json_path = output_path / "train.json"
    val_json_path = output_path / "val.json"
    test_json_path = output_path / "test.json"
    
    with open(train_json_path, 'w') as f:
        json.dump(train_files, f, indent=2)
    
    with open(val_json_path, 'w') as f:
        json.dump(val_files, f, indent=2)
    
    with open(test_json_path, 'w') as f:
        json.dump(test_files, f, indent=2)
    
    print("\n" + "="*60)
    print("Pre-processing completed!")
    print(f"\nTRAIN set: Success={train_success}, Skipped={train_skip}, Errors={train_error}, Total={len(train_files)}")
    print(f"VAL set: Success={val_success}, Skipped={val_skip}, Errors={val_error}, Total={len(val_files)}")
    print(f"TEST set: Success={test_success}, Skipped={test_skip}, Errors={test_error}, Total={len(test_files)}")
    print(f"\nOutput folder: {output_path}")
    print(f"Train JSON: {train_json_path}")
    print(f"Val JSON: {val_json_path}")
    print(f"Test JSON: {test_json_path}")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-process pose extraction offline")
    parser.add_argument("--raw_path", type=str, default="DATA/Segmented",
                        help="Path to raw video folder")
    parser.add_argument("--output_path", type=str, default="DATA/Processed_Pose",
                        help="Path to save processed .npz files")
    parser.add_argument("--num_frames", type=int, default=64,
                        help="Number of frames to sample")
    parser.add_argument("--skeleton_layout", type=str, default="mediapipe_27",
                        help="Skeleton layout")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize keypoints")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train split ratio (default: 0.7 = 70%%)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15 = 15%%)")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                        help="Test split ratio (default: 0.15 = 15%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")
    
    args = parser.parse_args()
    
    preprocess_all_videos(
        raw_path=args.raw_path,
        output_path=args.output_path,
        num_frames=args.num_frames,
        skeleton_layout=args.skeleton_layout,
        normalize=args.normalize,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )