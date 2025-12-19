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
    
    # Find all video files (.mp4) in Segmented/
    video_pattern = os.path.join(raw_path, "**", "*.mp4")
    video_files = glob.glob(video_pattern, recursive=True)
    video_files.sort()
    
    print(f"\nFound {len(video_files)} video files")
    print(f"Input: {raw_path}")
    print(f"Output: {output_path}")
    
    # Create mapping from folder name to label index
    folder_names = set()
    for video_file in video_files:
        folder_name = Path(video_file).parent.name
        folder_names.add(folder_name)
    
    folder_names = sorted(folder_names)
    folder_to_label = {folder: idx for idx, folder in enumerate(folder_names)}
    label_to_folder = {idx: folder for folder, idx in folder_to_label.items()}
    
    print(f"\nFound {len(folder_names)} classes (folders): {folder_names}")
    print(f"Label mapping: {folder_to_label}")
    
    # Group by class to ensure even distribution
    metadata_by_class = defaultdict(list)
    
    # Process all videos: use existing metadata if available, otherwise create new
    for video_file in video_files:
        video_path = Path(video_file)
        folder_name = video_path.parent.name
        label = folder_to_label[folder_name]
        
        # Check if metadata file exists
        metadata_file = video_path.with_suffix('.txt')
        
        if metadata_file.exists():
            # Use existing metadata
            try:
                with open(metadata_file, 'r') as f:
                    try:
                        metadata = json.load(f)
                    except json.JSONDecodeError:
                        f.seek(0)
                        content = f.read()
                        metadata = eval(content)
                
                # Update label to match folder name
                metadata['label'] = label
                metadata['origin'] = str(video_path.relative_to(raw_path))
                
                metadata_by_class[label].append({
                    'metadata_file': str(metadata_file),
                    'metadata': metadata,
                    'video_file': str(video_file)
                })
            except Exception as e:
                print(f"Error reading {metadata_file}: {e}, creating new metadata")
                # Fall through to create new metadata
                metadata = None
        else:
            metadata = None
        
        # Create new metadata if not found
        if metadata is None:
            # Get video duration
            import cv2
            cap = cv2.VideoCapture(str(video_file))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
            
            # Create metadata
            metadata = {
                'origin': str(video_path.relative_to(raw_path)),
                'begining': 0.0,
                'ending': duration,
                'label': label
            }
            
            metadata_by_class[label].append({
                'metadata_file': None,  # No metadata file
                'metadata': metadata,
                'video_file': str(video_file)
            })
    
    print(f"\nClass distribution:")
    for label, items in sorted(metadata_by_class.items()):
        print(f"   Class {label}: {len(items)} samples")
    
    # Check for classes with insufficient samples
    classes_with_insufficient = [label for label, items in metadata_by_class.items() if len(items) < 3]
    if classes_with_insufficient:
        print(f"\n[WARNING] Classes with < 3 samples (cannot be stratified): {classes_with_insufficient}")
        print(f"   These classes will only appear in the training set.")
    
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
    
    # Track classes that don't have enough samples for stratified split
    insufficient_samples = []
    
    for label, items in metadata_by_class.items():
        random.shuffle(items)
        total = len(items)
        
        # Stratified split: ensure each class has at least 1 sample in each set
        if total >= 3:
            # Calculate split points based on ratios
            train_count = max(1, int(total * train_ratio))
            val_count = max(1, int(total * val_ratio))
            test_count = total - train_count - val_count
            
            # Adjust to ensure at least 1 in each set
            if test_count < 1:
                # Need to take from train or val
                if train_count > 1:
                    train_count -= 1
                    test_count += 1
                elif val_count > 1:
                    val_count -= 1
                    test_count += 1
            
            # Final split
            train_metadata.extend(items[:train_count])
            val_metadata.extend(items[train_count:train_count + val_count])
            test_metadata.extend(items[train_count + val_count:])
        else:
            # Not enough samples for stratified split (need at least 3)
            insufficient_samples.append((label, total))
            # Put all in train
            train_metadata.extend(items)
    
    random.shuffle(train_metadata)
    random.shuffle(val_metadata)
    random.shuffle(test_metadata)
    
    total_samples = len(train_metadata) + len(val_metadata) + len(test_metadata)
    print(f"\nSplit distribution: Train={len(train_metadata)} ({len(train_metadata)/total_samples*100:.1f}%), "
          f"Val={len(val_metadata)} ({len(val_metadata)/total_samples*100:.1f}%), "
          f"Test={len(test_metadata)} ({len(test_metadata)/total_samples*100:.1f}%)")
    
    # Show per-class distribution
    if insufficient_samples:
        print(f"\n[INFO] Classes with insufficient samples (all in train):")
        for label, count in insufficient_samples:
            print(f"   Class {label}: {count} sample(s)")
    
    # Verify stratified split
    print(f"\n[INFO] Verifying stratified split...")
    train_labels = set()
    val_labels = set()
    test_labels = set()
    
    for item in train_metadata:
        train_labels.add(item['metadata']['label'])
    for item in val_metadata:
        val_labels.add(item['metadata']['label'])
    for item in test_metadata:
        test_labels.add(item['metadata']['label'])
    
    all_labels = set(metadata_by_class.keys())
    stratified_labels = train_labels & val_labels & test_labels
    
    print(f"   Total classes: {len(all_labels)}")
    print(f"   Classes in train: {len(train_labels)}")
    print(f"   Classes in val: {len(val_labels)}")
    print(f"   Classes in test: {len(test_labels)}")
    print(f"   Classes in ALL sets (stratified): {len(stratified_labels)}")
    
    if len(stratified_labels) < len(all_labels):
        missing = all_labels - stratified_labels
        print(f"   [WARNING] Classes missing from some sets: {sorted(missing)}")
    
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
                metadata_file = item.get('metadata_file')
                metadata = item['metadata']
                video_file = item.get('video_file')
                
                video_filename = metadata['origin']
                full_video_path = raw_path / video_filename
                
                # If video_file is provided, use it directly
                if video_file and Path(video_file).exists():
                    full_video_path = Path(video_file)
                elif not full_video_path.exists():
                    print(f"Video not found: {full_video_path}")
                    error_count += 1
                    continue
                
                begining = float(metadata['begining'])
                ending = float(metadata['ending'])
                label = metadata['label']
                
                # Output file name: based on original video name
                video_stem = Path(video_filename).stem
                if metadata_file:
                    metadata_stem = Path(metadata_file).stem
                    output_file = split_output / f"{metadata_stem}_{video_stem}.npz"
                else:
                    # No metadata file, use video name directly
                    output_file = split_output / f"{video_stem}.npz"
                
                # Skip if file already exists
                if output_file.exists():
                    skip_count += 1
                    processed_files.append({
                        'file_path': str(output_file.relative_to(output_path)),
                        'label': label,
                        'video_path': str(full_video_path),
                        'metadata_file': str(metadata_file) if metadata_file else None
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
                    'metadata_file': str(metadata_file) if metadata_file else None
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
    
    # Save label mapping
    label_mapping_path = output_path / "label_mapping.json"
    label_mapping = {
        'label_to_folder': label_to_folder,
        'folder_to_label': folder_to_label,
        'num_classes': len(folder_names)
    }
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Label mapping saved: {label_mapping_path}")
    
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