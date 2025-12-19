"""
Main training script for Vietnamese sign language recognition models.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from pathlib import Path
import random
import numpy as np
import sys
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.configs import Config
from src.model.model_factory import create_model
from src.data_module import (
    RGBDataset,
    SkeletonDataset,
    SkeletonKeypointDataset,
    SkeletonGraphDataset,
    VideoMAEDataset
)
from src.data_module_preprocessed import (
    create_preprocessed_datasets, 
    PreprocessedGraphDataset,
    PreprocessedKeypointDataset
)
from src.data.pose_extractor import MediaPipePoseExtractor
from src.data.gcn.graph_constructor import GraphConstructor
from src.training.trainer import Trainer
from src.evaluation.eval import Evaluator
from src.utils.logger import get_logger


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_num_classes_from_dataset(dataset) -> int:
    """Get number of classes from dataset"""
    # Handle Subset (from random_split)
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    
    # Try to get from dataset method
    if hasattr(dataset, 'get_num_classes'):
        return dataset.get_num_classes()
    
    # Try to infer from labels
    labels = []
    sample_size = min(100, len(dataset))  # Sample first 100
    for i in range(sample_size):
        try:
            if isinstance(dataset, Subset):
                _, label = dataset[i]
            else:
                _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                labels.append(label.item())
            else:
                labels.append(label)
        except:
            continue
    
    if labels:
        num_classes = max(labels) + 1
        return num_classes
    
    # Default fallback
    return 10


def create_dataset(config: Config, split: str = "train"):
    """Create dataset based on configuration"""
    data_config = config.data
    
    # Prepare config dict for dataset
    # Use validation metadata if available and split is "val"
    metadata_path = data_config.metadata_val if (split == "val" and data_config.metadata_val) else data_config.metadata
    
    dataset_config = {
        'raw_path': data_config.raw_path,
        'metadata': metadata_path,
    }
    
    if data_config.skeleton_path:
        dataset_config['skeleton_path'] = data_config.skeleton_path
    
    # Determine dataset type
    dataset_type = data_config.dataset_type
    
    if dataset_type == "rgb":
        # RGB dataset for ResNet-LSTM
        return RGBDataset(
            config_path=dataset_config,
            use_imagenet_norm=data_config.use_imagenet_norm,
            target_size=tuple(data_config.target_size) if data_config.target_size else None,
            augmentations=data_config.augmentations if split == "train" else None,
            is_training=(split == "train")
        )
    
    elif dataset_type == "skeleton":
        # Pre-extracted skeleton data
        return SkeletonDataset(config_path=dataset_config)
    
    elif dataset_type == "skeleton_keypoint":
        # Skeleton keypoints for Bi-LSTM, PoseFormer
        # Check if pre-processed data exists
        processed_data_root = getattr(data_config, 'processed_data_root', None) or "DATA/Processed_Pose"
        processed_data_path = Path(processed_data_root)
        train_json = processed_data_path / "train.json"
        val_json = processed_data_path / "val.json"
        test_json = processed_data_path / "test.json"
        
        if train_json.exists() and val_json.exists():
            # Use pre-processed data
            print(f"Using pre-processed data for skeleton_keypoint: {processed_data_root}")
            
            # Load pre-processed graph dataset and convert to keypoint format
            if split == "train":
                json_file = str(train_json)
            elif split == "val":
                json_file = str(val_json)
            elif split == "test":
                json_file = str(test_json) if test_json.exists() else str(val_json)
            else:
                json_file = str(train_json)
            
            train_duplicate_factor = getattr(data_config, 'train_duplicate_factor', 10) if split == "train" else 1
            
            # Create preprocessed dataset wrapper that returns keypoint format
            from src.data_module_preprocessed import PreprocessedKeypointDataset
            return PreprocessedKeypointDataset(
                json_file=json_file,
                data_root=str(processed_data_path),
                num_frames=data_config.num_frames or 64,
                is_training=(split == "train"),
                duplicate_factor=train_duplicate_factor
            )
        else:
            print(f"Pre-processed data not found at {processed_data_root}")
            print("Falling back to on-the-fly pose extraction (slower)")
            print("To use pre-processed data, run: python -m src.data.preprocess_data --raw_path DATA/Segmented --output_path DATA/Processed_Pose")
            
            pose_extractor = MediaPipePoseExtractor()
            return SkeletonKeypointDataset(
                config_path=dataset_config,
                pose_extractor=pose_extractor,
                num_frames=data_config.num_frames,
                skeleton_layout=data_config.skeleton_layout or "mediapipe_27",
                normalize=data_config.normalize
            )
    
    elif dataset_type == "skeleton_graph":
        processed_data_root = getattr(data_config, 'processed_data_root', None) or "DATA/Processed_Pose"
        processed_data_path = Path(processed_data_root)
        train_json = processed_data_path / "train.json"
        val_json = processed_data_path / "val.json"
        test_json = processed_data_path / "test.json"
        
        if train_json.exists() and val_json.exists():
            print(f"Using pre-processed data from: {processed_data_root}")
            train_duplicate_factor = getattr(data_config, 'train_duplicate_factor', 10) if split == "train" else 1
            
            if split == "train":
                json_file = str(train_json)
            elif split == "val":
                json_file = str(val_json)
            elif split == "test":
                json_file = str(test_json) if test_json.exists() else str(val_json)
            else:
                json_file = str(train_json)
            
            return PreprocessedGraphDataset(
                json_file=json_file,
                data_root=str(processed_data_path),
                num_frames=data_config.num_frames or 64,
                is_training=(split == "train"),
                augmentation_configs=None,
                duplicate_factor=train_duplicate_factor
            )
        else:
            print(f"Pre-processed data not found at {processed_data_root}")
            print("Falling back to on-the-fly pose extraction (slower)")
            print("To use pre-processed data, run: python -m src.data.preprocess_data --raw_path DATA/Segmented --output_path DATA/Processed_Pose")
            
            pose_extractor = MediaPipePoseExtractor()
            graph_constructor = GraphConstructor(
                skeleton_layout=data_config.skeleton_layout or "mediapipe_27",
                normalize=data_config.normalize
            )
            return SkeletonGraphDataset(
                config_path=dataset_config,
                pose_extractor=pose_extractor,
                graph_constructor=graph_constructor,
                num_frames=data_config.num_frames,
                skeleton_layout=data_config.skeleton_layout or "mediapipe_27",
                graph_strategy=data_config.graph_strategy or "spatial",
                normalize=data_config.normalize
            )
    
    elif dataset_type == "videomae":
        # VideoMAE dataset
        try:
            from transformers import VideoMAEProcessor
            processor_name = data_config.processor_name or "MCG-NJU/videomae-base"
            processor = VideoMAEProcessor.from_pretrained(processor_name)
        except ImportError:
            raise ImportError("transformers library is required for VideoMAE. Install with: pip install transformers")
        
        return VideoMAEDataset(
            config_path=dataset_config,
            processor=processor,
            num_frames=data_config.num_frames,
            target_size=tuple(data_config.target_size) if data_config.target_size else None
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def create_data_loaders(config: Config):
    """Create train and validation data loaders"""
    data_config = config.data
    
    processed_data_root = getattr(data_config, 'processed_data_root', None) or "DATA/Processed_Pose"
    processed_data_path = Path(processed_data_root)
    train_json = processed_data_path / "train.json"
    val_json = processed_data_path / "val.json"
    test_json = processed_data_path / "test.json"
    
    is_preprocessed = (
        data_config.dataset_type == "skeleton_graph" and 
        train_json.exists() and 
        val_json.exists()
    )
    
    has_test_set = is_preprocessed and test_json.exists()
    test_dataset = None
    
    if is_preprocessed:
        print(f"Using pre-processed data (already split): {processed_data_root}")
        train_dataset = create_dataset(config, split="train")
        val_dataset = create_dataset(config, split="val")
        if has_test_set:
            test_dataset = create_dataset(config, split="test")
    elif data_config.metadata_val:
        train_dataset = create_dataset(config, split="train")
        val_dataset = create_dataset(config, split="val")
        if hasattr(data_config, 'metadata_test') and data_config.metadata_test:
            test_dataset = create_dataset(config, split="test")
    else:
        full_dataset = create_dataset(config, split="train")
        num_classes = get_num_classes_from_dataset(full_dataset)
        
        total_size = len(full_dataset)
        train_ratio = getattr(data_config, 'train_ratio', 0.7)
        val_ratio = getattr(data_config, 'val_ratio', 0.15)
        test_ratio = getattr(data_config, 'test_ratio', 0.15)
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        if data_config.random_split:
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(config.seed or 42)
            )
        else:
            indices = list(range(total_size))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
            test_dataset = Subset(full_dataset, test_indices)
    
    num_classes = get_num_classes_from_dataset(train_dataset)
    if config.model.num_class is None:
        config.model.num_class = num_classes
    
    train_size = len(train_dataset)
    drop_last_train = train_size > config.training.batch_size
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False,
        drop_last=drop_last_train
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.device == "cuda" else False,
        drop_last=False
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if config.device == "cuda" else False,
            drop_last=False
        )
    
    return train_loader, val_loader, num_classes, test_loader


def main():
    parser = argparse.ArgumentParser(
        description="Train hand sign recognition models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    # Training options
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate model (no training)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation"
    )
    
    # Override config options
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate from config")  
    parser.add_argument("--num-epochs", type=int, default=None, help="Override number of epochs from config")
    parser.add_argument("--device", type=str, default=None, help="Override device from config (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Set device
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config.device = "cpu"
    
    # Set seed
    if config.seed:
        set_seed(config.seed)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    experiment_name = config.experiment_name or config.model.type
    log_file = output_dir / f"{experiment_name}_training.log"
    logger = get_logger(name="hand_sign_recognition", log_file=str(log_file))
    
    logger.info("=" * 80)
    logger.info("Hand Sign Recognition Training")
    logger.info("=" * 80)
    logger.info(f"Model type: {config.model.type}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration file: {args.config}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    result = create_data_loaders(config)
    if len(result) == 4:
        train_loader, val_loader, num_classes, test_loader = result
    else:
        train_loader, val_loader, num_classes = result
        test_loader = None
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model_config_dict = config.model.model_dump()
    model = create_model(
        {'model': model_config_dict},
        num_classes=num_classes
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Evaluation only mode
    if args.eval_only:
        logger.info("Evaluation mode")
        checkpoint_path = args.checkpoint or args.resume
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("No checkpoint provided, using untrained model")
        
        evaluator = Evaluator(model, device=device, logger=logger)
        metrics = evaluator.evaluate(val_loader, use_amp=config.training.use_amp)
        logger.info("Evaluation completed")
        return
    
    # Create trainer
    training_config_dict = config.training.model_dump()
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config_dict,
        device=device,
        logger=logger
    )
    
    # Resume from checkpoint
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume, load_optimizer=True)
    
    # Save configuration
    config.save_yaml(output_dir / "config.yaml")
    
    # Train
    trainer.train(output_dir=str(output_dir))
    
    # Load best model before evaluation 
    best_checkpoint_path = output_dir / "checkpoint_best.pth"
    if best_checkpoint_path.exists():
        logger.info(f"Loading best model from {best_checkpoint_path} for final evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint.get('epoch', 0)
        best_val_f1 = checkpoint.get('best_val_f1', 0)
        logger.info(f"Best model loaded (F1: {best_val_f1:.4f} at epoch {best_epoch + 1})")
    else:
        logger.warning("Best checkpoint not found, using current model state")
    
    # Final evaluation trên validation set
    logger.info("Running final evaluation on validation set...")
    evaluator = Evaluator(model, device=device, logger=logger)
    val_metrics = evaluator.evaluate(val_loader, use_amp=config.training.use_amp)
    
    logger.info(f"Validation - Accuracy: {val_metrics.get('accuracy', 0):.4f}, F1-macro: {val_metrics.get('f1_macro', 0):.4f}")
    
    if test_loader:
        logger.info("Running final evaluation on test set...")
        test_metrics = evaluator.evaluate(test_loader, use_amp=config.training.use_amp)
        logger.info(f"Test - Accuracy: {test_metrics.get('accuracy', 0):.4f}, F1-macro: {test_metrics.get('f1_macro', 0):.4f}")
    
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()