# 🧠 `src/` — Core Source Code

This directory contains all the main components of the **FineSign** system — from dataset loaders and video transforms to model architectures, trainers, and inference utilities.  
It’s designed for **clarity**, **reusability**, and **scalability**, following modular design principles inspired by large-scale ML systems.


## 📁 Directory Overview
```php
src/
├─ datasets/   # Handles data loading and preprocessing
├─ transforms/ # Augmentation and temporal/spatial transforms
├─ models/     # Backbone architectures, heads, and graph modules
├─ trainers/   # Training logic and Hugging Face wrapper
├─ utils/      # General-purpose helpers and logging tools
├─ inference/  # Inference and evaluation pipeline
└─ export/     # Model export to ONNX / TorchScript
```

## 🧩 Modules

### 🗂️ `datasets/`
Handles dataset creation and frame/pose sampling for training and evaluation.

- **`base_dataset.py`** – Abstract dataset class defining a unified API (`__getitem__`, `__len__`).
- **`video_dataset.py`** – Loads RGB video frames, applies temporal sampling and augmentation.
- **`pose_dataset.py`** – Loads pose/keypoint data (e.g., from MediaPipe or OpenPose) for multimodal fusion.

All datasets return a standardized dictionary:
```python
{
  "video": Tensor,       # [T, C, H, W]
  "pose": Optional[Tensor],
  "label": int,
}
```

### 🎞️ `transforms/`
Implements preprocessing and data augmentation logic for both video and pose data.
- `video_transforms.py` – Temporal sampling, frame normalization, resize, random crop, horizontal flip.
- `pose_transforms.py` – Normalization and geometric jittering for keypoint coordinates.

Supports chaining via `torchvision.transforms.Compose`.

### 🧠 `models/`
Contains the main neural architectures.
- `backbones/`
    - `videomae_backbone.py` – Transformer-based masked autoencoder for video representation.
    `timesformer_backbone.py` – TimeSformer model (space-time attention).
- `heads/`
    - l`inear_head.py` – Lightweight classification head.
    - `fusion_head.py` – Fuses features from multiple modalities (video + pose).
- `graph/`
    - `stgcn.py` – Spatial-Temporal Graph Convolutional Network for structured pose learning.
### ⚙️ `trainers/`
Training and fine-tuning logic.
- `trainer.py` – Main training loop supporting checkpointing, resuming, mixed precision, and evaluation.
- `hf_trainer_wrapper.py` – Optional integration with the Hugging Face Trainer API for faster experimentation.

Logging integrates with W&B or MLflow through `utils/logger.py`.

### 🧰 utils/
Utility scripts and helpers.
- `metrics.py` – Accuracy, F1-score, confusion matrix computation.
- `logger.py` – Abstract logger supporting W&B or MLflow.
- `seed.py` – Ensures reproducibility by fixing seeds across NumPy, PyTorch, etc.
- `checkpoints.py` – Save/load checkpoint management.

### 🔍 `inference/`
Handles single-video prediction and batch evaluation.
- `infer.py`
    - Loads a trained model and runs forward inference.
    - Accepts video file or frame folder.
    - Outputs top-k predictions and confidence scores.

### 📦 `export/`
Utilities for exporting trained models for deployment.
- `export_onnx.py` – Converts PyTorch model → ONNX or TorchScript for production environments.
    - Supports dynamic axes and half-precision export.
    - Example:
    ```
    python -m src.export.export_onnx \
        --checkpoint checkpoints/best_model.pt \
        --output onnx/finesign.onnx 
    ```


## APPENDIX - Design Principles
- Modular & Extensible: Add new backbones, datasets, or transforms with minimal coupling.
- Reproducible: Consistent training/evaluation through config-driven architecture.
- MLOps-Ready: Integrates smoothly with DVC, MLflow, and CI/CD pipelines.
- Research-Grade: Clear structure supports custom experimentation and ablations.


---
> *`src/` is the brain of project — where every frame, node, and signal becomes language*