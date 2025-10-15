# DETAILED PROJECT DESCRIPTION

## 1. General Description

### 1.1. Project Name
> **Word-Level Vietnamese Sign Language Recognition System**

### 1.2. Project Overview
FineSign is an intelligent system designed for word-level sign language recognition using video input. The core idea is to enable machines to interpret human gestures, particularly sign language, with precision and temporal awareness. The project aims to build a modular and reusable AI pipeline capable of fine-tuning video-based models for various gesture-recognition tasks.

This system will serve as a foundation for future developments, such as sentence-level translation, multimodal fusion (vision + text), or even graph-based hand–body representation learning.

## 2. Core idea
Modern sign language recognition requires understanding subtle, high-frequency movements of hands, arms, and facial expressions—something that static image models fail to capture.

To address this, FineSign leverages video-based action recognition architectures (e.g., Timesformer, VideoMAE, SlowFast, I3D) and adapts them for fine-grained word-level classification.

The project focuses on:
- Efficient data preprocessing and augmentation for temporal consistency.
- Fine-tuning pretrained video models on custom sign datasets (e.g., WLASL or custom-recorded videos).
- Designing a scalable, version-controlled ML pipeline for experimentation, model tracking, and deployment.

## 3. System Implementation
The workflow is structured as a modular MLOps-ready system consisting of the following major components:
### 3.1. Data Pipeline
- **Dataset Ingestion**: Raw videos organized by class labels (e.g., “hello”, “thank you”).
- **Preprocessing**: Frame extraction, resizing, normalization, and temporal alignment.
- **Augmentation**: Random cropping, flipping, color jitter, and time-shifting for better generalization.
- **Dataset Split**: Train/val/test split handled by YAML config.

### 3.2. Model Pipeline
- **Base Model**: Pretrained transformer-based video model (e.g., Timesformer or VideoMAE).
- **Head Layer (Classifier)**: Replace top layers with a custom classifier head for sign-language-specific classes.
- **Fine-tuning**: Using PyTorch Lightning for efficient training, logging, and checkpointing.
- **Metrics**: Accuracy, F1-score, confusion matrix for per-class analysis.

### 3.3. Experiment Tracking
- **Versioning**: Managed with DVC (Data Version Control).
- **Experiment tracking**: Done via MLflow or Weights & Biases (wandb) for metrics, artifacts, and reproducibility.
- **Configuration**: YAML-driven parameters (learning rate, batch size, model type, etc.) for reproducible runs.

### 3.4. Inference and Evaluation
- **Inference module**: Takes a video input → extracts frames → runs through the fine-tuned model → outputs the predicted sign label.
- **Evaluation script**: Computes metrics and visualizes confusion matrix or per-sign accuracy.
- **Export**: Supports torchscript or onnx format for deployment.

## 4. Repository Structure
```php

ds201-hand-sign-language/     # root repo
├─ configs/                   # all configs (yaml)
│  ├─ default.yaml
│  ├─ train_videomae.yaml
│  ├─ finetune_posefusion.yaml
│  └─ inference.yaml
├─ data/                      # data pointers (gitignored)
│  ├─ raw/                    # raw videos (managed by DVC or gdrive links)
│  ├─ processed/              # processed frames / npz / lmdb
│  └─ manifests/              # csv / jsonl list of samples
├─ src/                       # main code
│  ├─ __init__.py
│  ├─ datasets/
│  │  ├─ __init__.py
│  │  ├─ base_dataset.py      # generic dataset wrapper
│  │  ├─ video_dataset.py     # RGB frame loader + sampling
│  │  └─ pose_dataset.py      # pose/keypoint loader
│  ├─ transforms/
│  │  ├─ video_transforms.py  # temporal sampling, resize, crop
│  │  └─ pose_transforms.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ backbones/
│  │  │  ├─ videomae_backbone.py
│  │  │  └─ timesformer_backbone.py
│  │  ├─ heads/
│  │  │  ├─ linear_head.py
│  │  │  └─ fusion_head.py    # fusion of pose + video
│  │  └─ graph/
│  │     └─ stgcn.py          # spatial-temporal GCN
│  ├─ trainers/
│  │  ├─ trainer.py           # training loop (checkpointing, resume)
│  │  └─ hf_trainer_wrapper.py# optional wrapper using HF Trainer
│  ├─ utils/
│  │  ├─ metrics.py
│  │  ├─ logger.py           # wandb/MLflow logging helpers
│  │  ├─ seed.py
│  │  └─ checkpoints.py
│  ├─ inference/
│  │  └─ infer.py
│  └─ export/
│     └─ export_onnx.py
├─ scripts/                   # small CLI scripts
│  ├─ train.sh
│  ├─ finetune.sh
│  ├─ eval.sh
│  └─ export.sh
├─ experiments/               # experiments metadata (gitignored)
│  ├─ exp_2025-10-16_v1/
│  │  ├─ config.yaml
│  │  ├─ metrics.json
│  │  └─ README.md
├─ checkpoints/               # saved ckpts (gitignored or via artifact store)
├─ tests/                     # unit + integration tests
│  ├─ test_dataset.py
│  └─ test_model_forward.py
├─ README.md
├─ LICENSE
├─ .gitignore
├─ pyproject.toml / requirements.txt
├─ environment.yml            # optional conda env
├─ Dockerfile
├─ docker-compose.yml
└─ Makefile
```

## 5. Vision
FineSign is not just an isolated deep learning model—it’s a research-grade framework that grows into a full-fledged sign language understanding platform. It connects vision, linguistics, and accessibility, bridging communication gaps for the hearing-impaired community while advancing gesture recognition research.