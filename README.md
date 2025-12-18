# [UIT-DS201] End-to-end VSL Alphabet Recognition Application
> This repository serves as the version control platform for the final-semester project "End-to-end VSL Alphabet Recognition Application" for subject DS201- Deep Learning for Data Science at University of Information Technology, Vietnam National University Ho Chi Minh city.

## Repository Structure

```bash
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

## Usage

### Step 1: Segment Videos

Segment raw videos into word-level clips using annotation files:

```bash
python run_cut_words.py <video_name.mp4>
```

**Example:**
```bash
python cut_words.py "10 Ký hiệu Cảm Xúc Cơ Bản - Sign Language Emotions.mp4"
```

This will:
- Read video from `DATA/Raw data/<video_name>.mp4`
- Read annotation from `DATA/Gold data/<video_name>.txt`
- Output segmented videos to `DATA/Segmented/<label>/<label>_XXX.mp4`

### Step 2: Preprocess Pose Data

Extract pose features from segmented videos and create train/val/test splits:

```bash
python -m src.data.preprocess_data \
    --raw_path DATA/Segmented \
    --output_path DATA/Processed_Pose \
    --num_frames 64 \
    --skeleton_layout mediapipe_27 \
    --normalize \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

**Arguments:**
- `--raw_path`: Path to segmented videos folder (default: `DATA/Segmented`)
- `--output_path`: Path to save processed .npz files (default: `DATA/Processed_Pose`)
- `--num_frames`: Number of frames to sample (default: `64`)
- `--skeleton_layout`: Skeleton layout type (default: `mediapipe_27`)
- `--normalize`: Normalize keypoints (default: `True`)
- `--train_ratio`: Train split ratio (default: `0.7`)
- `--val_ratio`: Validation split ratio (default: `0.15`)
- `--test_ratio`: Test split ratio (default: `0.15`)
- `--seed`: Random seed for splitting (default: `42`)

This will:
- Extract pose features from all videos using MediaPipe
- Save features as `.npz` files in `DATA/Processed_Pose/{train,val,test}/`
- Create `train.json`, `val.json`, `test.json` with file paths and labels

### Step 3: Train Model

Train a model using preprocessed data:

```bash
python main.py \
    --config configs/poseformer.yaml \
    --device cuda \
    --output-dir logs/poseformer
```

**Arguments:**
- `--config`: Path to config YAML file (required)
- `--resume`: Path to checkpoint to resume training
- `--eval-only`: Evaluation mode only (no training)
- `--checkpoint`: Path to checkpoint for evaluation
- `--batch-size`: Override batch size from config
- `--learning-rate`: Override learning rate from config
- `--num-epochs`: Override number of epochs from config
- `--device`: Override device from config (`cuda` or `cpu`)
- `--output-dir`: Override output directory from config

**Available Models:**
- `configs/poseformer.yaml` - PoseFormer (Transformer-based)
- `configs/stgcn.yaml` - ST-GCN (Spatial-Temporal Graph CNN)
- `configs/ha_gcn.yaml` - HA-GCN (Hierarchical Attention GCN)
- `configs/bi_lstm.yaml` - Bi-LSTM (Bidirectional LSTM)
- `configs/resnet_lstm.yaml` - ResNet-LSTM (RGB-based)
- `configs/videomae.yaml` - VideoMAE (Video Transformer)

**Example:**
```bash
# Train PoseFormer
python main.py --config configs/poseformer.yaml

# Train ST-GCN with custom batch size
python main.py --config configs/stgcn.yaml --batch-size 32

# Resume training from checkpoint
python main.py --config configs/poseformer.yaml --resume logs/poseformer/checkpoint_best.pth

# Evaluate only
python main.py --config configs/poseformer.yaml --eval-only --checkpoint logs/poseformer/checkpoint_best.pth
```

## Acknowledgement
The whole project team would like to express our sincere gratitude to our supervisor, [Mr. Nguyễn Tấn Hoàng Phước (PhD)](https://fit.uit.edu.vn/index.php/gioi-thieu/doi-ngu-nhan-su), from the Faculty of Information Science and Engineering, University of Information Technology (UIT), VNU-HCM. His invaluable guidance, insightful feedback, and unwavering support were instrumental in the completion of this project.
