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

## Acknowledgement
The whole project team would like to express our sincere gratitude to our supervisor, [Mr. Nguyễn Tấn Hoàng Phước (PhD)](https://fit.uit.edu.vn/index.php/gioi-thieu/doi-ngu-nhan-su), from the Faculty of Information Science and Engineering, University of Information Technology (UIT), VNU-HCM. His invaluable guidance, insightful feedback, and unwavering support were instrumental in the completion of this project.
