"""
Training module for Vietnamese sign language recognition models.
Includes training loop, validation, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Callable
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from ..utils.logger import get_logger


class Trainer:
    """
    Trainer class for hand sign recognition models.
    
    Features:
    - CrossEntropy loss with optional label smoothing
    - Adam optimizer with weight decay
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training
    - Early stopping
    - Checkpointing
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, config: Optional[Dict] = None, device: str = "cuda", logger=None):
        """
        Initialize trainer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger or get_logger()
        
        # Configuration with defaults
        self.config = config or {}
        self.batch_size = self.config.get("batch_size", 32)
        self.num_epochs = self.config.get("num_epochs", 50)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.optimizer_type = self.config.get("optimizer", "adam").lower()
        self.weight_decay = self.config.get("weight_decay", 1e-4)
        self.clip_grad_norm = self.config.get("clip_grad_norm", 5.0)
        self.label_smoothing = self.config.get("label_smoothing", 0.0)
        self.use_amp = self.config.get("use_amp", False)
        self.scheduler_type = self.config.get("scheduler", "cosine")
        self.scheduler_params = self.config.get("scheduler_params", {})
        self.warmup_steps = self.config.get("warmup_steps", None)
        self.warmup_epochs = self.config.get("warmup_epochs", 0)
        self.early_stopping_patience = self.config.get("early_stopping_patience", None)
        
        # Loss function
        if self.label_smoothing > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0  # Best F1 score (used for model selection)
        self.best_val_acc = 0.0  # Best accuracy (for logging)
        self.patience_counter = 0
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_f1s = []
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration"""
        if self.scheduler_type is None:
            return None
        
        if self.scheduler_type == "cosine":
            T_max = self.num_epochs
            if self.warmup_epochs > 0:
                T_max -= self.warmup_epochs
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                **self.scheduler_params
            )
        elif self.scheduler_type == "step":
            step_size = self.scheduler_params.get("step_size", 10)
            gamma = self.scheduler_params.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                **self.scheduler_params
            )
        elif self.scheduler_type == "warmup_cosine":
            # Custom warmup + cosine scheduler
            from torch.optim.lr_scheduler import LambdaLR
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return epoch / self.warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)))
            return LambdaLR(self.optimizer, lr_lambda)
        else:
            self.logger.warning(f"Unknown scheduler type: {self.scheduler_type}, using None")
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare data
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch
                # Handle skeleton keypoint data (for Bi-LSTM, PoseFormer)
                # Input shape from dataset: (B, T, V, C) where T=frames, V=joints, C=coords
                if isinstance(inputs, torch.Tensor) and inputs.dim() == 4:
                    # Check model type to determine how to reshape
                    model_type = type(self.model).__name__.lower()
                    if 'bilstm' in model_type or 'bi_lstm' in model_type:
                        # Bi-LSTM needs (B, T, V*C) = (B, T, 81)
                        B, T, V, C = inputs.shape
                        inputs = inputs.view(B, T, V * C)
                    elif 'poseformer' in model_type:
                        # PoseFormer needs (B, C, T, V, M) = (B, 3, T, 27, 1)
                        inputs = inputs.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
                        inputs = inputs.unsqueeze(-1)  # (B, C, T, V, 1)
                    # For other models, keep as is
            elif isinstance(batch, dict):
                # Handle graph data (for ST-GCN, HA-GCN)
                if 'x' in batch:
                    # Graph data: extract 'x' and convert to (N, C, T, V, M) format
                    x = batch['x']  # (B, T, V, C) or (B, T, V, 3)
                    if x.dim() == 4:
                        # Convert (B, T, V, C) -> (B, C, T, V, 1)
                        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
                        x = x.unsqueeze(-1)  # (B, C, T, V, 1)
                    inputs = x
                else:
                    inputs = batch
                labels = batch.get('y', batch.get('label'))
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            inputs = self._move_to_device(inputs)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs.get('output', outputs))
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get('logits', outputs.get('output', outputs))
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Handle empty loader case
        if len(self.train_loader) == 0:
            self.logger.warning("Train loader is empty! Check batch_size and dataset size.")
            return {
                'loss': 0.0,
                'accuracy': 0.0
            }
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Prepare data
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch
                    # Handle skeleton keypoint data (for Bi-LSTM, PoseFormer)
                    if isinstance(inputs, torch.Tensor) and inputs.dim() == 4:
                        # Check model type to determine how to reshape
                        model_type = type(self.model).__name__.lower()
                        if 'bilstm' in model_type or 'bi_lstm' in model_type:
                            # Bi-LSTM needs (B, T, V*C) = (B, T, 81)
                            B, T, V, C = inputs.shape
                            inputs = inputs.view(B, T, V * C)
                        elif 'poseformer' in model_type:
                            # PoseFormer needs (B, C, T, V, M) = (B, 3, T, 27, 1)
                            inputs = inputs.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
                            inputs = inputs.unsqueeze(-1)  # (B, C, T, V, 1)
                        # For other models, keep as is
                elif isinstance(batch, dict):
                    # Handle graph data (for ST-GCN, HA-GCN)
                    if 'x' in batch:
                        # Graph data: extract 'x' and convert to (N, C, T, V, M) format
                        x = batch['x']  # (B, T, V, C) or (B, T, V, 3)
                        if x.dim() == 4:
                            # Convert (B, T, V, C) -> (B, C, T, V, 1)
                            x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
                            x = x.unsqueeze(-1)  # (B, C, T, V, 1)
                        inputs = x
                    else:
                        inputs = batch
                    labels = batch.get('y', batch.get('label'))
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")
                
                inputs = self._move_to_device(inputs)
                labels = labels.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast(device_type='cuda'):
                        outputs = self.model(inputs)
                        if isinstance(outputs, dict):
                            outputs = outputs.get('logits', outputs.get('output', outputs))
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs.get('logits', outputs.get('output', outputs))
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions and labels for F1 calculation
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        # Calculate F1-macro score
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_macro': f1_macro
        }
    
    def _move_to_device(self, inputs):
        """Move inputs to device (handles dict, list, tensor)"""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)([x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        else:
            return inputs
    
    def train(self, output_dir: Optional[str] = None):
        """
        Main training loop
        
        Args:
            output_dir: Directory to save checkpoints
        """
        output_dir = Path(output_dir) if output_dir else Path("./logs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Optimizer: {self.optimizer_type}")
        self.logger.info(f"Weight decay: {self.weight_decay}")
        self.logger.info(f"Label smoothing: {self.label_smoothing}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics.get('loss', 0))
                self.val_accs.append(val_metrics.get('accuracy', 0))
                self.val_f1s.append(val_metrics.get('f1_macro', 0))
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Logging
            log_msg = f"Epoch {epoch + 1}/{self.num_epochs} - "
            log_msg += f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%"
            if val_metrics:
                log_msg += f" | Val Loss: {val_metrics.get('loss', 0):.4f}, Val Acc: {val_metrics.get('accuracy', 0):.2f}%, Val F1: {val_metrics.get('f1_macro', 0):.4f}"
            log_msg += f" | LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            self.logger.info(log_msg)
            
            # Checkpointing - Best model based on F1-macro score
            is_best = False
            if val_metrics:
                val_loss = val_metrics.get('loss', float('inf'))
                val_acc = val_metrics.get('accuracy', 0)
                val_f1 = val_metrics.get('f1_macro', 0)
                
                # Update best loss (for logging)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                
                # Update best accuracy (for logging)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                
                # Best model selection based on F1-macro score
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    is_best = True
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            # Save checkpoint only when F1 improves 
            if is_best:
                self.save_checkpoint(output_dir, epoch, is_best=is_best)
            
            # Early stopping
            if self.early_stopping_patience and self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        if self.val_loader:
            self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
            self.logger.info(f"Best validation F1-macro: {self.best_val_f1:.4f}")
    
    def save_checkpoint(self, output_dir: Path, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'val_f1s': self.val_f1s,
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        
        # Save best
        if is_best:
            # Remove old best checkpoint if exists
            old_best = output_dir / "checkpoint_best.pth"
            if old_best.exists():
                old_best.unlink()
            
            # Save new best checkpoint
            torch.save(checkpoint, output_dir / "checkpoint_best.pth")
            self.logger.info(f"Saved best model checkpoint (F1: {self.best_val_f1:.4f}) at epoch {epoch + 1}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_f1 = checkpoint.get('best_val_f1', checkpoint.get('best_val_metric', 0.0))  # Backward compatibility
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        # Load history if available
        if 'val_f1s' in checkpoint:
            self.val_f1s = checkpoint['val_f1s']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")