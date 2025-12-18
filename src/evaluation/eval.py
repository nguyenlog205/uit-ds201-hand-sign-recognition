"""
Evaluation module for Vietnamese sign language recognition models.
Computes F1-macro, Precision, Recall, and Accuracy metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm

from ..utils.logger import get_logger


class Evaluator:
    """
    Evaluator class for Vietnamese sign language recognition models.
    
    Computes:
    - F1-macro score
    - Precision (macro-averaged)
    - Recall (macro-averaged)
    - Accuracy
    """
    
    def __init__(self, model: nn.Module, device: str = "cuda", logger=None):
        """
        Initialize evaluator
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger or get_logger()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = False,
        use_amp: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            return_predictions: Whether to return predictions and labels
            use_amp: Use automatic mixed precision
        
        Returns:
            Dictionary with metrics (and optionally predictions/labels)
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
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
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # Extract logits if output is dict
                if isinstance(outputs, dict):
                    outputs = outputs.get('logits', outputs.get('output', outputs))
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_predictions)
        
        if return_predictions:
            return {
                **metrics,
                'predictions': all_predictions,
                'labels': all_labels
            }
        
        return metrics
    
    def _compute_metrics(self, labels: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            labels: Ground truth labels
            predictions: Predicted labels
        
        Returns:
            Dictionary with metrics
        """
        # Accuracy
        accuracy = accuracy_score(labels, predictions)
        
        # F1-macro
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        
        # Precision (macro-averaged)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        
        # Recall (macro-averaged)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall,
        }
        
        # Log metrics
        self.logger.info("Evaluation Metrics:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  F1-macro: {f1_macro:.4f}")
        self.logger.info(f"  Precision (macro): {precision:.4f}")
        self.logger.info(f"  Recall (macro): {recall:.4f}")
        
        return metrics
    
    def _move_to_device(self, inputs):
        """Move inputs to device (handles dict, list, tensor)."""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return type(inputs)([x.to(self.device) if isinstance(x, torch.Tensor) else x for x in inputs])
        elif isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        else:
            return inputs
    
    def evaluate_class_wise(
        self,
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        use_amp: bool = False,
    ) -> Dict:
        """
        Evaluate model with per-class metrics.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: List of class names (optional)
            use_amp: Use automatic mixed precision
        
        Returns:
            Dictionary with overall and per-class metrics
        """
        result = self.evaluate(data_loader, return_predictions=True, use_amp=use_amp)
        
        labels = result['labels']
        predictions = result['predictions']
        
        # Per-class metrics
        num_classes = len(np.unique(labels))
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        per_class_metrics = {}
        for i in range(num_classes):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_labels = labels[class_mask]
                class_preds = predictions[class_mask]
                
                class_acc = accuracy_score(class_labels, class_preds)
                class_precision = precision_score(
                    labels, predictions, labels=[i], average='macro', zero_division=0
                )
                class_recall = recall_score(
                    labels, predictions, labels=[i], average='macro', zero_division=0
                )
                class_f1 = f1_score(
                    labels, predictions, labels=[i], average='macro', zero_division=0
                )
                
                per_class_metrics[class_names[i]] = {
                    'accuracy': class_acc,
                    'precision': class_precision,
                    'recall': class_recall,
                    'f1': class_f1,
                    'support': np.sum(class_mask)
                }
        
        return {
            'overall': {k: v for k, v in result.items() if k not in ['predictions', 'labels']},
            'per_class': per_class_metrics
        }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    use_amp: bool = False,
    return_predictions: bool = False,
) -> Dict[str, float]:
    """
    Function to evaluate a model.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to evaluate on
        use_amp: Use automatic mixed precision
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Dictionary with metrics (and optionally predictions/labels)
    """
    evaluator = Evaluator(model, device=device)
    return evaluator.evaluate(data_loader, return_predictions=return_predictions, use_amp=use_amp)