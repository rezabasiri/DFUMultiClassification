"""
Training utilities for Stable Diffusion fine-tuning

Includes:
- Perceptual loss (VGG-based)
- EMA (Exponential Moving Average)
- Learning rate schedulers
- Training callbacks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, List
import copy
from pathlib import Path


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features

    Measures high-level perceptual similarity between images
    using intermediate features from a pre-trained VGG network
    """

    def __init__(
        self,
        layer_index: int = 16,
        device: str = "cuda"
    ):
        """
        Initialize perceptual loss

        Args:
            layer_index: Which VGG layer to use for features
                Common choices: 4 (conv1_2), 9 (conv2_2), 16 (conv3_3), 23 (conv4_3)
            device: Device for computation
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features

        # Extract layers up to layer_index
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:layer_index + 1])

        # Freeze parameters (no training)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Move to device and set to eval mode
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # VGG normalization (ImageNet stats)
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize images for VGG

        Args:
            x: Input images in [0, 1] range

        Returns:
            Normalized images
        """
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss

        Args:
            pred: Predicted images [N, C, H, W] in [0, 1] range
            target: Target images [N, C, H, W] in [0, 1] range

        Returns:
            Perceptual loss scalar
        """
        # Ensure images are in [0, 1] range (convert from [-1, 1] if needed)
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2

        # Normalize for VGG
        pred_normalized = self.normalize(pred)
        target_normalized = self.normalize(target)

        # Extract features
        with torch.no_grad():
            target_features = self.feature_extractor(target_normalized)

        pred_features = self.feature_extractor(pred_normalized)

        # Compute MSE in feature space
        loss = F.mse_loss(pred_features, target_features)

        return loss


class EMAModel:
    """
    Exponential Moving Average of model weights

    Maintains a smoothed version of model parameters for better generation quality
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None
    ):
        """
        Initialize EMA model

        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update, typical: 0.9999)
            device: Device for EMA parameters
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device

        # Create EMA copy of model
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Freeze EMA parameters
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters

        Args:
            model: Current model with updated parameters
        """
        # Update each parameter using EMA formula:
        # ema_param = decay * ema_param + (1 - decay) * current_param
        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data,
                alpha=1 - self.decay
            )

    def state_dict(self) -> Dict:
        """Get EMA model state dict"""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        """Load EMA model state dict"""
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        """Move EMA model to device"""
        self.ema_model.to(device)
        return self


class EarlyStoppingTracker:
    """
    Track validation metrics and trigger early stopping
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.01,
        monitor: str = "val_fid",
        mode: str = "min"
    ):
        """
        Initialize early stopping tracker

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor (e.g., 'val_fid', 'val_loss')
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Update tracker with new metrics

        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics

        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            print(f"Warning: Monitored metric '{self.monitor}' not found in metrics")
            return False

        current_score = metrics[self.monitor]

        # Check if improved
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            # Improvement! Reset counter
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement, increment counter
            self.counter += 1

            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}")
                self.early_stop = True
                return True

            return False

    def get_state(self) -> Dict:
        """Get tracker state for checkpointing"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'early_stop': self.early_stop
        }

    def load_state(self, state: Dict):
        """Load tracker state from checkpoint"""
        self.best_score = state['best_score']
        self.best_epoch = state['best_epoch']
        self.counter = state['counter']
        self.early_stop = state['early_stop']


class MetricsTracker:
    """
    Track training and validation metrics over time
    """

    def __init__(self):
        """Initialize metrics tracker"""
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_fid': [],
            'val_ssim': [],
            'val_lpips': [],
            'val_is': [],
            'learning_rate': [],
            'epoch': []
        }

    def update(
        self,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: Optional[float] = None
    ):
        """
        Update metrics

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
            learning_rate: Current learning rate
        """
        self.history['epoch'].append(epoch)

        if train_loss is not None:
            self.history['train_loss'].append(train_loss)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)

        if val_metrics is not None:
            if 'fid' in val_metrics:
                self.history['val_fid'].append(val_metrics['fid'])
            if 'ssim_mean' in val_metrics:
                self.history['val_ssim'].append(val_metrics['ssim_mean'])
            if 'lpips_mean' in val_metrics:
                self.history['val_lpips'].append(val_metrics['lpips_mean'])
            if 'is_mean' in val_metrics:
                self.history['val_is'].append(val_metrics['is_mean'])

        if learning_rate is not None:
            self.history['learning_rate'].append(learning_rate)

    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value of a metric"""
        if metric in self.history and len(self.history[metric]) > 0:
            return self.history[metric][-1]
        return None

    def get_best(self, metric: str, mode: str = "min") -> Tuple[float, int]:
        """
        Get best value of a metric and corresponding epoch

        Args:
            metric: Metric name
            mode: 'min' or 'max'

        Returns:
            (best_value, best_epoch)
        """
        if metric not in self.history or len(self.history[metric]) == 0:
            return None, None

        values = self.history[metric]
        epochs = self.history['epoch'][:len(values)]

        if mode == "min":
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))

        return values[best_idx], epochs[best_idx]

    def save(self, filepath: str):
        """Save metrics history to file"""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load(self, filepath: str):
        """Load metrics history from file"""
        import json

        with open(filepath, 'r') as f:
            self.history = json.load(f)


def get_optimizer(
    parameters,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-5,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.01,
    epsilon: float = 1e-8
):
    """
    Create optimizer

    Args:
        parameters: Model parameters to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'adamw8bit')
        learning_rate: Learning rate
        beta1: Adam beta1
        beta2: Adam beta2
        weight_decay: Weight decay
        epsilon: Adam epsilon

    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=epsilon
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
            eps=epsilon
        )
    elif optimizer_type == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=epsilon
            )
        except ImportError:
            print("Warning: bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=epsilon
            )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_lr_scheduler(
    optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    num_cycles: int = 1
):
    """
    Create learning rate scheduler

    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps
        num_cycles: Number of cycles (for cosine scheduler)

    Returns:
        Scheduler instance
    """
    from diffusers.optimization import get_scheduler

    return get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles
    )
