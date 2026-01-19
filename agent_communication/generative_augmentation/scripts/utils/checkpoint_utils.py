"""
Checkpoint management utilities

Handles:
- Saving model checkpoints
- Loading checkpoints for resume
- Managing checkpoint history
- Best model tracking
"""

import torch
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
from datetime import datetime


class CheckpointManager:
    """
    Manage model checkpoints during training

    Features:
    - Save checkpoints at intervals
    - Keep only N most recent checkpoints
    - Always keep best checkpoint
    - Resume from latest or specific checkpoint
    """

    def __init__(
        self,
        output_dir: str,
        keep_last_n: int = 3,
        save_optimizer: bool = True,
        monitor_metric: str = "val_fid",
        monitor_mode: str = "min"
    ):
        """
        Initialize checkpoint manager

        Args:
            output_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_optimizer: Whether to save optimizer state
            monitor_metric: Metric to monitor for best model
            monitor_mode: 'min' or 'max' for monitored metric
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        # Track checkpoint history
        self.checkpoint_history = []
        self.best_checkpoint = None
        self.best_score = float('inf') if monitor_mode == 'min' else float('-inf')

        # Load existing history if available
        self.load_history()

    def save_checkpoint(
        self,
        epoch: int,
        unet_lora,
        optimizer,
        lr_scheduler,
        ema_model=None,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None,
        is_best: bool = False,
        extra_state: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint

        Args:
            epoch: Current epoch number
            unet_lora: UNet model with LoRA weights
            optimizer: Optimizer instance
            lr_scheduler: Learning rate scheduler
            ema_model: Optional EMA model
            metrics: Dictionary of metrics
            config: Training configuration
            is_best: Whether this is the best checkpoint
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}.pt"
        checkpoint_path = self.output_dir / checkpoint_name

        # Prepare checkpoint dict
        checkpoint = {
            'epoch': epoch,
            'unet_lora_state': unet_lora.state_dict(),
            'timestamp': datetime.now().isoformat(),
        }

        # Add optimizer and scheduler if requested
        if self.save_optimizer:
            checkpoint['optimizer_state'] = optimizer.state_dict()
            checkpoint['lr_scheduler_state'] = lr_scheduler.state_dict()

        # Add EMA if available
        if ema_model is not None:
            checkpoint['ema_state'] = ema_model.state_dict()

        # Add metrics
        if metrics is not None:
            checkpoint['metrics'] = metrics

        # Add config
        if config is not None:
            checkpoint['config'] = config

        # Add extra state
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state

        # Add random state for reproducibility
        checkpoint['random_state'] = {
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update history
        self.checkpoint_history.append({
            'epoch': epoch,
            'path': str(checkpoint_path),
            'metrics': metrics,
            'timestamp': checkpoint['timestamp']
        })

        # Save latest symlink
        latest_path = self.output_dir / "checkpoint_latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(checkpoint_path.name)

        # Check if this is the best checkpoint
        if metrics is not None and self.monitor_metric in metrics:
            current_score = metrics[self.monitor_metric]
            is_new_best = False

            if self.monitor_mode == 'min':
                is_new_best = current_score < self.best_score
            else:
                is_new_best = current_score > self.best_score

            if is_new_best or is_best:
                self.best_score = current_score
                self.best_checkpoint = str(checkpoint_path)

                # Save best symlink
                best_path = self.output_dir / "checkpoint_best.pt"
                if best_path.exists() or best_path.is_symlink():
                    best_path.unlink()
                best_path.symlink_to(checkpoint_path.name)

                print(f"  New best checkpoint! {self.monitor_metric}: {current_score:.4f}")

        # Cleanup old checkpoints (keep last N + best)
        self.cleanup_old_checkpoints()

        # Save history
        self.save_history()

        print(f"  Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N and best"""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # Sort by epoch
        sorted_history = sorted(self.checkpoint_history, key=lambda x: x['epoch'])

        # Identify checkpoints to keep
        keep_indices = set(range(len(sorted_history) - self.keep_last_n, len(sorted_history)))

        # Also keep best checkpoint
        if self.best_checkpoint is not None:
            for i, ckpt in enumerate(sorted_history):
                if ckpt['path'] == self.best_checkpoint:
                    keep_indices.add(i)
                    break

        # Remove old checkpoints
        for i, ckpt in enumerate(sorted_history):
            if i not in keep_indices:
                ckpt_path = Path(ckpt['path'])
                if ckpt_path.exists():
                    ckpt_path.unlink()
                    print(f"  Removed old checkpoint: {ckpt_path}")

        # Update history
        self.checkpoint_history = [sorted_history[i] for i in sorted(keep_indices)]

    def load_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        unet_lora=None,
        optimizer=None,
        lr_scheduler=None,
        ema_model=None,
        load_optimizer: bool = True,
        strict: bool = True
    ) -> Dict:
        """
        Load a checkpoint

        Args:
            checkpoint_path: Path to checkpoint (or 'latest'/'best')
            unet_lora: UNet model to load weights into
            optimizer: Optimizer to load state into
            lr_scheduler: Scheduler to load state into
            ema_model: EMA model to load state into
            load_optimizer: Whether to load optimizer state
            strict: Strict loading of state dicts

        Returns:
            Checkpoint dictionary
        """
        # Resolve checkpoint path
        if checkpoint_path is None or checkpoint_path == "latest":
            checkpoint_path = self.output_dir / "checkpoint_latest.pt"
        elif checkpoint_path == "best":
            checkpoint_path = self.output_dir / "checkpoint_best.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load UNet LoRA weights
        if unet_lora is not None and 'unet_lora_state' in checkpoint:
            unet_lora.load_state_dict(checkpoint['unet_lora_state'], strict=strict)
            print("  Loaded UNet LoRA weights")

        # Load optimizer
        if optimizer is not None and load_optimizer and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("  Loaded optimizer state")

        # Load scheduler
        if lr_scheduler is not None and load_optimizer and 'lr_scheduler_state' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
            print("  Loaded scheduler state")

        # Load EMA
        if ema_model is not None and 'ema_state' in checkpoint:
            ema_model.load_state_dict(checkpoint['ema_state'])
            print("  Loaded EMA weights")

        # Restore random state
        if 'random_state' in checkpoint:
            if 'torch' in checkpoint['random_state']:
                torch.set_rng_state(checkpoint['random_state']['torch'])
            if 'torch_cuda' in checkpoint['random_state'] and torch.cuda.is_available():
                if checkpoint['random_state']['torch_cuda'] is not None:
                    torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda'])
            print("  Restored random state")

        print(f"  Resumed from epoch {checkpoint.get('epoch', 0)}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        latest_path = self.output_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            return str(latest_path)
        return None

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint"""
        best_path = self.output_dir / "checkpoint_best.pt"
        if best_path.exists():
            return str(best_path)
        return None

    def save_history(self):
        """Save checkpoint history to JSON"""
        history_path = self.output_dir / "checkpoint_history.json"

        history_data = {
            'checkpoint_history': self.checkpoint_history,
            'best_checkpoint': self.best_checkpoint,
            'best_score': self.best_score,
            'monitor_metric': self.monitor_metric,
            'monitor_mode': self.monitor_mode
        }

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

    def load_history(self):
        """Load checkpoint history from JSON"""
        history_path = self.output_dir / "checkpoint_history.json"

        if not history_path.exists():
            return

        with open(history_path, 'r') as f:
            history_data = json.load(f)

        self.checkpoint_history = history_data.get('checkpoint_history', [])
        self.best_checkpoint = history_data.get('best_checkpoint', None)
        self.best_score = history_data.get('best_score',
            float('inf') if self.monitor_mode == 'min' else float('-inf'))

    def get_checkpoint_info(self) -> Dict:
        """Get information about saved checkpoints"""
        return {
            'total_checkpoints': len(self.checkpoint_history),
            'latest_epoch': self.checkpoint_history[-1]['epoch'] if self.checkpoint_history else None,
            'best_checkpoint': self.best_checkpoint,
            'best_score': self.best_score,
            'checkpoints': self.checkpoint_history
        }
