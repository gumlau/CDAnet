"""
Logging utilities for CDAnet training and evaluation.
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
import json

import torch
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Logger:
    """
    Multi-backend logger for experiment tracking.
    
    Args:
        log_dir: Directory for logging outputs
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard logging
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        config: Configuration dictionary for logging
    """
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 use_tensorboard: bool = True, use_wandb: bool = False,
                 wandb_project: str = 'cdanet', config: Optional[Dict] = None):
        
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard and HAS_TENSORBOARD
        self.use_wandb = use_wandb and HAS_WANDB
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup TensorBoard
        self.tb_writer = None
        if self.use_tensorboard:
            self._setup_tensorboard()
            
        # Setup W&B
        if self.use_wandb:
            self._setup_wandb(wandb_project, config)
            
        # Training state
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        
        self.info(f"Logger initialized for experiment: {experiment_name}")
        
    def _setup_file_logging(self):
        """Setup file-based logging."""
        log_file = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        
        # Create logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _setup_tensorboard(self):
        """Setup TensorBoard logging."""
        tb_dir = os.path.join(self.log_dir, 'tensorboard', self.experiment_name)
        os.makedirs(tb_dir, exist_ok=True)
        
        self.tb_writer = SummaryWriter(tb_dir)
        self.info("TensorBoard logging enabled")
        
    def _setup_wandb(self, project_name: str, config: Optional[Dict]):
        """Setup Weights & Biases logging."""
        wandb.init(
            project=project_name,
            name=self.experiment_name,
            config=config,
            dir=self.log_dir
        )
        self.info("W&B logging enabled")
        
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step
            
        # Log to file
        metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.info(f"Step {step} - {metrics_str}")
        
        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
                
        # Log to W&B
        if self.use_wandb:
            wandb_metrics = {**metrics, 'step': step}
            wandb.log(wandb_metrics)
            
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.info("Hyperparameters:")
        for key, value in hparams.items():
            self.info(f"  {key}: {value}")
            
        # Save to JSON file
        hparams_file = os.path.join(self.log_dir, f"{self.experiment_name}_hparams.json")
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
            
        # Log to TensorBoard
        if self.tb_writer:
            # Convert values to scalars for TB hparams logging
            scalar_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float)):
                    scalar_hparams[key] = value
                else:
                    scalar_hparams[key] = str(value)
            
            self.tb_writer.add_hparams(scalar_hparams, {'hparam/dummy': 0})
            
    def log_model_summary(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info("Model Summary:")
        self.info(f"  Total parameters: {total_params:,}")
        self.info(f"  Trainable parameters: {trainable_params:,}")
        self.info(f"  Input shape: {input_shape}")
        
        # Log model graph to TensorBoard (if possible)
        if self.tb_writer:
            try:
                # Create dummy input
                if len(input_shape) == 2:
                    dummy_input = torch.randn(1, *input_shape)
                else:
                    dummy_input = [torch.randn(1, *shape) for shape in input_shape]
                    
                # Log graph (this might not work for all model types)
                # self.tb_writer.add_graph(model, dummy_input)
                pass
            except Exception as e:
                self.warning(f"Could not log model graph to TensorBoard: {e}")
                
    def log_training_progress(self, epoch: int, batch_idx: int, total_batches: int,
                            train_loss: float, metrics: Dict[str, float]):
        """Log training progress within an epoch."""
        self.epoch = epoch
        self.step = epoch * total_batches + batch_idx
        
        # Calculate progress
        progress = 100.0 * batch_idx / total_batches
        elapsed = time.time() - self.start_time
        
        # Log progress
        if batch_idx % 50 == 0 or batch_idx == total_batches - 1:
            log_dict = {
                'train/loss': train_loss,
                'train/epoch': epoch,
                'train/progress': progress,
                'train/elapsed_time': elapsed,
                **{f'train/{k}': v for k, v in metrics.items()}
            }
            self.log_metrics(log_dict, self.step)
            
    def log_validation_results(self, epoch: int, val_metrics: Dict[str, float]):
        """Log validation results."""
        log_dict = {f'val/{k}': v for k, v in val_metrics.items()}
        log_dict['val/epoch'] = epoch
        
        self.log_metrics(log_dict, self.step)
        
        # Log best metrics
        if hasattr(self, 'best_val_loss'):
            if val_metrics.get('loss', float('inf')) < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.info(f"New best validation loss: {self.best_val_loss:.6f}")
        else:
            self.best_val_loss = val_metrics.get('loss', float('inf'))
            
    def log_learning_rate(self, lr: float, step: Optional[int] = None):
        """Log current learning rate."""
        if step is None:
            step = self.step
            
        self.log_metrics({'train/learning_rate': lr}, step)
        
    def save_checkpoint_info(self, checkpoint_path: str, epoch: int, 
                           metrics: Dict[str, float]):
        """Log checkpoint save information."""
        self.info(f"Checkpoint saved: {checkpoint_path}")
        self.info(f"  Epoch: {epoch}")
        for key, value in metrics.items():
            self.info(f"  {key}: {value:.6f}")
            
    def log_final_results(self, best_metrics: Dict[str, float], 
                         total_training_time: float):
        """Log final experiment results."""
        self.info("=" * 50)
        self.info("EXPERIMENT COMPLETED")
        self.info("=" * 50)
        self.info(f"Total training time: {total_training_time:.2f} seconds")
        self.info("Best validation metrics:")
        for key, value in best_metrics.items():
            self.info(f"  {key}: {value:.6f}")
        self.info("=" * 50)
        
    def close(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
            
        if self.use_wandb:
            wandb.finish()
            
        # Close file handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)