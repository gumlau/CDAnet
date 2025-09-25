"""
Training pipeline for CDAnet with physics-informed loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

from ..models.cdanet import CDAnet
from ..training.losses import CDAnetLoss, RRMSELoss
from ..data.data_loader import RBDataModule
from ..utils.logger import Logger
from ..utils.metrics import MetricsCalculator
from ..config.config import ExperimentConfig


class CDAnetTrainer:
    """
    Trainer class for CDAnet model.
    
    Args:
        config: Experiment configuration
        model: CDAnet model
        data_module: Data module for loading datasets
        logger: Logger for tracking experiments
    """
    
    def __init__(self, config: ExperimentConfig, model: CDAnet, 
                 data_module: RBDataModule, logger: Logger):
        
        self.config = config
        self.model = model
        self.data_module = data_module
        self.logger = logger
        
        # Device setup with automatic detection
        self.device, self.device_type = self._setup_device(config.training.device)
        
        # Convert model to float32 for MPS compatibility
        if self.device_type == 'mps':
            self.model = self.model.float()
        
        self.model.to(self.device)
        
        # Loss functions
        self.loss_fn = CDAnetLoss(
            lambda_pde=config.loss.lambda_pde,
            Ra=config.loss.Ra,
            Pr=config.loss.Pr,
            Lx=config.loss.Lx,
            Ly=config.loss.Ly,
            regression_norm=config.loss.regression_norm,
            pde_norm=config.loss.pde_norm
        )
        
        self.rrmse_fn = RRMSELoss()
        self.metrics_calc = MetricsCalculator()
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training setup based on device
        # Only enable AMP for CUDA (MPS support is limited, CPU doesn't benefit)
        self.use_amp = config.training.use_amp and self.device_type == 'cuda'
        self.scaler = None
        
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled for CUDA")
        elif config.training.use_amp and self.device_type != 'cuda':
            self.logger.info(f"Mixed precision training disabled for {self.device_type} (only supported on CUDA)")
            
        # Set default dtype to float32 for MPS compatibility
        if self.device_type == 'mps':
            torch.set_default_dtype(torch.float32)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics tracking
        self.train_metrics_history = defaultdict(list)
        self.val_metrics_history = defaultdict(list)
        
        # Create output directories
        os.makedirs(config.training.output_dir, exist_ok=True)
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        
        self.logger.info(f"Trainer initialized on device: {self.device} (type: {self.device_type})")
        
    def _setup_device(self, device_config: str) -> Tuple[torch.device, str]:
        """Setup device with automatic detection for CUDA, MPS, or CPU."""
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_type = 'cuda'
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                device_type = 'mps'
            else:
                device = torch.device('cpu')
                device_type = 'cpu'
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_type = 'cuda'
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
                device_type = 'cpu'
        elif device_config == 'mps':
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                device_type = 'mps'
            else:
                self.logger.warning("MPS requested but not available, falling back to CPU")
                device = torch.device('cpu')
                device_type = 'cpu'
        else:
            device = torch.device('cpu')
            device_type = 'cpu'
            
        return device, device_type
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        config = self.config.optimizer
        
        if config.optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer_type}")
            
        return optimizer
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        config = self.config.optimizer
        
        if config.scheduler_type.lower() == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.factor,
                patience=config.patience,
                min_lr=config.min_lr,
                verbose=True
            )
        elif config.scheduler_type.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=config.min_lr
            )
        elif config.scheduler_type.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.patience,
                gamma=config.factor
            )
        else:
            scheduler = None
            
        return scheduler
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Log hyperparameters
        hparams = self.config.to_dict()
        self.logger.log_hyperparameters(hparams)
        
        # Log model summary
        dummy_input_shape = [(self.config.data.clip_length, 4, 128, 128), (1000, 3)]
        self.logger.log_model_summary(self.model, dummy_input_shape)
        
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                if epoch % self.config.training.val_interval == 0:
                    val_metrics = self._validate_epoch()
                    self.logger.log_validation_results(epoch, val_metrics)

                    # Check if validation returned valid metrics
                    val_loss = val_metrics.get('total_loss', float('inf'))
                    if np.isnan(val_loss) or np.isinf(val_loss):
                        self.logger.warning(f"Invalid validation loss: {val_loss}. Skipping scheduler step.")
                        val_loss = float('inf')

                    # Update learning rate scheduler
                    if self.scheduler and isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    elif self.scheduler:
                        self.scheduler.step()

                    # Check for improvement
                    current_val_loss = val_loss
                    if current_val_loss < self.best_val_loss - self.config.training.min_delta:
                        self.best_val_loss = current_val_loss
                        self.epochs_without_improvement = 0
                        
                        # Save best model
                        if self.config.training.save_best:
                            self._save_checkpoint('best_model.pth', epoch, val_metrics, is_best=True)
                    else:
                        self.epochs_without_improvement += 1
                        
                    # Early stopping
                    if (self.config.training.early_stopping and 
                        self.epochs_without_improvement >= self.config.training.patience):
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                # Save periodic checkpoint
                if epoch % self.config.training.checkpoint_interval == 0:
                    checkpoint_name = f'checkpoint_epoch_{epoch}.pth'
                    metrics = val_metrics if 'val_metrics' in locals() else train_metrics
                    self._save_checkpoint(checkpoint_name, epoch, metrics)
                    
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.log_learning_rate(current_lr)
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            training_time = time.time() - training_start_time
            self.logger.log_final_results({'best_val_loss': self.best_val_loss}, training_time)
            
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = defaultdict(list)
        
        # Get training dataloader
        Ra = self.config.data.Ra_numbers[0]
        train_loader = self.data_module.get_dataloader(Ra, 'train')
        clip_value = getattr(self.config.optimizer, 'grad_clip_value', None)
        if clip_value is None:
            clip_value = getattr(self.config.optimizer, 'grad_clip_max_norm', None)

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass and loss computation
            loss, metrics = self._compute_loss(batch, compute_pde=True)
            
            # Check for NaN loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"Invalid loss detected: {loss.item()}. Skipping backward pass.")
                self.optimizer.zero_grad()
                continue

            # Backward pass
            if self.device_type == 'cuda' and self.scaler:
                # CUDA with GradScaler
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                if clip_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # MPS or CPU (no GradScaler)
                loss.backward()

                # Gradient clipping
                if clip_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)

                self.optimizer.step()

            self.optimizer.zero_grad()
            
            # Update metrics
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
                
            # Log progress
            if batch_idx % self.config.training.log_interval == 0:
                self.logger.log_training_progress(
                    self.current_epoch, batch_idx, len(train_loader),
                    metrics['total_loss'], {k: v for k, v in metrics.items() if k != 'total_loss'}
                )
                
            self.global_step += 1
            
            # Limit batches per epoch if specified
            if (self.config.training.clips_per_epoch and 
                batch_idx >= self.config.training.clips_per_epoch // self.config.data.batch_size):
                break
                
        # Average epoch metrics
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        
        # Store metrics history
        for key, value in avg_metrics.items():
            self.train_metrics_history[f'train_{key}'].append(value)
            
        return avg_metrics
    
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_metrics = defaultdict(list)
        
        Ra = self.config.data.Ra_numbers[0]
        val_loader = self.data_module.get_dataloader(Ra, 'val')
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                # Forward pass (without PDE loss for efficiency)
                loss, metrics = self._compute_loss(batch, compute_pde=False)
                
                # Update metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                    
        # Average metrics with NaN handling
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:  # Check if list is not empty
                mean_val = np.mean(values)
                # Replace NaN/Inf with large values for loss-based metrics
                if np.isnan(mean_val) or np.isinf(mean_val):
                    if 'loss' in key.lower():
                        avg_metrics[key] = float('inf')
                    else:
                        avg_metrics[key] = float('nan')
                else:
                    avg_metrics[key] = mean_val
            else:
                # No values computed, set defaults
                if 'loss' in key.lower():
                    avg_metrics[key] = float('inf')
                else:
                    avg_metrics[key] = float('nan')

        # Ensure 'total_loss' key exists
        if 'total_loss' not in avg_metrics:
            avg_metrics['total_loss'] = float('inf')

        # Store metrics history
        for key, value in avg_metrics.items():
            self.val_metrics_history[f'val_{key}'].append(value)

        return avg_metrics
    
    def _compute_loss(self, batch: Dict, compute_pde: bool = True) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch."""
        low_res = batch['low_res']  # [B, 8, 4, H_low, W_low]
        targets = batch['targets']  # [B, N, 4]
        coords = batch['coords']   # [B, N, 3]
        
        if compute_pde and 'pde_coords' in batch:
            pde_coords = batch['pde_coords']  # [B, N_pde, 3]
            pde_targets = batch['pde_targets']  # [B, N_pde, 4]
        else:
            pde_coords = None
            
        # Forward pass through model with appropriate autocast context
        if self.use_amp and self.device_type == 'cuda':
            # Use autocast only for CUDA
            with torch.cuda.amp.autocast():
                if compute_pde and pde_coords is not None:
                    predictions, derivatives = self.model.forward_with_derivatives(low_res, pde_coords)
                    # Use pde_targets when computing PDE loss
                    total_loss, loss_dict = self.loss_fn(predictions, pde_targets, derivatives, pde_coords)
                else:
                    predictions = self.model(low_res, coords)
                    derivatives = None
                    # Use regular targets for regression only
                    total_loss, loss_dict = self.loss_fn(predictions, targets, derivatives, coords)
        else:
            if compute_pde and pde_coords is not None:
                predictions, derivatives = self.model.forward_with_derivatives(low_res, pde_coords)
                # Use pde_targets when computing PDE loss
                total_loss, loss_dict = self.loss_fn(predictions, pde_targets, derivatives, pde_coords)
            else:
                predictions = self.model(low_res, coords)
                derivatives = None
                # Use regular targets for regression only
                total_loss, loss_dict = self.loss_fn(predictions, targets, derivatives, coords)
        
        # Compute additional metrics (use appropriate targets)
        if compute_pde and pde_coords is not None and 'pde_targets' in locals():
            additional_metrics = self.rrmse_fn(predictions, pde_targets)
        else:
            additional_metrics = self.rrmse_fn(predictions, targets)
        loss_dict.update(additional_metrics)
        
        return total_loss, loss_dict
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                # Convert to float32 if using MPS (doesn't support float64)
                if self.device_type == 'mps' and value.dtype == torch.float64:
                    value = value.float()
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'metrics': metrics
        }
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.logger.save_checkpoint_info(checkpoint_path, epoch, metrics)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
            # Try saving with a different name
            backup_path = checkpoint_path.replace('.pth', '_backup.pth')
            try:
                # Save only essential parts
                minimal_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config.to_dict()
                }
                torch.save(minimal_checkpoint, backup_path)
                self.logger.info(f"Saved minimal checkpoint to {backup_path}")
            except Exception as e2:
                self.logger.error(f"Failed to save backup checkpoint: {e2}")
                # Continue training without saving this checkpoint
        
        if is_best:
            val_loss = metrics.get('total_loss', metrics.get('loss', float('inf')))
            if isinstance(val_loss, (int, float)) and not (np.isnan(val_loss) or np.isinf(val_loss)):
                self.logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
            else:
                self.logger.info(f"New best model saved with validation loss: {val_loss}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
        # Load scaler state
        if load_optimizer and self.scaler and 'scaler_state_dict' in checkpoint:
            if checkpoint['scaler_state_dict'] is not None:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
        
        return checkpoint
