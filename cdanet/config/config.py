"""
Configuration classes for CDAnet training and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch


@dataclass 
class ModelConfig:
    """Configuration for CDAnet model architecture."""
    # U-Net parameters
    in_channels: int = 4  # T, p, u, v
    feature_channels: int = 32
    base_channels: int = 16
    
    # MLP parameters  
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = 'softplus'  # 'softplus', 'relu', 'tanh'
    coord_dim: int = 3  # x, y, t
    output_dim: int = 4  # T, p, u, v
    igres: Tuple[int, int, int] = (4, 64, 64)
    unet_nf: int = 16
    unet_mf: int = 256
    imnet_nf: int = 32


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    # Data paths
    data_dir: str = './rb_data_numerical'
    
    # Downsampling parameters
    spatial_downsample: int = 4  # γ_s
    temporal_downsample: int = 2  # γ_t
    clip_length: int = 8
    
    # Physics parameters
    Ra_numbers: List[float] = field(default_factory=lambda: [1e5])
    domain_size: Tuple[float, float] = (3.0, 1.0)  # Lx, Ly
    
    # Data loading
    batch_size: int = 10
    num_workers: int = 6
    pde_points: int = 1024
    normalize: bool = True
    
    # Data splits (fraction of total data)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class LossConfig:
    """Configuration for loss functions."""
    # PDE loss weight
    lambda_pde: float = 1.0  # Matches reference implementation
    
    # Loss norms
    regression_norm: str = 'l1'  # 'l1' or 'l2'
    pde_norm: str = 'l1'  # 'l1' or 'l2'
    
    # Physics parameters (will be updated from data)
    Ra: float = 1e5
    Pr: float = 0.7
    Lx: float = 3.0
    Ly: float = 1.0


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and learning rate scheduling."""
    # Optimizer
    optimizer_type: str = 'sgd'  # Matches reference training setup
    learning_rate: float = 1e-2
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Gradient clipping
    grad_clip_value: float = 1.0  # Clip gradient values to this magnitude

    # Learning rate scheduling
    scheduler_type: str = 'plateau'  # 'plateau', 'cosine', 'step'
    patience: int = 10  # For plateau scheduler
    factor: float = 0.1  # Learning rate reduction factor
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    # Training parameters
    num_epochs: int = 100
    clips_per_epoch: int = 3000
    
    # Validation and checkpointing
    val_interval: int = 1  # Validate every N epochs
    checkpoint_interval: int = 10  # Save checkpoint every N epochs
    save_best: bool = True
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    
    # Mixed precision training
    use_amp: bool = False
    
    # Device and distributed training
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    num_gpus: int = 1
    
    # Logging
    log_interval: int = 50  # Log every N batches
    tensorboard: bool = True
    wandb: bool = False
    
    # Output directories
    output_dir: str = './outputs'
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # Metrics
    compute_rrmse: bool = True
    compute_physics_loss: bool = True
    
    # Evaluation parameters
    eval_batch_size: int = 16
    save_predictions: bool = True
    save_visualizations: bool = True
    
    # Out-of-distribution evaluation
    ood_Ra_numbers: List[float] = field(default_factory=lambda: [7e5, 8e5, 9e5, 1.1e6, 1.2e6])
    
    # Visualization
    plot_fields: List[str] = field(default_factory=lambda: ['T', 'u', 'v'])
    plot_timesteps: List[int] = field(default_factory=lambda: [0, 2, 4, 6])
    colormap: str = 'RdBu_r'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Experiment metadata
    experiment_name: str = 'cdanet_rb_experiment'
    description: str = 'CDAnet for Rayleigh-Bénard convection'
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Hyperparameter search
    hyperparameter_search: bool = False
    search_params: Dict[str, List] = field(default_factory=dict)
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    # Update nested config
                    nested_config = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
        return config_dict
    
    def validate(self):
        """Validate configuration parameters."""
        # Validate data splits
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate device
        if self.training.device == 'auto':
            self.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Validate downsample factors
        if self.data.spatial_downsample < 1:
            raise ValueError("Spatial downsample factor must be >= 1")
        if self.data.temporal_downsample < 1:
            raise ValueError("Temporal downsample factor must be >= 1")
        
        # Update loss config with data parameters
        if len(self.data.Ra_numbers) > 0:
            self.loss.Ra = self.data.Ra_numbers[0]
        self.loss.Lx, self.loss.Ly = self.data.domain_size


def create_default_config() -> ExperimentConfig:
    """Create default configuration for CDAnet experiments."""
    return ExperimentConfig()


def create_config_for_ra(Ra: float, spatial_downsample: int = 4, 
                        temporal_downsample: int = 4) -> ExperimentConfig:
    """Create configuration for specific Rayleigh number and downsample factors."""
    config = ExperimentConfig()
    
    # Update data configuration
    config.data.Ra_numbers = [Ra]
    config.data.spatial_downsample = spatial_downsample
    config.data.temporal_downsample = temporal_downsample
    
    # Update loss configuration
    config.loss.Ra = Ra
    
    # Update experiment name
    config.experiment_name = f'cdanet_Ra_{Ra:.0e}_ds_{spatial_downsample}_{temporal_downsample}'
    
    # Validate configuration
    config.validate()
    
    return config


# Predefined configurations for paper experiments
PAPER_CONFIGS = {
    'Ra_1e5_ds_2_2': create_config_for_ra(1e5, 2, 2),
    'Ra_1e5_ds_4_4': create_config_for_ra(1e5, 4, 4),
    'Ra_1e5_ds_8_8': create_config_for_ra(1e5, 8, 8),
    'Ra_1e6_ds_2_2': create_config_for_ra(1e6, 2, 2),
    'Ra_1e6_ds_4_4': create_config_for_ra(1e6, 4, 4),
    'Ra_1e7_ds_2_2': create_config_for_ra(1e7, 2, 2),
}
