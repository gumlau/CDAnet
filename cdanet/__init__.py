"""
CDAnet: Continuous Data Assimilation Network for Rayleigh-Bénard Convection
"""

__version__ = "1.0.0"

from .models import CDAnet
from .config import ExperimentConfig, create_default_config
from .data import RBDataModule
from .training import CDAnetTrainer
from .evaluation import CDAnetEvaluator

__all__ = [
    'CDAnet',
    'ExperimentConfig', 
    'create_default_config',
    'RBDataModule',
    'CDAnetTrainer',
    'CDAnetEvaluator'
]