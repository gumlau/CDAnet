"""
Evaluation utilities for CDAnet model performance assessment.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

DATASET_TO_MODEL = torch.tensor([1, 0, 2, 3], dtype=torch.long)
MODEL_TO_DATASET = torch.tensor([1, 0, 2, 3], dtype=torch.long)

from ..models.cdanet import CDAnet
from ..data.data_loader import RBDataModule
from ..utils.logger import Logger
from ..utils.metrics import MetricsCalculator, VisualizationUtils
from ..config.config import ExperimentConfig, EvaluationConfig


class CDAnetEvaluator:
    """
    Evaluator for CDAnet model performance.
    
    Args:
        model: Trained CDAnet model
        data_module: Data module for evaluation datasets
        config: Evaluation configuration
        logger: Logger for tracking evaluation
    """
    
    def __init__(self, model: CDAnet, data_module: RBDataModule, 
                 config: EvaluationConfig, logger: Optional[Logger] = None):
        
        self.model = model
        self.data_module = data_module
        self.config = config
        self.logger = logger
        
        self.device = next(model.parameters()).device
        
        # Metrics and visualization utilities
        self.metrics_calc = MetricsCalculator()
        self.viz_utils = VisualizationUtils()
        
        # Results storage
        self.evaluation_results = defaultdict(dict)
        
    def evaluate_on_dataset(self, Ra: float, split: str = 'test') -> Dict[str, float]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            Ra: Rayleigh number
            split: Dataset split ('test', 'val', 'train')
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.logger:
            self.logger.info(f"Evaluating on Ra={Ra:.0e}, split={split}")
            
        self.model.eval()
        
        # Get data loader
        try:
            dataloader = self.data_module.get_dataloader(Ra, split)
        except KeyError:
            if self.logger:
                self.logger.warning(f"Dataset not found for Ra={Ra:.0e}, split={split}")
            return {}
        
        all_predictions = []
        all_targets = []
        all_physics_metrics = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                predictions = self._predict_batch(batch)
                targets = batch['targets']
                
                # Store results
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                
                # Compute physics metrics if requested
                if self.config.compute_physics_loss and batch_idx < 5:  # Only for subset
                    physics_metrics = self._compute_physics_metrics(batch, predictions)
                    all_physics_metrics.append(physics_metrics)
                    
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self._compute_comprehensive_metrics(predictions, targets, all_physics_metrics)
        
        # Store results
        key = f"Ra_{Ra:.0e}_{split}"
        self.evaluation_results[key] = {
            'metrics': metrics,
            'predictions': predictions.numpy(),
            'targets': targets.numpy()
        }
        
        if self.logger:
            self.logger.info(f"Evaluation completed for {key}")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.logger.info(f"  {metric_name}: {metric_value:.6f}")
                    
        return metrics
    
    def evaluate_generalization(self, train_Ra: float, test_Ra_list: List[float]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model generalization across different Rayleigh numbers.
        
        Args:
            train_Ra: Rayleigh number used for training
            test_Ra_list: List of Rayleigh numbers for testing
            
        Returns:
            Dictionary of evaluation results for each test Ra
        """
        if self.logger:
            self.logger.info(f"Evaluating generalization from Ra={train_Ra:.0e} to {test_Ra_list}")
            
        generalization_results = {}
        
        for test_Ra in test_Ra_list:
            try:
                metrics = self.evaluate_on_dataset(test_Ra, 'test')
                generalization_results[f"Ra_{test_Ra:.0e}"] = metrics
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to evaluate on Ra={test_Ra:.0e}: {e}")
                    
        return generalization_results
    
    def evaluate_temporal_evolution(self, Ra: float, split: str = 'test', 
                                  max_batches: int = 5) -> Dict[str, np.ndarray]:
        """
        Evaluate temporal evolution of predictions.
        
        Args:
            Ra: Rayleigh number
            split: Dataset split
            max_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary with temporal evolution metrics
        """
        if self.logger:
            self.logger.info(f"Evaluating temporal evolution for Ra={Ra:.0e}")
            
        self.model.eval()
        dataloader = self.data_module.get_dataloader(Ra, split)
        
        temporal_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                batch = self._move_batch_to_device(batch)
                
                # Get predictions for full clip
                predictions = self._predict_batch(batch)  # [B, N, 4]
                targets = batch['targets']  # [B, N, 4]
                
                # Reshape to [B, T, H, W, 4] for temporal analysis
                B = predictions.shape[0]
                T = 8  # clip_length
                H = int(np.sqrt(predictions.shape[1] // T))
                W = H
                
                pred_reshaped = predictions.view(B, H, W, T, 4).permute(0, 3, 1, 2, 4)  # [B, T, H, W, 4]
                target_reshaped = targets.view(B, H, W, T, 4).permute(0, 3, 1, 2, 4)
                
                # Compute metrics for each timestep
                for t in range(T):
                    pred_t = pred_reshaped[:, t]  # [B, H, W, 4]
                    target_t = target_reshaped[:, t]
                    
                    metrics_t = self.metrics_calc.compute_all_metrics(pred_t, target_t)
                    
                    for key, value in metrics_t.items():
                        temporal_metrics[f"t_{t}_{key}"].append(value)
        
        # Average over batches
        avg_temporal_metrics = {}
        for key, values in temporal_metrics.items():
            avg_temporal_metrics[key] = np.mean(values)
            
        return avg_temporal_metrics
    
    def create_evaluation_report(self, output_dir: str):
        """Create comprehensive evaluation report."""
        if self.logger:
            self.logger.info("Creating evaluation report...")
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary report
        self._create_summary_report(output_dir)
        
        # Create visualizations
        if self.config.save_visualizations:
            self._create_visualizations(output_dir)
            
        # Save raw results
        self._save_raw_results(output_dir)
        
        if self.logger:
            self.logger.info(f"Evaluation report saved to {output_dir}")
    
    def _predict_batch(self, batch: Dict) -> torch.Tensor:
        """Make predictions for a batch."""
        low_res = batch['low_res']
        coords = batch['coords']

        reorder_in = DATASET_TO_MODEL.to(low_res.device)
        reorder_out = MODEL_TO_DATASET.to(low_res.device)

        low_res_reordered = low_res.index_select(1, reorder_in)
        predictions = self.model(low_res_reordered, coords)
        predictions = predictions.index_select(-1, reorder_out)
        return predictions
    
    def _compute_physics_metrics(self, batch: Dict, predictions: torch.Tensor) -> Dict[str, float]:
        """Compute physics-based metrics for a batch."""
        # Use subset of points for efficiency
        pde_coords = batch.get('pde_coords', batch['coords'][:, :1000])  # Use first 1000 points
        
        # Get derivatives
        _, derivatives = self.model.forward_with_derivatives(batch['low_res'], pde_coords)
        
        # Compute physics metrics
        Ra = batch.get('Ra', 1e5)
        Pr = batch.get('Pr', 0.7)
        
        physics_metrics = self.metrics_calc.compute_physics_metrics(
            predictions[:, :pde_coords.shape[1]], derivatives, Ra, Pr
        )
        
        return physics_metrics
    
    def _compute_comprehensive_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                                     physics_metrics_list: List[Dict]) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Standard metrics
        if self.config.compute_rrmse:
            metrics.update(self.metrics_calc.compute_all_metrics(predictions, targets))
        
        # Physics metrics (averaged)
        if self.config.compute_physics_loss and physics_metrics_list:
            physics_metrics_avg = defaultdict(list)
            for pm in physics_metrics_list:
                for key, value in pm.items():
                    physics_metrics_avg[key].append(value)
                    
            for key, values in physics_metrics_avg.items():
                metrics[f"avg_{key}"] = np.mean(values)
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _create_summary_report(self, output_dir: str):
        """Create summary report text file."""
        report_path = os.path.join(output_dir, 'evaluation_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("CDAnet Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset_key, results in self.evaluation_results.items():
                f.write(f"Dataset: {dataset_key}\n")
                f.write("-" * 30 + "\n")
                
                metrics = results['metrics']
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        f.write(f"{metric_name:25s}: {metric_value:.6f}\n")
                        
                f.write("\n")
    
    def _create_visualizations(self, output_dir: str):
        """Create visualization plots."""
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        for dataset_key, results in self.evaluation_results.items():
            predictions = results['predictions']
            targets = results['targets']
            
            # Reshape for visualization
            # Assume predictions shape: [N_samples, 4] where N_samples = B*H*W*T
            # Need to reshape to [T, H, W, 4] for a single sample
            sample_size = predictions.shape[0] // 8  # Assume 8 timesteps
            H = W = int(np.sqrt(sample_size))
            
            pred_reshaped = predictions[:sample_size*8].reshape(8, H, W, 4)
            target_reshaped = targets[:sample_size*8].reshape(8, H, W, 4)
            
            # Create field comparison plots
            for t in self.config.plot_timesteps:
                if t < pred_reshaped.shape[0]:
                    plot_path = os.path.join(viz_dir, f'{dataset_key}_fields_t{t}.png')
                    self.viz_utils.plot_field_comparison(
                        pred_reshaped, target_reshaped, timestep=t, save_path=plot_path
                    )
                    
            # Create correlation matrix
            corr_path = os.path.join(viz_dir, f'{dataset_key}_correlation.png')
            self.viz_utils.plot_correlation_matrix(
                predictions[:10000], targets[:10000], save_path=corr_path
            )
    
    def _save_raw_results(self, output_dir: str):
        """Save raw evaluation results."""
        results_path = os.path.join(output_dir, 'raw_results.npz')
        
        save_dict = {}
        for dataset_key, results in self.evaluation_results.items():
            save_dict[f'{dataset_key}_predictions'] = results['predictions']
            save_dict[f'{dataset_key}_targets'] = results['targets']
            save_dict[f'{dataset_key}_metrics'] = results['metrics']
            
        np.savez_compressed(results_path, **save_dict)


def evaluate_model_from_checkpoint(checkpoint_path: str, config: ExperimentConfig,
                                 data_module: RBDataModule, output_dir: str) -> CDAnetEvaluator:
    """
    Evaluate a model from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Experiment configuration
        data_module: Data module for evaluation
        output_dir: Output directory for results
        
    Returns:
        Evaluator instance with results
    """
    # Load model
    model = CDAnet(
        in_channels=config.model.in_channels,
        feature_channels=config.model.feature_channels,
        mlp_hidden_dims=config.model.mlp_hidden_dims,
        activation=config.model.activation,
        coord_dim=config.model.coord_dim,
        output_dim=config.model.output_dim,
        igres=getattr(config.model, 'igres', (4, 64, 64)),
        unet_nf=getattr(config.model, 'unet_nf', 16),
        unet_mf=getattr(config.model, 'unet_mf', 256),
        imnet_nf=getattr(config.model, 'imnet_nf', None)
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = CDAnetEvaluator(model, data_module, config.evaluation)
    
    # Run evaluation
    for Ra in config.data.Ra_numbers:
        evaluator.evaluate_on_dataset(Ra, 'test')
        
    # Create report
    evaluator.create_evaluation_report(output_dir)
    
    return evaluator
