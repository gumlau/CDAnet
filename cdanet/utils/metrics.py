"""
Evaluation metrics for CDAnet model performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate various evaluation metrics for CDAnet predictions.
    
    Args:
        variable_names: Names of output variables ['T', 'p', 'u', 'v']
        eps: Small epsilon for numerical stability
    """
    
    def __init__(self, variable_names: List[str] = ['T', 'p', 'u', 'v'], eps: float = 1e-8):
        self.variable_names = variable_names
        self.eps = eps
        
    def compute_rrmse(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute Relative Root Mean Square Error (RRMSE) for each variable.
        
        RRMSE = ||pred - true||_2 / ||true||_2
        
        Args:
            predictions: Predicted fields [B, N, 4] or [B, 4, H, W]
            targets: Target fields with same shape
            
        Returns:
            Dictionary with RRMSE values for each variable
        """
        metrics = {}
        
        # Handle different input shapes
        if predictions.dim() == 4:  # [B, 4, H, W]
            predictions = predictions.permute(0, 2, 3, 1).reshape(-1, predictions.shape[1])
            targets = targets.permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        elif predictions.dim() == 3:  # [B, N, 4]
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
        for i, var_name in enumerate(self.variable_names):
            pred_var = predictions[:, i]
            target_var = targets[:, i]
            
            # Compute RRMSE
            numerator = torch.sqrt(torch.mean((pred_var - target_var) ** 2))
            denominator = torch.sqrt(torch.mean(target_var ** 2)) + self.eps
            rrmse = (numerator / denominator).item()
            
            metrics[f'RRMSE_{var_name}'] = rrmse
            
        # Average RRMSE
        metrics['RRMSE_avg'] = np.mean([metrics[f'RRMSE_{var}'] for var in self.variable_names])
        
        return metrics
    
    def compute_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute Mean Absolute Error for each variable."""
        metrics = {}
        
        if predictions.dim() == 4:
            predictions = predictions.permute(0, 2, 3, 1).reshape(-1, predictions.shape[1])
            targets = targets.permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        elif predictions.dim() == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
        for i, var_name in enumerate(self.variable_names):
            mae = F.l1_loss(predictions[:, i], targets[:, i]).item()
            metrics[f'MAE_{var_name}'] = mae
            
        metrics['MAE_avg'] = np.mean([metrics[f'MAE_{var}'] for var in self.variable_names])
        return metrics
    
    def compute_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute Mean Squared Error for each variable."""
        metrics = {}
        
        if predictions.dim() == 4:
            predictions = predictions.permute(0, 2, 3, 1).reshape(-1, predictions.shape[1])
            targets = targets.permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        elif predictions.dim() == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
        for i, var_name in enumerate(self.variable_names):
            mse = F.mse_loss(predictions[:, i], targets[:, i]).item()
            metrics[f'MSE_{var_name}'] = mse
            
        metrics['MSE_avg'] = np.mean([metrics[f'MSE_{var}'] for var in self.variable_names])
        return metrics
    
    def compute_correlation(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute Pearson correlation coefficient for each variable."""
        metrics = {}
        
        if predictions.dim() == 4:
            predictions = predictions.permute(0, 2, 3, 1).reshape(-1, predictions.shape[1])
            targets = targets.permute(0, 2, 3, 1).reshape(-1, targets.shape[1])
        elif predictions.dim() == 3:
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
        for i, var_name in enumerate(self.variable_names):
            pred_var = predictions[:, i]
            target_var = targets[:, i]
            
            # Compute correlation
            pred_mean = torch.mean(pred_var)
            target_mean = torch.mean(target_var)
            
            numerator = torch.mean((pred_var - pred_mean) * (target_var - target_mean))
            pred_std = torch.sqrt(torch.mean((pred_var - pred_mean) ** 2))
            target_std = torch.sqrt(torch.mean((target_var - target_mean) ** 2))
            
            correlation = numerator / (pred_std * target_std + self.eps)
            metrics[f'Corr_{var_name}'] = correlation.item()
            
        metrics['Corr_avg'] = np.mean([metrics[f'Corr_{var}'] for var in self.variable_names])
        return metrics
    
    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {}
        
        metrics.update(self.compute_rrmse(predictions, targets))
        metrics.update(self.compute_mae(predictions, targets))
        metrics.update(self.compute_mse(predictions, targets))
        metrics.update(self.compute_correlation(predictions, targets))
        
        return metrics
    
    def compute_physics_metrics(self, predictions: torch.Tensor, derivatives: Dict[str, torch.Tensor],
                              Ra: float, Pr: float) -> Dict[str, float]:
        """
        Compute physics-based metrics (PDE residuals).
        
        Args:
            predictions: Predicted fields [B, N, 4] (T, p, u, v)
            derivatives: Dictionary of partial derivatives
            Ra: Rayleigh number
            Pr: Prandtl number
            
        Returns:
            Dictionary with physics residual metrics
        """
        # Extract field variables
        T = predictions[..., 0]
        p = predictions[..., 1] 
        u = predictions[..., 2]
        v = predictions[..., 3]
        
        # Extract derivatives
        dT_dx = derivatives['dT_dx']
        dT_dy = derivatives['dT_dy']
        dT_dt = derivatives['dT_dt']
        d2T_dx2 = derivatives['d2T_dx2']
        d2T_dy2 = derivatives['d2T_dy2']
        
        dp_dx = derivatives['dp_dx']
        dp_dy = derivatives['dp_dy']
        
        du_dx = derivatives['du_dx']
        du_dy = derivatives['du_dy'] 
        du_dt = derivatives['du_dt']
        d2u_dx2 = derivatives['d2u_dx2']
        d2u_dy2 = derivatives['d2u_dy2']
        
        dv_dx = derivatives['dv_dx']
        dv_dy = derivatives['dv_dy']
        dv_dt = derivatives['dv_dt']
        d2v_dx2 = derivatives['d2v_dx2']
        d2v_dy2 = derivatives['d2v_dy2']
        
        # Compute residuals
        continuity_residual = du_dx + dv_dy
        momentum_x_residual = (du_dt + u * du_dx + v * du_dy + dp_dx 
                              - Pr * (d2u_dx2 + d2u_dy2))
        momentum_y_residual = (dv_dt + u * dv_dx + v * dv_dy + dp_dy 
                              - Pr * (d2v_dx2 + d2v_dy2) - Ra * Pr * T)
        energy_residual = dT_dt + u * dT_dx + v * dT_dy - (d2T_dx2 + d2T_dy2)
        
        # Compute residual norms
        metrics = {
            'Physics_Continuity_L1': torch.mean(torch.abs(continuity_residual)).item(),
            'Physics_Momentum_X_L1': torch.mean(torch.abs(momentum_x_residual)).item(),
            'Physics_Momentum_Y_L1': torch.mean(torch.abs(momentum_y_residual)).item(),
            'Physics_Energy_L1': torch.mean(torch.abs(energy_residual)).item(),
        }
        
        # Total physics loss
        metrics['Physics_Total_L1'] = sum(metrics.values())
        
        # L2 norms
        metrics.update({
            'Physics_Continuity_L2': torch.sqrt(torch.mean(continuity_residual ** 2)).item(),
            'Physics_Momentum_X_L2': torch.sqrt(torch.mean(momentum_x_residual ** 2)).item(),
            'Physics_Momentum_Y_L2': torch.sqrt(torch.mean(momentum_y_residual ** 2)).item(),
            'Physics_Energy_L2': torch.sqrt(torch.mean(energy_residual ** 2)).item(),
        })
        
        metrics['Physics_Total_L2'] = sum([
            metrics['Physics_Continuity_L2'],
            metrics['Physics_Momentum_X_L2'],
            metrics['Physics_Momentum_Y_L2'],
            metrics['Physics_Energy_L2']
        ])
        
        return metrics


class VisualizationUtils:
    """Utilities for visualizing predictions and metrics."""
    
    def __init__(self, variable_names: List[str] = ['T', 'p', 'u', 'v']):
        self.variable_names = variable_names
        
    def plot_field_comparison(self, predictions: np.ndarray, targets: np.ndarray,
                            timestep: int = 0, save_path: Optional[str] = None):
        """
        Plot side-by-side comparison of predicted and target fields.
        
        Args:
            predictions: Predicted fields [T, H, W, 4]
            targets: Target fields [T, H, W, 4]
            timestep: Timestep to visualize
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for i, var_name in enumerate(self.variable_names):
            pred_field = predictions[timestep, :, :, i]
            target_field = targets[timestep, :, :, i]
            error_field = pred_field - target_field
            
            # Prediction
            im1 = axes[0, i].imshow(pred_field, cmap='RdBu_r', aspect='auto')
            axes[0, i].set_title(f'Predicted {var_name}')
            plt.colorbar(im1, ax=axes[0, i])
            
            # Target
            im2 = axes[1, i].imshow(target_field, cmap='RdBu_r', aspect='auto')
            axes[1, i].set_title(f'Target {var_name}')
            plt.colorbar(im2, ax=axes[1, i])
            
            # Error
            im3 = axes[2, i].imshow(error_field, cmap='RdBu_r', aspect='auto')
            axes[2, i].set_title(f'Error {var_name}')
            plt.colorbar(im3, ax=axes[2, i])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def plot_metrics_evolution(self, metrics_history: Dict[str, List[float]], 
                             save_path: Optional[str] = None):
        """Plot evolution of metrics during training."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metric_groups = {
            'RRMSE': [k for k in metrics_history.keys() if 'RRMSE' in k],
            'MAE': [k for k in metrics_history.keys() if 'MAE' in k],
            'Physics': [k for k in metrics_history.keys() if 'Physics' in k and 'Total' in k],
            'Loss': [k for k in metrics_history.keys() if 'loss' in k.lower()]
        }
        
        for idx, (group_name, metric_names) in enumerate(metric_groups.items()):
            if idx >= len(axes):
                break
                
            for metric_name in metric_names:
                if metric_name in metrics_history:
                    axes[idx].plot(metrics_history[metric_name], label=metric_name)
                    
            axes[idx].set_title(f'{group_name} Metrics')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Value')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def plot_correlation_matrix(self, predictions: np.ndarray, targets: np.ndarray,
                              save_path: Optional[str] = None):
        """Plot correlation matrix between predicted and target variables."""
        # Flatten arrays
        pred_flat = predictions.reshape(-1, predictions.shape[-1])
        target_flat = targets.reshape(-1, targets.shape[-1])
        
        # Compute correlation matrix
        data = np.concatenate([pred_flat, target_flat], axis=1)
        var_names = [f'Pred_{v}' for v in self.variable_names] + [f'Target_{v}' for v in self.variable_names]
        
        corr_matrix = np.corrcoef(data.T)
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   xticklabels=var_names, yticklabels=var_names)
        plt.title('Correlation Matrix: Predictions vs Targets')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()