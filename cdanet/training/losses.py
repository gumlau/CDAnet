"""
Loss functions for CDAnet training including regression and PDE losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CDAnetLoss(nn.Module):
    """
    Combined loss function for CDAnet: L_total = L_reg + λ * L_PDE
    
    Args:
        lambda_pde: Weight for PDE loss term
        Ra: Rayleigh number
        Pr: Prandtl number  
        Lx, Ly: Domain dimensions
        regression_norm: Norm for regression loss ('l1' or 'l2')
        pde_norm: Norm for PDE loss ('l1' or 'l2')
    """
    
    def __init__(self, lambda_pde=0.01, Ra=1e5, Pr=0.7, Lx=3.0, Ly=1.0, 
                 regression_norm='l1', pde_norm='l1'):
        super(CDAnetLoss, self).__init__()
        
        self.lambda_pde = lambda_pde
        self.Ra = Ra
        self.Pr = Pr
        self.Lx = Lx
        self.Ly = Ly
        self.regression_norm = regression_norm
        self.pde_norm = pde_norm
        
        # Loss functions
        if regression_norm == 'l1':
            self.regression_loss = nn.L1Loss()
        elif regression_norm == 'l2':
            self.regression_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported regression norm: {regression_norm}")
    
    def forward(self, predictions, targets, derivatives=None, coords=None):
        """
        Compute combined loss.
        
        Args:
            predictions: Predicted fields [B, N, 4] (T, p, u, v)
            targets: Target fields [B, N, 4]
            derivatives: Dictionary of partial derivatives (optional)
            coords: Coordinates [B, N, 3] (optional)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Check for NaN/Inf in inputs
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: NaN/Inf in predictions")
            predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)

        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: NaN/Inf in targets")
            targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)

        # Regression loss
        L_reg = self.regression_loss(predictions, targets)

        # PDE loss (if derivatives are provided)
        if derivatives is not None:
            L_pde = self.compute_pde_loss(predictions, derivatives)
        else:
            L_pde = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = L_reg + self.lambda_pde * L_pde

        # Only check for invalid values without forcing replacement
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss detected - L_reg: {L_reg.item():.6f}, L_pde: {L_pde.item():.6f}")
            # Return zero gradients instead of a fixed value
            total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Return loss components
        loss_dict = {
            'total_loss': total_loss.item(),
            'regression_loss': L_reg.item(),
            'pde_loss': L_pde.item(),
            'lambda_pde': self.lambda_pde
        }
        
        return total_loss, loss_dict
    
    def compute_pde_loss(self, predictions, derivatives):
        """
        Compute PDE residual loss for Rayleigh-Bénard equations.
        
        Governing equations:
        1. Continuity: ∂u/∂x + ∂v/∂y = 0
        2. Momentum (u): ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + Pr(∂²u/∂x² + ∂²u/∂y²)
        3. Momentum (v): ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + Pr(∂²v/∂x² + ∂²v/∂y²) + Ra*Pr*T
        4. Energy: ∂T/∂t + u∂T/∂x + v∂T/∂y = ∂²T/∂x² + ∂²T/∂y²
        """
        
        # Extract field variables
        T = predictions[..., 0]  # Temperature
        p = predictions[..., 1]  # Pressure  
        u = predictions[..., 2]  # u-velocity
        v = predictions[..., 3]  # v-velocity
        
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
        
        # PDE residuals
        # 1. Continuity equation: ∂u/∂x + ∂v/∂y = 0
        continuity_residual = du_dx + dv_dy
        
        # 2. Momentum equation (x): ∂u/∂t + u∂u/∂x + v∂u/∂y + ∂p/∂x - Pr(∂²u/∂x² + ∂²u/∂y²) = 0
        momentum_x_residual = (du_dt + u * du_dx + v * du_dy + dp_dx 
                              - self.Pr * (d2u_dx2 + d2u_dy2))
        
        # 3. Momentum equation (y): ∂v/∂t + u∂v/∂x + v∂v/∂y + ∂p/∂y - Pr(∂²v/∂x² + ∂²v/∂y²) - Ra*Pr*T = 0
        momentum_y_residual = (dv_dt + u * dv_dx + v * dv_dy + dp_dy 
                              - self.Pr * (d2v_dx2 + d2v_dy2) - self.Ra * self.Pr * T)
        
        # 4. Energy equation: ∂T/∂t + u∂T/∂x + v∂T/∂y - (∂²T/∂x² + ∂²T/∂y²) = 0
        energy_residual = dT_dt + u * dT_dx + v * dT_dy - (d2T_dx2 + d2T_dy2)
        
        # Compute norm of residuals
        if self.pde_norm == 'l1':
            pde_loss = (torch.mean(torch.abs(continuity_residual)) +
                       torch.mean(torch.abs(momentum_x_residual)) +
                       torch.mean(torch.abs(momentum_y_residual)) +
                       torch.mean(torch.abs(energy_residual)))
        elif self.pde_norm == 'l2':
            pde_loss = (torch.mean(continuity_residual**2) +
                       torch.mean(momentum_x_residual**2) +
                       torch.mean(momentum_y_residual**2) +
                       torch.mean(energy_residual**2))
        else:
            raise ValueError(f"Unsupported PDE norm: {self.pde_norm}")
            
        return pde_loss


class RRMSELoss(nn.Module):
    """Relative Root Mean Square Error for evaluation."""
    
    def __init__(self, eps=1e-8):
        super(RRMSELoss, self).__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        Compute RRMSE for each variable separately.
        
        Args:
            predictions: Predicted fields [B, N, 4]
            targets: Target fields [B, N, 4]
            
        Returns:
            rrmse_dict: Dictionary with RRMSE for each variable
        """
        rrmse_dict = {}
        variable_names = ['T', 'p', 'u', 'v']
        
        for i, var_name in enumerate(variable_names):
            pred_var = predictions[..., i]
            target_var = targets[..., i]
            
            numerator = torch.sqrt(torch.mean((pred_var - target_var)**2))
            denominator = torch.sqrt(torch.mean(target_var**2)) + self.eps
            rrmse = numerator / denominator
            
            rrmse_dict[f'RRMSE_{var_name}'] = rrmse.item()
        
        # Average RRMSE
        rrmse_dict['RRMSE_avg'] = np.mean(list(rrmse_dict.values()))
        
        return rrmse_dict