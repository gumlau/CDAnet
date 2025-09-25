"""
Rayleigh-Bénard convection visualization utilities.
Creates publication-quality plots similar to the original CDAnet paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
from typing import Dict, List, Optional, Tuple
import os


class RBVisualization:
    """
    Visualization utilities for Rayleigh-Bénard convection fields.
    Creates comparison plots between Input, Truth, CDAnet predictions, and CDA results.
    """
    
    def __init__(self, domain_size: Tuple[float, float] = (3.0, 1.0)):
        self.Lx, self.Ly = domain_size
        
        # Create custom colormap similar to the paper
        self.setup_colormap()
        
    def setup_colormap(self):
        """Setup custom colormap for temperature fields."""
        # Create blue-yellow-red colormap similar to the paper
        colors = ['#0066CC', '#00AAFF', '#66DDFF', '#CCFFFF', '#FFFFFF', 
                 '#FFFFCC', '#FFDD66', '#FFAA00', '#FF6600', '#CC0000']
        self.cmap = LinearSegmentedColormap.from_list('rb_temp', colors, N=256)
        
    def create_comparison_plot(self, 
                             input_field: np.ndarray,
                             truth_field: np.ndarray, 
                             cdanet_pred: np.ndarray,
                             cda_pred: Optional[np.ndarray] = None,
                             times: List[float] = [15.0, 18.2, 21.5],
                             variable: str = 'T',
                             vmin: float = -0.5,
                             vmax: float = 0.5,
                             save_path: Optional[str] = None,
                             figsize: Tuple[float, float] = (12, 10)) -> plt.Figure:
        """
        Create publication-style comparison plot.
        
        Args:
            input_field: Input low-resolution field [T, H, W]
            truth_field: Ground truth high-resolution field [T, H, W]
            cdanet_pred: CDAnet prediction [T, H, W]
            cda_pred: CDA prediction [T, H, W] (optional)
            times: List of time values to display
            variable: Variable name ('T', 'u', 'v', 'p')
            vmin, vmax: Color scale limits
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        n_times = len(times)
        n_cols = 4 if cda_pred is not None else 3
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_times, n_cols, figure=fig, 
                              hspace=0.3, wspace=0.1)
        
        # Column titles
        col_titles = ['Input', 'Truth', 'CDAnet']
        if cda_pred is not None:
            col_titles.append('CDA')
            
        for row, time_val in enumerate(times):
            # Find closest time index
            time_idx = min(range(len(input_field)), 
                          key=lambda i: abs(i * 0.1 - time_val))  # Assuming dt=0.1
            
            fields = [input_field[time_idx], truth_field[time_idx], cdanet_pred[time_idx]]
            if cda_pred is not None:
                fields.append(cda_pred[time_idx])
                
            for col, (field, title) in enumerate(zip(fields, col_titles)):
                ax = fig.add_subplot(gs[row, col])
                
                # Create field plot
                im = ax.imshow(field, 
                             cmap=self.cmap, 
                             vmin=vmin, vmax=vmax,
                             extent=[0, self.Lx, 0, self.Ly],
                             origin='lower',
                             aspect='auto')
                
                # Formatting
                ax.set_xlim(0, self.Lx)
                ax.set_ylim(0, self.Ly)
                
                # Add title for first row
                if row == 0:
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    
                # Add time label for first column
                if col == 0:
                    ax.text(-0.3, 0.5, f'Time = {time_val}', 
                           transform=ax.transAxes, fontsize=12, 
                           rotation=90, ha='center', va='center',
                           fontweight='bold')
                
                # Remove ticks for cleaner look
                ax.set_xticks([0, 1, 2, 3])
                ax.set_yticks([0, 0.5, 1])
                
                # Only show labels on edges
                if row == n_times - 1:
                    ax.set_xlabel('x', fontsize=10)
                else:
                    ax.set_xticklabels([])
                    
                if col == 0:
                    ax.set_ylabel('y', fontsize=10)
                else:
                    ax.set_yticklabels([])
                
                # Add colorbar for last column
                if col == len(fields) - 1:
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.1)
                    cbar.ax.tick_params(labelsize=9)
                    if variable == 'T':
                        cbar.set_label('Temperature', fontsize=10)
                    elif variable == 'u':
                        cbar.set_label('u-velocity', fontsize=10)
                    elif variable == 'v':
                        cbar.set_label('v-velocity', fontsize=10)
                    elif variable == 'p':
                        cbar.set_label('Pressure', fontsize=10)
        
        # Add main title
        fig.suptitle(f'Rayleigh-Bénard Convection: {variable} Field Comparison', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
        
        return fig
    
    def create_temporal_evolution(self,
                                truth_field: np.ndarray,
                                cdanet_pred: np.ndarray,
                                variable: str = 'T',
                                n_snapshots: int = 8,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create temporal evolution visualization.
        
        Args:
            truth_field: Ground truth field [T, H, W]
            cdanet_pred: CDAnet predictions [T, H, W]
            variable: Variable name
            n_snapshots: Number of time snapshots to show
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, n_snapshots, figsize=(16, 6))
        
        time_indices = np.linspace(0, len(truth_field)-1, n_snapshots, dtype=int)
        
        for i, t_idx in enumerate(time_indices):
            # Truth
            im1 = axes[0, i].imshow(truth_field[t_idx], cmap=self.cmap, 
                                   vmin=-0.5, vmax=0.5, origin='lower')
            axes[0, i].set_title(f't = {t_idx * 0.1:.1f}')
            axes[0, i].axis('off')
            
            # Prediction
            im2 = axes[1, i].imshow(cdanet_pred[t_idx], cmap=self.cmap,
                                   vmin=-0.5, vmax=0.5, origin='lower')
            axes[1, i].axis('off')
            
            # Error
            # error = cdanet_pred[t_idx] - truth_field[t_idx]
            # axes[2, i].imshow(error, cmap='RdBu_r', vmin=-0.1, vmax=0.1, origin='lower')
            # axes[2, i].axis('off')
        
        # Row labels
        axes[0, 0].set_ylabel('Truth', fontsize=12, rotation=90)
        axes[1, 0].set_ylabel('CDAnet', fontsize=12, rotation=90)
        # axes[2, 0].set_ylabel('Error', fontsize=12, rotation=90)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_error_analysis(self,
                            truth_field: np.ndarray,
                            cdanet_pred: np.ndarray,
                            cda_pred: Optional[np.ndarray] = None,
                            time_idx: int = 50,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create detailed error analysis plot.
        
        Args:
            truth_field: Ground truth [T, H, W]
            cdanet_pred: CDAnet predictions [T, H, W]
            cda_pred: CDA predictions [T, H, W] (optional)
            time_idx: Time index to analyze
            save_path: Save path
            
        Returns:
            matplotlib Figure
        """
        n_cols = 3 if cda_pred is not None else 2
        fig, axes = plt.subplots(2, n_cols, figsize=(12, 8))
        
        # Fields
        axes[0, 0].imshow(truth_field[time_idx], cmap=self.cmap, vmin=-0.5, vmax=0.5)
        axes[0, 0].set_title('Ground Truth')
        
        axes[0, 1].imshow(cdanet_pred[time_idx], cmap=self.cmap, vmin=-0.5, vmax=0.5)
        axes[0, 1].set_title('CDAnet Prediction')
        
        if cda_pred is not None:
            axes[0, 2].imshow(cda_pred[time_idx], cmap=self.cmap, vmin=-0.5, vmax=0.5)
            axes[0, 2].set_title('CDA Prediction')
        
        # Errors
        cdanet_error = cdanet_pred[time_idx] - truth_field[time_idx]
        im1 = axes[1, 0].imshow(cdanet_error, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        axes[1, 0].set_title('CDAnet Error')
        plt.colorbar(im1, ax=axes[1, 0])
        
        if cda_pred is not None:
            cda_error = cda_pred[time_idx] - truth_field[time_idx]
            im2 = axes[1, 1].imshow(cda_error, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            axes[1, 1].set_title('CDA Error')
            plt.colorbar(im2, ax=axes[1, 1])
            
            # Error difference
            error_diff = np.abs(cdanet_error) - np.abs(cda_error)
            im3 = axes[1, 2].imshow(error_diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
            axes[1, 2].set_title('Error Difference\n(|CDAnet| - |CDA|)')
            plt.colorbar(im3, ax=axes[1, 2])
        else:
            # Remove unused subplot
            fig.delaxes(axes[1, 1])
            
        for ax in axes.flat:
            if ax.get_xlabel() != '':  # Check if axis exists
                ax.axis('off')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def visualize_training_results(model_output_dir: str, 
                             data_path: str,
                             checkpoint_path: str,
                             output_dir: str = None):
    """
    Create comprehensive visualization from training results.
    
    Args:
        model_output_dir: Directory containing model outputs
        data_path: Path to test data
        checkpoint_path: Path to trained model checkpoint
        output_dir: Output directory for visualizations
    """
    if output_dir is None:
        output_dir = os.path.join(model_output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and model predictions
    # This would load your actual results
    # For now, create dummy data matching the paper format
    
    visualizer = RBVisualization()
    
    # Create dummy data for demonstration
    T, H, W = 100, 64, 192  # Time steps, Height, Width
    
    # Generate sample convection patterns
    x = np.linspace(0, 3, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Truth field with convection cells
    truth_field = np.zeros((T, H, W))
    input_field = np.zeros((T, H//4, W//4))  # Low resolution input
    cdanet_pred = np.zeros((T, H, W))
    
    for t in range(T):
        time = t * 0.1
        # Create convection pattern
        temp = 0.5 * np.sin(2*np.pi*X/1.5) * np.sin(np.pi*Y) * np.cos(0.1*time)
        temp += 0.3 * np.sin(4*np.pi*X/1.5) * np.sin(2*np.pi*Y) * np.sin(0.05*time)
        temp += np.random.normal(0, 0.05, (H, W))  # Add noise
        
        truth_field[t] = temp
        
        # Downsample for input
        input_field[t] = temp[::4, ::4]
        
        # Add slight error to CDAnet prediction
        cdanet_pred[t] = temp + np.random.normal(0, 0.02, (H, W))
    
    # Create comparison plot like the paper
    fig1 = visualizer.create_comparison_plot(
        input_field=input_field,
        truth_field=truth_field,
        cdanet_pred=cdanet_pred,
        times=[15.0, 18.2, 21.5],
        variable='T',
        save_path=os.path.join(output_dir, 'rb_comparison.png')
    )
    
    # Create temporal evolution
    fig2 = visualizer.create_temporal_evolution(
        truth_field=truth_field,
        cdanet_pred=cdanet_pred,
        save_path=os.path.join(output_dir, 'temporal_evolution.png')
    )
    
    # Create error analysis
    fig3 = visualizer.create_error_analysis(
        truth_field=truth_field,
        cdanet_pred=cdanet_pred,
        save_path=os.path.join(output_dir, 'error_analysis.png')
    )
    
    plt.close('all')  # Clean up
    
    print(f"Visualizations saved to {output_dir}")
    print("Generated:")
    print("  - rb_comparison.png (Paper-style comparison)")
    print("  - temporal_evolution.png (Time evolution)")  
    print("  - error_analysis.png (Error analysis)")


if __name__ == "__main__":
    # Demo visualization
    visualizer = RBVisualization()
    visualize_training_results('./', './data', './checkpoint.pth', './demo_viz')