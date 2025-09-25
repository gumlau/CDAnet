"""
Inception-ResNet blocks for 3D U-Net feature extraction.
Based on the architecture described in the CDAnet paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionResNetBlock3D(nn.Module):
    """
    3D Inception-ResNet block with three convolution branches.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_sizes: List of kernel sizes for the three branches [(1,1,1), (5,5,3), (9,9,5)]
    """
    
    def __init__(self, in_channels, out_channels, kernel_sizes=[(1,1,1), (5,5,3), (9,9,5)]):
        super(InceptionResNetBlock3D, self).__init__()
        
        # Three convolution branches - ensure total output matches out_channels
        branch_channels = out_channels // 3
        remaining_channels = out_channels - 2 * branch_channels
        
        self.branch1 = self._make_branch(in_channels, branch_channels, kernel_sizes[0])
        self.branch2 = self._make_branch(in_channels, branch_channels, kernel_sizes[1])
        self.branch3 = self._make_branch(in_channels, remaining_channels, kernel_sizes[2])
        
        # Residual connection - adjust channels if needed
        self.residual = nn.Identity() if in_channels == out_channels else \
                      nn.Conv3d(in_channels, out_channels, 1, bias=False)
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def _make_branch(self, in_channels, out_channels, kernel_size):
        """Create a convolution branch with zero padding to maintain spatial size."""
        padding = tuple(k//2 for k in kernel_size)  # Zero padding
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Apply three branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)  
        branch3_out = self.branch3(x)
        
        # Concatenate branch outputs
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        
        # Residual connection
        residual = self.residual(x)
        
        # Add residual and apply final activation
        out = self.relu(self.bn(concat_out + residual))
        
        return out


class DoubleInceptionResNet3D(nn.Module):
    """Double Inception-ResNet block for encoder/decoder stages."""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleInceptionResNet3D, self).__init__()
        
        self.block1 = InceptionResNetBlock3D(in_channels, out_channels)
        self.block2 = InceptionResNetBlock3D(out_channels, out_channels)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x