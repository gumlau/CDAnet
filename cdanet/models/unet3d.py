"""
3D U-Net feature extractor for spatio-temporal feature extraction.
Modified for 4-channel input (T, p, u, v) and incorporating Inception-ResNet blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .inception_resnet import DoubleInceptionResNet3D


class UNet3D(nn.Module):
    """
    3D U-Net with Inception-ResNet blocks for feature extraction.
    
    Args:
        in_channels: Number of input channels (4 for T, p, u, v)
        feature_channels: Number of output feature channels
        base_channels: Base number of channels for the first layer
    """
    
    def __init__(self, in_channels=4, feature_channels=256, base_channels=32):
        super(UNet3D, self).__init__()
        
        self.in_channels = in_channels
        self.feature_channels = feature_channels
        
        # Encoder path with Inception-ResNet blocks
        self.inc = DoubleInceptionResNet3D(in_channels, base_channels)
        self.down1 = self._make_encoder_block(base_channels, base_channels*2)
        self.down2 = self._make_encoder_block(base_channels*2, base_channels*4)
        self.down3 = self._make_encoder_block(base_channels*4, base_channels*8)
        self.down4 = self._make_encoder_block(base_channels*8, base_channels*16)
        
        # Bottleneck (modified to match x3 output)
        self.bottleneck = DoubleInceptionResNet3D(base_channels*4, base_channels*8)
        
        # Decoder path with upsampling and skip connections (adjusted for modified architecture)
        self.up1 = self._make_decoder_block(base_channels*16, base_channels*8)  # Not used in simplified version
        self.up2 = self._make_decoder_block(base_channels*8 + base_channels*2, base_channels*4)  # 512+128=640 -> 256
        self.up3 = self._make_decoder_block(base_channels*4, base_channels*2)  # Not used in simplified version  
        self.up4 = self._make_decoder_block(base_channels*4 + base_channels, base_channels)  # 256+64=320 -> 64
        
        # Final feature projection
        self.out_conv = nn.Conv3d(base_channels, feature_channels, 1)
        
        # Max pooling for encoder
        self.pool = nn.MaxPool3d(2, stride=2)
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create encoder block with Inception-ResNet and max pooling."""
        return nn.Sequential(
            nn.MaxPool3d(2, stride=2),
            DoubleInceptionResNet3D(in_channels, out_channels)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with upsampling and Inception-ResNet."""
        return DoubleInceptionResNet3D(in_channels, out_channels)
    
    def forward(self, x):
        # Encoder path with skip connections (reduced depth)
        x1 = self.inc(x)  # [B, 64, T, H, W]
        
        x2_pool = self.pool(x1)
        x2 = self.down1[1](x2_pool)  # [B, 128, T/2, H/2, W/2]
        
        x3_pool = self.pool(x2)
        x3 = self.down2[1](x3_pool)  # [B, 256, T/4, H/4, W/4]
        
        # Skip the deeper layers to avoid size issues
        # x4_pool = self.pool(x3)
        # x4 = self.down3[1](x4_pool)  # [B, 512, T/8, H/8, W/8]
        
        # Use x3 as bottleneck instead
        bottleneck = self.bottleneck(x3)
        
        # Decoder path with skip connections (simplified)
        up1 = F.interpolate(bottleneck, size=x2.shape[2:], mode='trilinear', align_corners=False)
        up1 = torch.cat([up1, x2], dim=1)  # [B, 512+128, T/2, H/2, W/2] = [B, 640, T/2, H/2, W/2]
        up1 = self.up2(up1)
        
        up2 = F.interpolate(up1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        up2 = torch.cat([up2, x1], dim=1)
        up2 = self.up4(up2)
        
        # Final feature projection
        features = self.out_conv(up2)  # [B, feature_channels, T, H, W]
        
        return features