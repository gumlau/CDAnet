"""
Dataset classes for CDAnet training and evaluation.
Handles multi-resolution spatio-temporal data for Rayleigh-Bénard convection.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from typing import Dict, List, Tuple, Optional


class RBDataset(Dataset):
    """
    Dataset for Rayleigh-Bénard convection data.
    
    Args:
        data_path: Path to HDF5 data files
        spatial_downsample: Spatial downsampling factor (γ_s)
        temporal_downsample: Temporal downsampling factor (γ_t) 
        clip_length: Length of temporal clips (8 snapshots)
        domain_size: Tuple of domain dimensions (Lx, Ly)
        split: Dataset split ('train', 'val', 'test')
        transform: Optional data transform function
    """
    
    def __init__(self, data_path: str, spatial_downsample: int = 4, temporal_downsample: int = 4,
                 clip_length: int = 8, domain_size: Tuple[float, float] = (3.0, 1.0),
                 split: str = 'train', transform: Optional = None):
        
        self.data_path = data_path
        self.spatial_downsample = spatial_downsample  # γ_s
        self.temporal_downsample = temporal_downsample  # γ_t
        self.clip_length = clip_length
        self.domain_size = domain_size
        self.split = split
        self.transform = transform
        
        self._load_data()
        self._setup_coordinates()
        
    def _load_data(self):
        """Load data from HDF5 files."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Load high-resolution data
        with h5py.File(self.data_path, 'r') as f:
            # Get frame keys and sort them
            frame_keys = [k for k in f.keys() if k.startswith('frame_')]
            frame_keys.sort()

            if not frame_keys:
                raise ValueError(f"No frame data found in {self.data_path}")

            # Load first frame to get dimensions
            first_frame = f[frame_keys[0]]
            field_keys = ['temperature', 'pressure', 'velocity_x', 'velocity_y']  # Order: [T, p, u, v]

            # Get spatial dimensions
            temp_data = first_frame['temperature']
            H, W = temp_data.shape
            T = len(frame_keys)

            # Pre-allocate tensor: [T, H, W, 4]
            self.high_res_data = torch.zeros((T, H, W, 4), dtype=torch.float32)

            # Load all frames
            for t_idx, frame_key in enumerate(frame_keys):
                frame = f[frame_key]
                for var_idx, field in enumerate(field_keys):
                    self.high_res_data[t_idx, :, :, var_idx] = torch.tensor(
                        frame[field][:], dtype=torch.float32
                    )

            # Get metadata
            if 'Ra' in f.attrs:
                self.Ra = f.attrs['Ra']
            else:
                self.Ra = 1e5  # Default

            if 'Pr' in f.attrs:
                self.Pr = f.attrs['Pr']
            else:
                self.Pr = 0.7  # Default

            if 'dt' in f.attrs:
                self.dt = f.attrs['dt']
            else:
                self.dt = 0.1  # Default
        
        # Data shape: [T, H, W, 4]
        self.T_steps, self.H_high, self.W_high, self.n_vars = self.high_res_data.shape
        
        # Create low-resolution data by downsampling
        self._create_low_res_data()
        
        # Split data into clips
        self._create_clips()
    
    def _create_low_res_data(self):
        """Create low-resolution data by spatial and temporal downsampling."""
        # Rearrange to [T, 4, H, W] for downsampling
        data_tchw = self.high_res_data.permute(0, 3, 1, 2)  # [T, 4, H, W]
        
        # Spatial downsampling using average pooling
        if self.spatial_downsample > 1:
            data_downsampled = F.avg_pool2d(
                data_tchw.view(-1, self.n_vars, self.H_high, self.W_high),
                kernel_size=self.spatial_downsample,
                stride=self.spatial_downsample
            )
            data_downsampled = data_downsampled.view(
                self.T_steps, self.n_vars, 
                self.H_high // self.spatial_downsample,
                self.W_high // self.spatial_downsample
            )
        else:
            data_downsampled = data_tchw
            
        # Temporal downsampling
        if self.temporal_downsample > 1:
            indices = torch.arange(0, self.T_steps, self.temporal_downsample)
            self.low_res_data = data_downsampled[indices]  # [T', 4, H', W']
        else:
            self.low_res_data = data_downsampled
            
        self.T_low, _, self.H_low, self.W_low = self.low_res_data.shape
        
    def _create_clips(self):
        """Split data into overlapping temporal clips."""
        self.clips_low = []
        self.clips_high = []

        # Determine available clips
        max_clips_low = max(0, self.T_low - self.clip_length + 1)
        max_clips_high = max(0, self.T_steps - self.clip_length * self.temporal_downsample + 1)

        max_clips = min(max_clips_low, max_clips_high // self.temporal_downsample)

        # Use smaller step size to create more overlapping clips for training data
        step_size = 1 if self.split == 'train' else max(1, self.clip_length // 2)

        # Create clips with overlapping
        for i in range(0, max_clips, step_size):
            if i + self.clip_length > self.T_low:
                break

            # Low-resolution clip
            low_clip = self.low_res_data[i:i + self.clip_length]  # [8, 4, H_low, W_low]
            self.clips_low.append(low_clip)

            # Corresponding high-resolution clip
            start_high = i * self.temporal_downsample
            end_high = start_high + self.clip_length * self.temporal_downsample
            high_indices = torch.arange(start_high, end_high, self.temporal_downsample)
            high_clip = self.high_res_data[high_indices].permute(0, 3, 1, 2)  # [8, 4, H_high, W_high]
            self.clips_high.append(high_clip)

        print(f"Created {len(self.clips_low)} clips for {self.split} split")
        
    def _setup_coordinates(self):
        """Setup normalized spatio-temporal coordinates."""
        # Spatial and temporal coordinates normalized to [0, 1]
        t = torch.linspace(0.0, 1.0, self.clip_length)
        z = torch.linspace(0.0, 1.0, self.H_high)
        x = torch.linspace(0.0, 1.0, self.W_high)

        # Create meshgrid with ordering [t, z, x]
        self.T, self.Z, self.X = torch.meshgrid(t, z, x, indexing='ij')

        # Flatten and stack coordinates [T*Z*X, 3] in [t, z, x] order
        coords = torch.stack([
            self.T.flatten(),
            self.Z.flatten(),
            self.X.flatten()
        ], dim=1)
        
        self.coords_template = coords  # [N, 3] where N = H*W*T
        
    def __len__(self):
        return len(self.clips_low)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns:
            sample: Dictionary containing:
                - low_res: Low-resolution clip [8, 4, H_low, W_low]
                - high_res: High-resolution clip [8, 4, H_high, W_high] 
                - coords: Coordinates for high-res grid [T*H*W, 3]
                - targets: Flattened high-res targets [T*H*W, 4]
        """
        low_res_clip = self.clips_low[idx]  # [8, 4, H_low, W_low]
        high_res_clip = self.clips_high[idx]  # [8, 4, H_high, W_high]
        
        # Flatten high-res targets: [8, 4, H, W] -> [H*W*8, 4]
        targets = high_res_clip.permute(0, 2, 3, 1).reshape(-1, 4)  # [T*H*W, 4]
        
        # Coordinates
        coords = self.coords_template.clone()  # [T*H*W, 3]
        
        # Convert to expected format: [8, 4, H, W] -> [4, 8, H, W] for model input
        low_res_model = low_res_clip.permute(1, 0, 2, 3)  # [4, 8, H_low, W_low]
        high_res_model = high_res_clip.permute(1, 0, 2, 3)  # [4, 8, H_high, W_high]
        
        sample = {
            'low_res': low_res_model,  # [4, 8, H_low, W_low]
            'high_res': high_res_model,  # [4, 8, H_high, W_high] 
            'coords': coords,  # [T*H*W, 3] in [t, z, x] order
            'targets': targets,  # [T*H*W, 4] 
            'Ra': self.Ra,
            'Pr': self.Pr,
            'dt': self.dt
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class RandomCoordinateSampler:
    """
    Randomly samples coordinates for PDE loss computation.
    
    Args:
        n_points: Number of points to sample for PDE loss (default: 1024)
    """
    
    def __init__(self, n_points: int = 1024):
        self.n_points = n_points
        
    def __call__(self, sample):
        """Sample random coordinates from the full coordinate set."""
        total_coords = sample['coords'].shape[0]
        
        if total_coords <= self.n_points:
            # If we have fewer points than requested, use all
            sample['pde_coords'] = sample['coords']
            sample['pde_targets'] = sample['targets']
        else:
            # Random sampling
            indices = torch.randperm(total_coords)[:self.n_points]
            sample['pde_coords'] = sample['coords'][indices]
            sample['pde_targets'] = sample['targets'][indices]
            
        return sample


class DataNormalizer:
    """Normalize data to improve training stability."""
    
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        self.mean = mean
        self.std = std
        
    def fit(self, data: torch.Tensor):
        """Compute mean and std from data."""
        # data shape: [N, 4] for targets
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0) + 1e-8
        
    def normalize(self, data: torch.Tensor):
        """Normalize data."""
        return (data - self.mean) / self.std
        
    def denormalize(self, data: torch.Tensor):
        """Denormalize data."""
        return data * self.std + self.mean
        
    def __call__(self, sample):
        """Apply normalization to sample."""
        if self.mean is not None and self.std is not None:
            sample['targets'] = self.normalize(sample['targets'])
            if 'pde_targets' in sample:
                sample['pde_targets'] = self.normalize(sample['pde_targets'])
        return sample
