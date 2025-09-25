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

        self.run_lengths = []  # Track per-run temporal lengths

        # Load high-resolution data
        with h5py.File(self.data_path, 'r') as f:
            # Get frame-style keys and sort them if present
            frame_keys = [k for k in f.keys() if k.startswith('frame_')]
            frame_keys.sort()

            if frame_keys:
                self._load_frame_datasets(f, frame_keys)
            elif all(k in f.keys() for k in ('p', 'b', 'u', 'w')):
                self._load_consolidated_dataset(f)
            else:
                available = list(f.keys())
                raise ValueError(
                    f"Unsupported dataset structure in {self.data_path}. Available top-level keys: {available}"
                )

    def _load_frame_datasets(self, file_handle, frame_keys: List[str]):
        """Load legacy frame_* style datasets."""
        field_keys = ['temperature', 'pressure', 'velocity_x', 'velocity_y']  # Order: [T, p, u, v]

        # Get spatial dimensions from first frame
        first_frame = file_handle[frame_keys[0]]
        temp_data = first_frame['temperature']
        H, W = temp_data.shape
        T = len(frame_keys)

        # Pre-allocate tensor: [T, H, W, 4]
        self.high_res_data = torch.zeros((T, H, W, 4), dtype=torch.float32)

        # Load all frames
        for t_idx, frame_key in enumerate(frame_keys):
            frame = file_handle[frame_key]
            for var_idx, field in enumerate(field_keys):
                self.high_res_data[t_idx, :, :, var_idx] = torch.tensor(
                    frame[field][:], dtype=torch.float32
                )

        # Metadata
        self.Ra = file_handle.attrs.get('Ra', 1e5)
        self.Pr = file_handle.attrs.get('Pr', 0.7)
        self.dt = file_handle.attrs.get('dt', 0.1)

        self.run_lengths = [T]

    def _load_consolidated_dataset(self, file_handle):
        """Load consolidated datasets with [p,b,u,w] structure."""
        # Load channels: shapes [n_runs, n_samples, H, W]
        pressure = torch.from_numpy(file_handle['p'][:]).float()
        temperature = torch.from_numpy(file_handle['b'][:]).float()
        velocity_x = torch.from_numpy(file_handle['u'][:]).float()
        velocity_y = torch.from_numpy(file_handle['w'][:]).float()

        n_runs, n_samples, H, W = pressure.shape
        self.run_lengths = [n_samples] * n_runs

        # Stack to [n_runs, n_samples, H, W, 4] with order [T, p, u, v]
        stacked = torch.stack(
            [temperature, pressure, velocity_x, velocity_y], dim=-1
        )
        self.high_res_data = stacked.view(-1, H, W, 4).clone()

        # Metadata
        self.Ra = file_handle.attrs.get('Ra', 1e5)
        self.Pr = file_handle.attrs.get('Pr', 0.7)
        self.dt = file_handle.attrs.get('dt', 0.1)

        
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

        # Split data per run to avoid mixing sequences at boundaries
        if not self.run_lengths:
            self.run_lengths = [self.T_steps]

        run_slices = []
        cursor = 0
        for length in self.run_lengths:
            run_slices.append(data_tchw[cursor:cursor + length])
            cursor += length

        low_res_runs = []
        low_run_lengths = []

        for run_data in run_slices:
            if run_data.numel() == 0:
                continue

            # Spatial downsampling using average pooling
            if self.spatial_downsample > 1:
                run_downsampled = F.avg_pool2d(
                    run_data,
                    kernel_size=self.spatial_downsample,
                    stride=self.spatial_downsample
                )
            else:
                run_downsampled = run_data

            # Temporal downsampling
            if self.temporal_downsample > 1:
                indices = torch.arange(0, run_downsampled.shape[0], self.temporal_downsample)
                run_downsampled = run_downsampled[indices]

            low_res_runs.append(run_downsampled)
            low_run_lengths.append(run_downsampled.shape[0])

        if low_res_runs:
            self.low_res_data = torch.cat(low_res_runs, dim=0)
        else:
            # Fallback empty tensor if no data was loaded
            self.low_res_data = torch.empty((0, self.n_vars, 0, 0))

        self.low_res_run_lengths = low_run_lengths if low_run_lengths else [self.low_res_data.shape[0]]

        self.T_low = self.low_res_data.shape[0]
        if self.T_low > 0:
            _, _, self.H_low, self.W_low = self.low_res_data.shape
        else:
            self.H_low = self.W_low = 0

    def _create_clips(self):
        """Split data into overlapping temporal clips."""
        self.clips_low = []
        self.clips_high = []

        if not hasattr(self, 'low_res_run_lengths'):
            self.low_res_run_lengths = [self.T_low]

        # Use smaller step size to create more overlapping clips for training data
        step_size = 1 if self.split == 'train' else max(1, self.clip_length // 2)

        low_cursor = 0
        high_cursor = 0

        for run_idx, low_length in enumerate(self.low_res_run_lengths):
            high_length = self.run_lengths[run_idx] if run_idx < len(self.run_lengths) else self.run_lengths[-1]

            if low_length < self.clip_length or high_length < self.clip_length * self.temporal_downsample:
                low_cursor += low_length
                high_cursor += high_length
                continue

            max_start = low_length - self.clip_length + 1

            for i in range(0, max_start, step_size):
                if i + self.clip_length > low_length:
                    break

                low_slice = self.low_res_data[
                    low_cursor + i : low_cursor + i + self.clip_length
                ]
                self.clips_low.append(low_slice)

                start_high = high_cursor + i * self.temporal_downsample
                end_high = start_high + self.clip_length * self.temporal_downsample
                high_indices = torch.arange(start_high, end_high, self.temporal_downsample)
                high_slice = self.high_res_data[high_indices].permute(0, 3, 1, 2)
                self.clips_high.append(high_slice)

            low_cursor += low_length
            high_cursor += high_length

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
