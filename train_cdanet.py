#!/usr/bin/env python3
"""
Fixed CDAnet training script based on sourcecodeCDAnet reference implementation.
This version follows the exact architecture and training procedure from the paper.
"""

from __future__ import division, print_function

import argparse
import os
import sys
import numpy as np
import random
import time
from collections import defaultdict
import json

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

# Add sourcecodeCDAnet to path for importing reference modules
sys.path.append("sourcecodeCDAnet/model")
sys.path.append("sourcecodeCDAnet/train")

# Import reference modules
try:
    from unet3d import UNet3d
    from implicit_net import ImNet
    from nonlinearities import NONLINEARITIES
    from physics import get_rb2_pde_layer
    from local_implicit_grid import query_local_implicit_grid
    import utilities
    print("✅ Successfully imported reference modules")
except ImportError as e:
    print(f"❌ Failed to import reference modules: {e}")
    print("Please ensure sourcecodeCDAnet directory contains the required modules")
    sys.exit(1)

# Fixed data loader based on reference implementation
class FixedRB2DataLoader:
    """
    Fixed data loader that matches the reference implementation exactly.
    Based on sourcecodeCDAnet/train/dataloader_spacetime.py
    """

    def __init__(self, data_dir, data_filename, nx=256, nz=256, nt=8,
                 n_samp_pts_per_crop=1024, downsamp_xz=4, downsamp_t=2,
                 normalize_output=True, return_hres=False, velOnly=False):
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.nx_hres = nx
        self.nz_hres = nz
        self.nt_hres = nt
        self.nx_lres = int(nx/downsamp_xz)
        self.nz_lres = int(nz/downsamp_xz)
        self.nt_lres = int(nt/downsamp_t)
        self.n_samp_pts_per_crop = n_samp_pts_per_crop
        self.downsamp_xz = downsamp_xz
        self.downsamp_t = downsamp_t
        self.normalize_output = normalize_output
        self.return_hres = return_hres
        self.velOnly = velOnly

        # Load data
        self.data = self._load_data_files()
        print(f"Loaded data shape: {self.data.shape}")

        # Setup sampling
        self._setup_sampling()

        # Compute normalization statistics
        self._compute_stats()

    def _load_data_files(self):
        """Load data from HDF5 files in reference format [p,b,u,w]."""
        import h5py
        import glob

        # Try consolidated file first
        consolidated_file = os.path.join(self.data_dir, self.data_filename)
        if os.path.exists(consolidated_file):
            print(f"Loading consolidated data: {consolidated_file}")
            with h5py.File(consolidated_file, 'r') as f:
                # Data format: [channel, file, sample, y, x] -> [channel, time, y, x]
                p_data = f['p'][:]  # Pressure
                b_data = f['b'][:]  # Temperature (buoyancy)
                u_data = f['u'][:]  # x-velocity
                w_data = f['w'][:]  # y-velocity

                # Reshape to [channel, time, y, x] format
                n_files, n_samples, ny, nx = p_data.shape
                total_time = n_files * n_samples

                p_data = p_data.reshape(total_time, ny, nx)
                b_data = b_data.reshape(total_time, ny, nx)
                u_data = u_data.reshape(total_time, ny, nx)
                w_data = w_data.reshape(total_time, ny, nx)

                # Stack channels: [channel, time, y, x]
                data = np.stack([p_data, b_data, u_data, w_data], axis=0)
                return data.astype(np.float32)

        # Fallback: load individual files
        pattern = os.path.join(self.data_dir, "rb_data_Ra_*_run_*.h5")
        files = sorted(glob.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        print(f"Loading {len(files)} individual data files...")

        all_data = []
        for filepath in files:
            with h5py.File(filepath, 'r') as f:
                n_samples = f.attrs['n_samples']
                ny, nx = f.attrs['ny'], f.attrs['nx']

                file_data = np.zeros((4, n_samples, ny, nx), dtype=np.float32)

                for i in range(n_samples):
                    frame_key = f'frame_{i:03d}'
                    if frame_key in f:
                        grp = f[frame_key]
                        file_data[0, i] = grp['pressure'][:]    # p
                        file_data[1, i] = grp['temperature'][:] # b
                        file_data[2, i] = grp['velocity_x'][:]  # u
                        file_data[3, i] = grp['velocity_y'][:]  # w

                all_data.append(file_data)

        # Concatenate along time axis
        data = np.concatenate(all_data, axis=1)  # [4, total_time, ny, nx]
        return data

    def _setup_sampling(self):
        """Setup random sampling following reference implementation."""
        nf_data, nt_data, nz_data, nx_data = self.data.shape

        # Ranges for random cropping
        self.nf_start_range = np.arange(nf_data)  # Files (always 0 for our case)
        self.nx_start_range = np.arange(0, nx_data - self.nx_hres + 1)
        self.nz_start_range = np.arange(0, nz_data - self.nz_hres + 1)
        self.nt_start_range = np.arange(0, nt_data - self.nt_hres + 1)

        # Create sampling grid
        self.rand_grid = np.stack(np.meshgrid(
            [0],  # Single file for our consolidated data
            self.nt_start_range,
            self.nz_start_range,
            self.nx_start_range,
            indexing='ij'
        ), axis=-1)

        self.rand_start_id = self.rand_grid.reshape([-1, 4])
        self.scale_hres = np.array([self.nt_hres, self.nz_hres, self.nx_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nt_lres, self.nz_lres, self.nx_lres], dtype=np.int32)

    def _compute_stats(self):
        """Compute normalization statistics."""
        # Channel-wise statistics over all spatial-temporal dimensions
        self.channel_mean = np.mean(self.data, axis=(1, 2, 3))  # [4]
        self.channel_std = np.std(self.data, axis=(1, 2, 3))    # [4]

        print(f"Channel means: {self.channel_mean}")
        print(f"Channel stds: {self.channel_std}")

    def denormalize_grid(self, normalized_data):
        """Denormalize data back to original scale."""
        if not self.normalize_output:
            return normalized_data

        # normalized_data: [channel, batch, t, z, x]
        # Reshape statistics for broadcasting
        mean = torch.from_numpy(self.channel_mean).float().to(normalized_data.device)
        std = torch.from_numpy(self.channel_std).float().to(normalized_data.device)

        # Reshape for broadcasting: [channel, 1, 1, 1, 1]
        mean = mean.view(-1, 1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1, 1)

        return normalized_data * std + mean

    def __len__(self):
        return self.rand_start_id.shape[0]

    def __getitem__(self, idx):
        """Get training sample following reference format."""
        # Get random crop position
        nf_start, nt_start, nz_start, nx_start = self.rand_start_id[idx]

        # Extract high-res crop: [4, nt_hres, nz_hres, nx_hres]
        hres_crop = self.data[:,
                              nt_start:nt_start + self.nt_hres,
                              nz_start:nz_start + self.nz_hres,
                              nx_start:nx_start + self.nx_hres].copy()

        # Create low-res version by downsampling
        lres_crop = hres_crop[:,
                              ::self.downsamp_t,
                              ::self.downsamp_xz,
                              ::self.downsamp_xz]

        # Normalize if requested
        if self.normalize_output:
            hres_crop = (hres_crop - self.channel_mean.reshape(4, 1, 1, 1)) / self.channel_std.reshape(4, 1, 1, 1)
            lres_crop = (lres_crop - self.channel_mean.reshape(4, 1, 1, 1)) / self.channel_std.reshape(4, 1, 1, 1)

        # Generate random coordinate points for training
        point_coords = np.random.rand(self.n_samp_pts_per_crop, 3).astype(np.float32)

        # Sample values at these coordinates using trilinear interpolation
        point_values = self._sample_at_coordinates(hres_crop, point_coords)

        # Convert to tensors
        hres_crop = torch.from_numpy(hres_crop).float()
        lres_crop = torch.from_numpy(lres_crop).float()
        point_coords = torch.from_numpy(point_coords).float()
        point_values = torch.from_numpy(point_values).float()

        if self.return_hres:
            return hres_crop, lres_crop, point_coords, point_values
        else:
            return lres_crop, point_coords, point_values

    def _sample_at_coordinates(self, field_data, coords):
        """Sample field data at given normalized coordinates using trilinear interpolation."""
        from scipy.interpolate import RegularGridInterpolator

        # field_data: [4, nt, nz, nx]
        # coords: [n_points, 3] in range [0,1]

        nt, nz, nx = field_data.shape[1:]

        # Create coordinate grids
        t_grid = np.linspace(0, 1, nt)
        z_grid = np.linspace(0, 1, nz)
        x_grid = np.linspace(0, 1, nx)

        # Sample each channel
        sampled_values = np.zeros((len(coords), 4), dtype=np.float32)

        for ch in range(4):
            interpolator = RegularGridInterpolator(
                (t_grid, z_grid, x_grid),
                field_data[ch],
                bounds_error=False,
                fill_value=0.0
            )
            sampled_values[:, ch] = interpolator(coords)

        return sampled_values


def train_one_epoch(args, unet, imnet, train_loader, epoch, global_step, device,
                    criterion, writer, optimizer, pde_layer):
    """Training loop following reference implementation."""
    unet.train()
    imnet.train()

    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    metric_logger = utilities.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utilities.SmoothedValue(window_size=1, fmt='{value:.2e}'))
    header = f'Train Epoch {epoch}:'

    for batch_idx, data_tensors in enumerate(
            metric_logger.log_every(train_loader, args.log_interval, header, device=device)):

        data_tensors = [t.to(device) for t in data_tensors]
        input_grid, point_coord, point_value = data_tensors

        optimizer.zero_grad()

        # Forward through U-Net
        latent_grid = unet(input_grid)  # [batch, N, C, T, X, Y]
        # Permute for implicit grid query: [batch, N, T, X, Y, C]
        latent_grid = latent_grid.permute(0, 2, 3, 4, 1)

        # Define forward function for PDE layer
        pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)

        # Update PDE layer and compute predictions + residues
        pde_layer.update_forward_method(pde_fwd_fn)
        pred_value, residue_dict = pde_layer(point_coord, return_residue=True)

        # Function value regression loss
        reg_loss = criterion(pred_value, point_value)

        # PDE residue loss
        pde_tensors = torch.stack([d for d in residue_dict.values()], dim=0)
        pde_loss = criterion(pde_tensors, torch.zeros_like(pde_tensors))

        # Total loss
        total_loss = reg_loss + args.alpha_pde * pde_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(list(unet.parameters()) + list(imnet.parameters()), args.clip_grad)

        optimizer.step()

        # Logging
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(reg_loss=reg_loss.item())
        metric_logger.update(pde_loss=pde_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        if batch_idx % args.log_interval == 0 and utilities.is_main_process():
            writer.add_scalar('train/reg_loss', reg_loss.item(), global_step=global_step[0])
            writer.add_scalar('train/pde_loss', pde_loss.item(), global_step=global_step[0])
            writer.add_scalar('train/total_loss', total_loss.item(), global_step=global_step[0])

        global_step[0] += 1

    return metric_logger.meters['loss'].global_avg


def evaluate_one_epoch(args, unet, imnet, eval_loader, epoch, device, criterion,
                       writer, pde_layer, eval_dataset):
    """Evaluation following reference implementation."""
    unet.eval()
    imnet.eval()

    phys_channels = ["p", "b", "u", "w"]
    xmin = torch.zeros(3, dtype=torch.float32).to(device)
    xmax = torch.ones(3, dtype=torch.float32).to(device)

    metric_logger = utilities.MetricLogger(delimiter="  ")
    header = f'Eval Epoch {epoch}:'

    with torch.no_grad():
        for batch_idx, data_tensors in enumerate(
                metric_logger.log_every(eval_loader, args.log_interval, header, device=device)):

            if batch_idx > 0:  # Only evaluate first batch for speed
                break

            data_tensors = [t.to(device) for t in data_tensors]
            hres_grid, lres_grid, _, _ = data_tensors

            # Forward through models
            latent_grid = unet(lres_grid)  # [batch, C, T, Z, X]
            nb, nc, nt, nz, nx = hres_grid.shape

            # Permute for implicit grid query
            latent_grid = latent_grid.permute(0, 2, 3, 4, 1)  # [batch, T, Z, X, C]

            # Define forward function
            pde_fwd_fn = lambda points: query_local_implicit_grid(imnet, latent_grid, points, xmin, xmax)
            pde_layer.update_forward_method(pde_fwd_fn)

            # Create evaluation grid
            eps = 1e-6
            t_seq = torch.linspace(eps, 1 - eps, nt).to(device)
            z_seq = torch.linspace(eps, 1 - eps, nz).to(device)
            x_seq = torch.linspace(eps, 1 - eps, nx).to(device)

            query_coord = torch.stack(torch.meshgrid(t_seq, z_seq, x_seq, indexing='ij'), dim=-1)
            query_coord = query_coord.reshape([-1, 3])
            n_query = query_coord.shape[0]

            # Evaluate in batches
            res_dict = defaultdict(list)
            n_iters = int(np.ceil(n_query / args.pseudo_epoch_size))

            for idx in range(n_iters):
                sid = idx * args.pseudo_epoch_size
                eid = min(sid + args.pseudo_epoch_size, n_query)
                query_coord_batch = query_coord[sid:eid]
                query_coord_batch = query_coord_batch[None].expand(nb, eid - sid, 3)

                pred_value = pde_layer(query_coord_batch, return_residue=False)
                pred_value = pred_value.detach()

                for name, chan_id in zip(phys_channels, range(4)):
                    res_dict[name].append(pred_value[..., chan_id])

            # Reconstruct full grids
            for key in phys_channels:
                res_dict[key] = torch.cat(res_dict[key], dim=1).reshape([nb, nt, nz, nx])

            # Stack predictions: [batch, channel, t, z, x]
            pred_tensor = torch.stack([res_dict[key] for key in phys_channels], dim=1)

            # Denormalize predictions
            pred_tensor = eval_dataset.denormalize_grid(pred_tensor.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)

            # Compute loss
            loss = criterion(pred_tensor, hres_grid)
            metric_logger.update(loss=loss.item())

            # Compute evaluation metrics
            eval_metrics = compute_eval_metrics(pred_tensor.detach(), hres_grid.detach())
            metric_logger.update(**eval_metrics)

            break  # Only evaluate first batch

    total_loss = metric_logger.meters['loss'].global_avg
    total_eval_metrics = {k: metric_logger.meters[k].global_avg
                         for k in eval_metrics.keys()}
    return total_loss, total_eval_metrics


def compute_eval_metrics(pred, target):
    """Compute evaluation metrics following reference implementation."""
    nBatch, n_ch, n_t, n_z, n_x = target.shape

    difference = pred - target
    diffSq = torch.square(difference)

    # Sum over space
    sumX = torch.sum(diffSq, dim=4)
    sumSquared = torch.sum(sumX, dim=3)

    # Normalization
    normalization = torch.sqrt(
        torch.sum(torch.sum(torch.square(target), dim=4), dim=3) * (1 / (float(n_x) * float(n_z)))
    )

    RMSE = torch.sqrt(sumSquared * (1 / (float(n_x) * float(n_z))))
    RRMSE = RMSE / (normalization + 1e-10)  # Add small epsilon to avoid division by zero

    # Average over time and batches
    RRMSE = torch.mean(RRMSE, dim=2)  # Over time
    RRMSE = torch.mean(RRMSE, dim=0)  # Over batch

    results = {
        'RRMSE_p': RRMSE[0].item(),
        'RRMSE_T': RRMSE[1].item(),
        'RRMSE_u': RRMSE[2].item(),
        'RRMSE_v': RRMSE[3].item(),
    }

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fixed CDAnet Training')

    # Data parameters
    parser.add_argument('--data_folder', type=str, default='./rb_data_numerical',
                       help='Path to data directory')
    parser.add_argument('--train_data', type=str, default='rb2d_ra1e5_consolidated.h5',
                       help='Training data filename')
    parser.add_argument('--eval_data', type=str, default='rb2d_ra1e5_consolidated.h5',
                       help='Evaluation data filename')

    # Grid parameters
    parser.add_argument('--nx', type=int, default=256, help='High-resolution grid points in x')
    parser.add_argument('--nz', type=int, default=256, help='High-resolution grid points in z')
    parser.add_argument('--nt', type=int, default=8, help='Number of timesteps per clip')
    parser.add_argument('--downsamp_xz', type=int, default=4, help='Spatial downsampling factor')
    parser.add_argument('--downsamp_t', type=int, default=2, help='Temporal downsampling factor')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size per GPU (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--alpha_pde', type=float, default=1.0, help='PDE loss weight (default: 1.0)')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--pseudo_epoch_size', type=int, default=3000,
                       help='Number of samples per pseudo-epoch (default: 3000)')
    parser.add_argument('--num_workers', type=int, default=6, help='Data loader workers')
    parser.add_argument('--log_interval', type=int, default=50, help='Logging interval in batches')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'huber'],
                       help='Regression loss type (default: l1)')

    # Model parameters
    parser.add_argument('--lat_dims', type=int, default=32, help='Latent dimensions')
    parser.add_argument('--unet_nf', type=int, default=16, help='U-Net base features')
    parser.add_argument('--unet_mf', type=int, default=256, help='U-Net max features')
    parser.add_argument('--imnet_nf', type=int, default=32, help='ImNet width')
    parser.add_argument('--nonlin', type=str, default='softplus', choices=list(NONLINEARITIES.keys()),
                       help='Implicit decoder nonlinearity')

    # Physics parameters
    parser.add_argument('--rayleigh', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--prandtl', type=float, default=0.7, help='Prandtl number')
    parser.add_argument('--nt_numerical', type=float, default=0.05, help='Numerical time scale')
    parser.add_argument('--nz_numerical', type=float, default=256.0, help='Numerical z scale')
    parser.add_argument('--nx_numerical', type=float, default=256.0, help='Numerical x scale')

    # Regularisation and options
    parser.add_argument('--velocityOnly', action='store_true', help='Use velocity observations only')
    parser.add_argument('--n_samp_pts_per_crop', type=int, default=1024,
                       help='Sample points per crop')

    parser.add_argument('--normalize_channels', dest='normalize_channels', action='store_true',
                       help='Normalize data channels (default: True)')
    parser.add_argument('--no_normalize_channels', dest='normalize_channels', action='store_false',
                       help='Disable channel normalization')
    parser.set_defaults(normalize_channels=True)

    parser.add_argument('--lr_scheduler', dest='lr_scheduler', action='store_true',
                       help='Use ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--no_lr_scheduler', dest='lr_scheduler', action='store_false',
                       help='Disable learning rate scheduler')
    parser.set_defaults(lr_scheduler=True)

    parser.add_argument('--use_continuity', dest='use_continuity', action='store_true',
                       help='Use continuity equation (mass conservation)')
    parser.add_argument('--no_use_continuity', dest='use_continuity', action='store_false',
                       help='Do not enforce continuity equation')
    parser.set_defaults(use_continuity=False)

    parser.add_argument('--output_folder', type=str, default='./outputs_fixed',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device (auto=自动选择最佳设备)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    print("=" * 60)
    print("Fixed CDAnet Training (Reference Implementation)")
    print("=" * 60)

    # Setup device - 兼容Mac和CUDA
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Using CUDA GPU")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Using Apple Silicon MPS")
        else:
            device = torch.device('cpu')
            print("💻 Using CPU")
    else:
        device = torch.device(args.device)
        print(f"Using specified device: {device}")

    print(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_folder, 'tensorboard'))

    # Create datasets
    print("Creating datasets...")
    train_dataset = FixedRB2DataLoader(
        data_dir=args.data_folder,
        data_filename=args.train_data,
        nx=args.nx, nz=args.nz, nt=args.nt,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels,
        return_hres=False,
        velOnly=args.velocityOnly
    )

    eval_dataset = FixedRB2DataLoader(
        data_dir=args.data_folder,
        data_filename=args.eval_data,
        nx=args.nx, nz=args.nz, nt=args.nt,
        n_samp_pts_per_crop=args.n_samp_pts_per_crop,
        downsamp_xz=args.downsamp_xz, downsamp_t=args.downsamp_t,
        normalize_output=args.normalize_channels,
        return_hres=True,
        velOnly=args.velocityOnly
    )

    # Create data loaders
    train_sampler = RandomSampler(train_dataset, replacement=True,
                                 num_samples=args.pseudo_epoch_size)
    eval_sampler = RandomSampler(eval_dataset, replacement=True, num_samples=1000)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             sampler=train_sampler, num_workers=args.num_workers,
                             pin_memory=True, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                            sampler=eval_sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)

    # Create models
    print("Creating models...")
    in_features = 2 if args.velocityOnly else 4

    unet = UNet3d(in_features=in_features, out_features=args.lat_dims,
                  igres=train_dataset.scale_lres, nf=args.unet_nf, mf=args.unet_mf)
    imnet = ImNet(dim=3, in_features=args.lat_dims, out_features=4,
                  nf=args.imnet_nf, activation=NONLINEARITIES[args.nonlin])

    unet.to(device)
    imnet.to(device)

    # Count parameters
    unet_params = sum(p.numel() for p in unet.parameters())
    imnet_params = sum(p.numel() for p in imnet.parameters())
    print(f"Model parameters: {unet_params:,} (U-Net) + {imnet_params:,} (ImNet) = {unet_params + imnet_params:,} total")

    # Create optimizer
    optimizer = torch.optim.SGD(
        list(unet.parameters()) + list(imnet.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Create loss function
    if args.loss_type == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss_type == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.SmoothL1Loss()

    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=6
        )

    # Create PDE layer
    mean = train_dataset.channel_mean if args.normalize_channels else None
    std = train_dataset.channel_std if args.normalize_channels else None

    pde_layer = get_rb2_pde_layer(
        mean=mean, std=std,
        t_crop=args.nt * args.nt_numerical,
        z_crop=args.nz * (1.0 / args.nz_numerical),
        x_crop=args.nx * (1.0 / args.nx_numerical),
        prandtl=args.prandtl, rayleigh=args.rayleigh,
        use_continuity=args.use_continuity
    )

    # Training state
    start_epoch = 0
    global_step = [0]
    best_loss = float('inf')
    best_metric = float('inf')

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        unet.load_state_dict(checkpoint['unet_state_dict'])
        imnet.load_state_dict(checkpoint['imnet_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step[0] = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_metric = checkpoint.get('best_metric', float('inf'))
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Starting training from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_one_epoch(
            args, unet, imnet, train_loader, epoch, global_step, device,
            criterion, writer, optimizer, pde_layer
        )

        # Evaluate
        eval_loss, eval_metrics = evaluate_one_epoch(
            args, unet, imnet, eval_loader, epoch, device, criterion,
            writer, pde_layer, eval_dataset
        )

        print(f"Train Loss: {train_loss:.2e}")
        print(f"Eval Loss: {eval_loss:.2e}")
        print(f"Eval Metrics: {eval_metrics}")

        # Scheduler step
        if scheduler is not None:
            scheduler.step(eval_loss)

        current_metric = eval_metrics.get('RRMSE_T', eval_loss)
        best_loss = min(best_loss, eval_loss)
        is_best = current_metric < best_metric
        if is_best:
            best_metric = current_metric

        # Tensorboard logging aligned with reference naming
        writer.add_scalar('metrics_per_epoch/train_loss', train_loss, epoch)
        writer.add_scalar('metrics_per_epoch/valid_loss', eval_loss, epoch)
        writer.add_scalar('metrics_per_epoch/lr', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('metrics_per_epoch/RRMSE_T_best', best_metric, epoch)
        for k, v in eval_metrics.items():
            writer.add_scalar(f'metrics_per_epoch/{k}', v, epoch)

        model_config = {
            'in_channels': in_features,
            'feature_channels': args.lat_dims,
            'mlp_hidden_dims': [args.lat_dims, args.imnet_nf],
            'activation': args.nonlin,
            'coord_dim': 3,
            'output_dim': 4,
            'igres': tuple(int(x) for x in train_dataset.scale_lres),
            'unet_nf': args.unet_nf,
            'unet_mf': args.unet_mf,
            'imnet_nf': args.imnet_nf
        }

        checkpoint = {
            'epoch': epoch,
            'unet_state_dict': unet.state_dict(),
            'imnet_state_dict': imnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'global_step': global_step[0],
            'best_loss': best_loss,
            'best_metric': best_metric,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_metrics': eval_metrics,
            'model_config': model_config,
            'args': args
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.output_folder, 'checkpoint_latest.pth'))

        # Save best checkpoint
        if is_best:
            print(f"New best model! RRMSE_T: {current_metric:.2e}")
            torch.save(checkpoint, os.path.join(args.output_folder, 'checkpoint_best.pth'))

        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_folder, f'checkpoint_epoch_{epoch:03d}.pth'))

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best evaluation loss: {best_loss:.2e}")
    print(f"Best RRMSE_T: {best_metric:.2e}")
    print(f"Checkpoints saved to: {args.output_folder}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()
