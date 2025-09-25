#!/usr/bin/env python3
"""
Simple data converter to create training-ready data from your existing files.
"""

import h5py
import numpy as np
import os

def convert_existing_data():
    """Convert existing data to training format."""

    # Use the largest data file
    input_file = 'rb_data_numerical/rb_data_Ra_1e+05.h5'
    output_file = 'rb_data_numerical/rb2d_ra1e+05_consolidated.h5'

    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return False

    print(f"Converting {input_file} to training format...")

    with h5py.File(input_file, 'r') as f:
        # Your data: [n_samples, ny, nx, 4] where channels are [T, p, u, v]
        data = f['data'][:]
        print(f"Input data shape: {data.shape}")

        # Split channels correctly
        n_samples, ny, nx, n_channels = data.shape

        # Your channel order is [T, p, u, v]
        # Reference expects [p, b, u, w] where b=temperature, w=v-velocity
        T_channel = data[:, :, :, 0]  # Temperature
        p_channel = data[:, :, :, 1]  # Pressure
        u_channel = data[:, :, :, 2]  # U-velocity
        v_channel = data[:, :, :, 3]  # V-velocity

        print("Channel statistics:")
        print(f"  Temperature: [{T_channel.min():.3f}, {T_channel.max():.3f}]")
        print(f"  Pressure: [{p_channel.min():.3f}, {p_channel.max():.3f}]")
        print(f"  U-velocity: [{u_channel.min():.3f}, {u_channel.max():.3f}]")
        print(f"  V-velocity: [{v_channel.min():.3f}, {v_channel.max():.3f}]")

        # Create reference format: split into multiple "runs"
        # Reference expects [n_runs, n_samples_per_run, ny, nx]
        # Let's make 2 runs of 25 samples each
        n_runs = 2
        samples_per_run = n_samples // n_runs

        # Reshape to [n_runs, samples_per_run, ny, nx] for each channel
        p_data = p_channel[:n_runs*samples_per_run].reshape(n_runs, samples_per_run, ny, nx)
        b_data = T_channel[:n_runs*samples_per_run].reshape(n_runs, samples_per_run, ny, nx)  # T -> b
        u_data = u_channel[:n_runs*samples_per_run].reshape(n_runs, samples_per_run, ny, nx)
        w_data = v_channel[:n_runs*samples_per_run].reshape(n_runs, samples_per_run, ny, nx)  # v -> w

        print(f"Reshaped to: {n_runs} runs × {samples_per_run} samples × {ny}×{nx}")

    # Save in reference format
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data.astype(np.float32), compression='gzip')
        f.create_dataset('b', data=b_data.astype(np.float32), compression='gzip')
        f.create_dataset('u', data=u_data.astype(np.float32), compression='gzip')
        f.create_dataset('w', data=w_data.astype(np.float32), compression='gzip')

        # Metadata
        f.attrs['Ra'] = 1e5
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = samples_per_run
        f.attrs['format'] = 'reference_compatible'
        f.attrs['channels'] = 'p,b,u,w'

    print(f"✅ Converted data saved: {output_file}")
    print(f"   Format: [p,b,u,w] × {n_runs} × {samples_per_run} × {ny} × {nx}")

    return True

def create_minimal_training_script():
    """Create a minimal training script that should work."""
    script_content = '''#!/usr/bin/env python3
"""
Minimal CDAnet training using your existing data.
"""

import sys
import os
sys.path.append("sourcecodeCDAnet/model")
sys.path.append("sourcecodeCDAnet/train")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py

class SimpleRBDataset(Dataset):
    def __init__(self, data_file, nx=64, nz=64, nt=8):
        with h5py.File(data_file, 'r') as f:
            # Load data: [n_runs, n_samples, ny, nx] for each channel
            self.p_data = f['p'][:]
            self.b_data = f['b'][:]
            self.u_data = f['u'][:]
            self.w_data = f['w'][:]

        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.n_runs, self.n_samples, self.ny, self.nx_full = self.p_data.shape

        # Compute normalization statistics
        all_data = np.stack([self.p_data, self.b_data, self.u_data, self.w_data], axis=0)
        self.mean = np.mean(all_data, axis=(1,2,3,4))
        self.std = np.std(all_data, axis=(1,2,3,4)) + 1e-8

    def __len__(self):
        return self.n_runs * (self.n_samples - self.nt + 1) * 10  # Multiple crops per sequence

    def __getitem__(self, idx):
        # Random crop in space and time
        run_idx = np.random.randint(0, self.n_runs)
        t_start = np.random.randint(0, self.n_samples - self.nt + 1)
        y_start = np.random.randint(0, self.ny - self.nz + 1)
        x_start = np.random.randint(0, self.nx_full - self.nx + 1)

        # Extract crop
        p_crop = self.p_data[run_idx, t_start:t_start+self.nt, y_start:y_start+self.nz, x_start:x_start+self.nx]
        b_crop = self.b_data[run_idx, t_start:t_start+self.nt, y_start:y_start+self.nz, x_start:x_start+self.nx]
        u_crop = self.u_data[run_idx, t_start:t_start+self.nt, y_start:y_start+self.nz, x_start:x_start+self.nx]
        w_crop = self.w_data[run_idx, t_start:t_start+self.nt, y_start:y_start+self.nz, x_start:x_start+self.nx]

        # Stack and normalize
        data = np.stack([p_crop, b_crop, u_crop, w_crop], axis=0)  # [4, nt, nz, nx]
        data = (data - self.mean.reshape(4,1,1,1)) / self.std.reshape(4,1,1,1)

        return torch.from_numpy(data).float()

def simple_train():
    """Simple training loop."""
    print("🚀 Starting simple CDAnet training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create dataset
    dataset = SimpleRBDataset('rb_data_numerical/rb2d_ra1e+05_consolidated.h5', nx=64, nz=64, nt=8)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    # Simple model (placeholder)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(4, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(64, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(64, 4, 3, padding=1)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    print("Training for 10 epochs...")
    for epoch in range(10):
        total_loss = 0
        count = 0

        for batch_idx, data in enumerate(dataloader):
            if batch_idx >= 10:  # Only 10 batches per epoch for speed
                break

            data = data.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)  # Reconstruction loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count > 0 else 0
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")

        if avg_loss < 0.1:
            print("✅ Training converged!")
            break

    print("✅ Training completed!")

if __name__ == '__main__':
    simple_train()
'''

    with open('simple_train.py', 'w') as f:
        f.write(script_content)

    print("✅ Minimal training script created: simple_train.py")

if __name__ == '__main__':
    print("🔧 Converting Data for CDAnet Training")
    print("=" * 50)

    if convert_existing_data():
        create_minimal_training_script()

        print("\n🚀 Next steps:")
        print("1. Run: python3 simple_train.py")
        print("2. If it works, try the full training with reference architecture")
        print("3. Check that loss decreases and no NaN values occur")

    else:
        print("❌ Data conversion failed")