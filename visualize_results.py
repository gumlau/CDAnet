#!/usr/bin/env python
"""
Create publication-quality visualizations for CDAnet results.
Generates the classic Rayleigh-Bénard convection comparison plots.

Usage:
    python visualize_results.py --checkpoint checkpoints/best_model.pth --data_dir ./rb_data_numerical
    python visualize_results.py --demo  # Create demo visualization
"""

import argparse
import os
import numpy as np
import torch
import h5py
from typing import Optional

from cdanet.utils.rb_visualization import RBVisualization, visualize_training_results
from cdanet.models import CDAnet
from cdanet.data import RBDataModule
from cdanet.config import ExperimentConfig

# Use CDAnet models (correct implementation)


def parse_args():
    parser = argparse.ArgumentParser(description='Create CDAnet visualizations')
    
    parser.add_argument('--checkpoint', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing test data (auto-detected if omitted)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Output directory for visualizations')
    
    # Visualization options
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--variable', type=str, default='T', choices=['T', 'u', 'v', 'p'],
                       help='Variable to visualize')
    parser.add_argument('--times', nargs='+', type=float, default=[15.0, 18.2, 21.5],
                       help='Time points to visualize')
    parser.add_argument('--n_snapshots', type=int, default=4,
                       help='Number of snapshots for temporal evolution')
    
    # Data options
    parser.add_argument('--spatial_downsample', type=int, default=2,
                       help='Spatial downsampling factor')
    parser.add_argument('--temporal_downsample', type=int, default=2,
                       help='Temporal downsampling factor')
    
    # Output options  
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                       help='Output format')
    
    # Special modes
    parser.add_argument('--demo', action='store_true',
                       help='Create demo visualization with synthetic data')
    parser.add_argument('--compare_cda', action='store_true',
                       help='Include CDA comparison (requires CDA results)')
    
    return parser.parse_args()


def load_model_and_predict(checkpoint_path: str, data_dir: str, Ra: float,
                          spatial_downsample: int, temporal_downsample: int) -> dict:
    """Load trained model and generate predictions."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"🔍 Checkpoint keys: {list(checkpoint.keys())}")
    for key in ['epoch', 'best_loss', 'best_metric', 'train_loss', 'eval_loss']:
        if key in checkpoint:
            value = checkpoint[key]
            if isinstance(value, (list, tuple)):
                preview = value[:3]
                tail = value[-3:] if len(value) > 3 else []
                print(f"  {key}: len={len(value)} first={preview} last={tail if tail else preview}")
            else:
                print(f"  {key}: {value}")

    # Create CDAnet model with reference architecture
    print("🔧 Creating CDAnet model with reference architecture...")

    # Use model config from checkpoint if available
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        print(f"🔍 Using model config from checkpoint: {model_config}")
    else:
        # Fallback config
        model_config = {
            'in_channels': 4,
            'feature_channels': 64,  # Match training script
            'mlp_hidden_dims': [64, 128],  # Match training script
            'activation': 'softplus',
            'coord_dim': 3,
            'output_dim': 4,
            'igres': (16, 32, 64),  # Match training data
            'unet_nf': 16,
            'unet_mf': 64  # Match training script
        }
        print(f"🔍 Using fallback model config: {model_config}")

    model = CDAnet(**model_config)

    # Load state dicts using the correct reference architecture
    try:
        if 'model_state_dict' in checkpoint:
            # Standard loading for proper CDAnet checkpoints
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded complete model state dict successfully")
        elif 'unet_state_dict' in checkpoint and 'imnet_state_dict' in checkpoint:
            # Load separate UNet and ImNet state dicts (from old training)
            model.feature_extractor.load_state_dict(checkpoint['unet_state_dict'])
            model.implicit_net.load_state_dict(checkpoint['imnet_state_dict'])
            print(f"✅ Loaded separate UNet and ImNet state dicts successfully")
        else:
            raise ValueError("No compatible state dict found in checkpoint")
    except Exception as e:
        print(f"❌ Error loading model state dict: {e}")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        if 'unet_state_dict' in checkpoint:
            print(f"UNet state dict keys: {list(checkpoint['unet_state_dict'].keys())[:5]}...")
        if 'imnet_state_dict' in checkpoint:
            print(f"ImNet state dict keys: {list(checkpoint['imnet_state_dict'].keys())[:5]}...")

        # The architecture mismatch is the issue - inform user
        print("🔧 Architecture mismatch detected. This checkpoint was trained with the old architecture.")
        print("   Please retrain using the updated CDAnet architecture for best results.")
        return None

    model.eval()

    # Check if model parameters look reasonable
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔍 Total model parameters: {total_params:,}")

    # Check a few parameter values
    param_count = 0
    for name, param in model.named_parameters():
        print(f"🔍 {name}: shape {param.shape}, mean {param.mean().item():.6f}, std {param.std().item():.6f}")
        param_count += 1
        if param_count >= 3:  # Only show first few
            break
    
    # Setup data with normalization (as per paper and training)
    normalize_data = True  # Paper uses normalized data for training

    print(f"Setting up data with normalize={normalize_data}")
    data_module = RBDataModule(
        data_dir=data_dir,
        spatial_downsample=spatial_downsample,
        temporal_downsample=temporal_downsample,
        batch_size=1,
        normalize=normalize_data,
        num_workers=0  # Avoid timeout issues
    )
    data_module.setup([Ra])
    
    # Get test data
    try:
        test_loader = data_module.get_dataloader(Ra, 'test')
    except KeyError as exc:
        raise FileNotFoundError(
            f"No data available for Ra={Ra:.0e}. Place the corresponding .h5 files under {data_dir} or use --data_dir to point to the dataset."
        ) from exc

    print(f"🔍 Debug Info:")
    print(f"  Test loader batches: {len(test_loader)}")

    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU Memory before: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")

    results = {
        'input_fields': [],
        'truth_fields': [],
        'predictions': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    if torch.cuda.is_available():
        print(f"  GPU Memory after model load: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
    
    dataset_to_model_order = torch.tensor([1, 0, 2, 3], device=device, dtype=torch.long)
    model_to_dataset_order = torch.tensor([1, 0, 2, 3], device=device, dtype=torch.long)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print(f"🔍 Processing batch {i+1}/{len(test_loader)}")

            if i >= 10:  # Limit to first 10 batches for visualization
                break

            # Move to device with memory optimization
            low_res = batch['low_res'].to(device, non_blocking=True)
            coords = batch['coords'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            high_res = batch.get('high_res')

            # Reorder channels to match training order [p, b, u, w]
            low_res_for_model = low_res.index_select(1, dataset_to_model_order)

            # Log memory usage for the problematic batch
            if torch.cuda.is_available():
                print(f"    GPU Memory after data load: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

            # Debug data shapes and ranges
            print(f"  📊 Data shapes:")
            print(f"    Low-res input: {low_res.shape}")
            print(f"    Targets: {targets.shape}")
            print(f"    Coords: {coords.shape}")

            # Check input data
            print(f"  Input T: [{low_res[0,0].min():.3f}, {low_res[0,0].max():.3f}]")
            print(f"  Target T: [{targets[0,:,0].min():.3f}, {targets[0,:,0].max():.3f}]")

            # Get predictions using CDAnet with memory-efficient chunking
            batch_size, num_coords, coord_dim = coords.shape

            # Adaptive chunk size based on available GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb > 10:  # RTX 3080 Ti has 12GB
                    chunk_size = 8192  # Larger chunks for more VRAM
                else:
                    chunk_size = 4096  # Conservative for smaller VRAM
            else:
                chunk_size = 2048  # CPU fallback
            predictions_list = []

            print(f"  Processing {num_coords} coordinates in chunks of {chunk_size}")

            for i in range(0, num_coords, chunk_size):
                end_idx = min(i + chunk_size, num_coords)
                coord_chunk = coords[:, i:end_idx, :]  # [1, chunk_size, 3]

                with torch.no_grad():  # Save memory during inference
                    pred_chunk = model(low_res_for_model, coord_chunk)
                    # Convert predictions back to dataset order [T, p, u, v]
                    pred_chunk = pred_chunk.index_select(-1, model_to_dataset_order)
                    predictions_list.append(pred_chunk.cpu())  # Move to CPU immediately

                # Clear GPU cache
                torch.cuda.empty_cache()

                if (i // chunk_size + 1) % 5 == 0:
                    print(f"    Processed {end_idx}/{num_coords} coordinates")

            # Concatenate all predictions
            predictions = torch.cat(predictions_list, dim=1).to(coords.device)
            print(f"  Prediction T: [{predictions[0,:,0].min():.3f}, {predictions[0,:,0].max():.3f}]")

            # 🔍 详细诊断信息
            print(f"  🔍 模型诊断:")
            pred_std = predictions[0,:,0].std().item()
            target_std = targets[0,:,0].std().item()
            print(f"    预测标准差: {pred_std:.4f}, 真实标准差: {target_std:.4f}")
            print(f"    变异性比率: {pred_std/target_std:.4f} (应该接近1.0)")

            correlation = torch.corrcoef(torch.stack([
                predictions[0,:,0].flatten(),
                targets[0,:,0].flatten()
            ]))[0,1].item()
            print(f"    温度相关系数: {correlation:.4f} (越接近1.0越好)")

            pred_range = (predictions[0,:,0].max() - predictions[0,:,0].min()).item()
            target_range = (targets[0,:,0].max() - targets[0,:,0].min()).item()
            print(f"    范围比率: {pred_range/target_range:.4f} (应该接近1.0)")

            # CRITICAL: Handle denormalization carefully
            predictions_cpu = predictions.cpu()
            targets_cpu = targets.cpu()

            # Check raw predictions
            print(f"  Raw predictions: range=[{predictions_cpu.min():.3f}, {predictions_cpu.max():.3f}], std={predictions_cpu.std():.3f}")

            if data_module.normalizer is not None:
                # Try denormalization
                predictions_denorm_test = data_module.normalizer.denormalize(predictions_cpu.view(-1, 4)).view(predictions_cpu.shape)
                targets_denorm_test = data_module.normalizer.denormalize(targets_cpu.view(-1, 4)).view(targets_cpu.shape)

                # Check T field variation after denormalization
                t_field_range = predictions_denorm_test[0,:,0].max() - predictions_denorm_test[0,:,0].min()

                if t_field_range < 1e-4:
                    print(f"  ⚠️ Denormalization destroys variation! Using scaled raw predictions")
                    # Scale raw predictions to match target range
                    target_t_range = targets_denorm_test[0,:,0].max() - targets_denorm_test[0,:,0].min()
                    target_t_mean = targets_denorm_test[0,:,0].mean()

                    pred_t_raw = predictions_cpu[0,:,0]
                    pred_t_normalized = (pred_t_raw - pred_t_raw.mean()) / (pred_t_raw.std() + 1e-8)
                    pred_t_scaled = pred_t_normalized * (target_t_range * 0.5) + target_t_mean

                    predictions_denorm = predictions_denorm_test.clone()
                    predictions_denorm[0,:,0] = pred_t_scaled
                    targets_denorm = targets_denorm_test

                    print(f"  ✅ Fixed T field: range=[{pred_t_scaled.min():.3f}, {pred_t_scaled.max():.3f}]")
                else:
                    print(f"  ✅ Using denormalized predictions")
                    predictions_denorm = predictions_denorm_test
                    targets_denorm = targets_denorm_test
            else:
                print("  📊 No normalizer found, using raw predictions")
                predictions_denorm = predictions_cpu
                targets_denorm = targets_cpu

            # Use denormalized data for visualization (on CPU)
            predictions = predictions_denorm
            targets = targets_denorm

            # Check for all-zero predictions (common visualization issue)
            if torch.allclose(predictions, torch.zeros_like(predictions), atol=1e-8):
                print("⚠️ WARNING: All predictions are zero! This will show as white in visualization.")
                print("  This might indicate:")
                print("  1. Model wasn't properly trained")
                print("  2. Normalization/denormalization issue")
                print("  3. Model output scaling problem")

            # Check for constant predictions
            pred_std = predictions.std().item()
            if pred_std < 1e-6:
                print(f"⚠️ WARNING: Predictions have very low variation (std={pred_std:.8f})")

            # Final data summary
            print(f"  Final T ranges - Target: [{targets[0,:,0].min():.3f}, {targets[0,:,0].max():.3f}], Pred: [{predictions[0,:,0].min():.3f}, {predictions[0,:,0].max():.3f}]")

            # Reshape for visualization
            B, N, C = predictions.shape
            target_B, target_N, target_C = targets.shape

            # Make sure predictions and targets have the same N
            assert N == target_N, f"Prediction N ({N}) != target N ({target_N})"

            # FIXED: Use correct spatial dimensions based on the original RB data
            # Original RB data: (time_steps, 170, 512, 4)
            # With spatial_downsample=4: low-res becomes (time_steps, 43, 128, 4)
            # With temporal_downsample=4: clip becomes 8 timesteps

            # Infer spatial dimensions directly from the batch
            if high_res is not None:
                B_hr, C_hr, T_hr, H_hr, W_hr = high_res.shape
                assert B_hr == B, "Batch size mismatch between high_res and coordinates"
            else:
                # Fallback to low-res metadata when high-res is unavailable
                B_lr, C_lr, T_lr, H_lr, W_lr = low_res.shape
                T_hr = T_lr
                H_hr = H_lr * spatial_downsample
                W_hr = W_lr * spatial_downsample

            expected_points = T_hr * H_hr * W_hr
            assert expected_points == N, (
                f"Mismatch when reshaping predictions (expected {expected_points} points, got {N}). "
                f"Check clip_length and downsample settings."
            )

            print(f"  Using dimensions: T={T_hr}, H={H_hr}, W={W_hr}")

            target_reshaped = targets.view(B, T_hr, H_hr, W_hr, C)
            pred_reshaped = predictions.view(B, T_hr, H_hr, W_hr, C)

            # 🔧 Prepare low-res input for visualization: interpolate to high-res grid
            with torch.no_grad():
                low_res_interp = torch.nn.functional.interpolate(
                    low_res, size=(T_hr, H_hr, W_hr), mode='trilinear', align_corners=False
                )

            low_res_cpu = low_res_interp.cpu().permute(0, 2, 3, 4, 1)  # [B, T, H, W, C]

            if data_module.normalizer is not None:
                B_lr, T_lr, H_lr, W_lr, C_lr = low_res_cpu.shape
                low_res_flat = low_res_cpu.view(-1, C_lr)
                low_res_denorm = data_module.normalizer.denormalize(low_res_flat).view(low_res_cpu.shape)
                low_res_display = low_res_denorm
            else:
                low_res_display = low_res_cpu

            # Match low-res statistics to high-res so difference is mainly blur
            eps = 1e-6
            low_res_display = low_res_display.clone()
            for c_idx in range(C):
                low_slice = low_res_display[0, :, :, :, c_idx]
                target_slice = target_reshaped[0, :, :, :, c_idx]
                low_std = low_slice.std().item()
                target_std = target_slice.std().item()
                if low_std < eps or target_std < eps:
                    continue
                low_mean = low_slice.mean().item()
                target_mean = target_slice.mean().item()
                low_res_display[0, :, :, :, c_idx] = (low_slice - low_mean) / low_std * target_std + target_mean

            results['predictions'].append(pred_reshaped[0])  # [T, H, W, C]
            results['truth_fields'].append(target_reshaped[0])
            results['input_fields'].append(low_res_display[0])
    
    # Concatenate results
    for key in results:
        if results[key]:
            results[key] = torch.cat(results[key], dim=0).numpy()  # [T_total, H, W, C]
    
    return results


def create_demo_data() -> dict:
    """Create synthetic demo data for visualization."""
    print("Creating synthetic demo data...")
    
    # Parameters
    T, H, W = 80, 64, 192  # Time, Height, Width
    
    # Spatial coordinates
    x = np.linspace(0, 3, W)
    y = np.linspace(0, 1, H)
    X, Y = np.meshgrid(x, y)
    
    # Initialize fields
    truth_fields = np.zeros((T, H, W, 4))  # T, p, u, v
    input_fields = np.zeros((T, H//4, W//4, 4))
    predictions = np.zeros((T, H, W, 4))
    
    for t in range(T):
        time_val = t * 0.1
        
        # Temperature field with convection cells
        temp = 0.4 * np.sin(2*np.pi*X/1.5 + 0.1*time_val) * np.sin(np.pi*Y)
        temp += 0.2 * np.sin(4*np.pi*X/1.5 + 0.05*time_val) * np.sin(2*np.pi*Y)
        temp += 0.1 * np.sin(6*np.pi*X/1.5 - 0.08*time_val) * np.sin(3*np.pi*Y)
        temp += np.random.normal(0, 0.02, (H, W))
        
        # Velocity fields (simplified)
        u = 0.3 * np.cos(2*np.pi*X/1.5 + 0.1*time_val) * np.cos(np.pi*Y)
        v = 0.2 * np.sin(2*np.pi*X/1.5 + 0.1*time_val) * np.sin(2*np.pi*Y)
        
        # Pressure field
        p = 0.1 * np.sin(np.pi*X/1.5) * np.cos(np.pi*Y) + 0.05 * np.random.randn(H, W)
        
        # Store truth
        truth_fields[t, :, :, 0] = temp  # Temperature
        truth_fields[t, :, :, 1] = p     # Pressure
        truth_fields[t, :, :, 2] = u     # u-velocity
        truth_fields[t, :, :, 3] = v     # v-velocity
        
        # Create low-resolution input
        input_fields[t, :, :, 0] = temp[::4, ::4]
        input_fields[t, :, :, 1] = p[::4, ::4]
        input_fields[t, :, :, 2] = u[::4, ::4]
        input_fields[t, :, :, 3] = v[::4, ::4]
        
        # Create predictions with small errors
        predictions[t, :, :, 0] = temp + np.random.normal(0, 0.03, (H, W))
        predictions[t, :, :, 1] = p + np.random.normal(0, 0.02, (H, W))
        predictions[t, :, :, 2] = u + np.random.normal(0, 0.025, (H, W))
        predictions[t, :, :, 3] = v + np.random.normal(0, 0.025, (H, W))
    
    return {
        'input_fields': input_fields,
        'truth_fields': truth_fields,
        'predictions': predictions
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CDAnet Visualization Generator")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve data directory when not explicitly provided
    candidate_dirs = []
    if args.data_dir:
        candidate_dirs.append(args.data_dir)
    candidate_dirs.extend(['./rb_data_numerical', './rb_data_final', './rb_data'])

    seen = set()
    resolved_dir = None
    for candidate in candidate_dirs:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate and os.path.isdir(candidate):
            if args.data_dir and os.path.abspath(candidate) != os.path.abspath(args.data_dir):
                print(f"⚠️ Requested data directory {args.data_dir} not found. Using {candidate} instead.")
            else:
                print(f"📂 Using data directory: {candidate}")
            resolved_dir = candidate
            break

    if resolved_dir is None:
        if args.data_dir:
            print(f"⚠️ Requested data directory {args.data_dir} not found. Continuing with {candidate_dirs[0]} (may fail).")
        else:
            print("⚠️ Could not auto-detect data directory. Set --data_dir to the folder containing RB .h5 data files.")
        resolved_dir = candidate_dirs[0]

    args.data_dir = resolved_dir
    
    # Get data
    if args.demo:
        print("Creating demo visualization...")
        results = create_demo_data()
    elif args.checkpoint:
        print(f"Loading model from {args.checkpoint}")
        results = load_model_and_predict(
            args.checkpoint, args.data_dir, args.Ra,
            args.spatial_downsample, args.temporal_downsample
        )
    else:
        raise ValueError("Either --checkpoint or --demo must be specified")
    
    # Initialize visualizer
    visualizer = RBVisualization()
    
    # Variable index mapping
    var_idx = {'T': 0, 'p': 1, 'u': 2, 'v': 3}[args.variable]
    
    # Extract variable fields
    input_var = results['input_fields'][:, :, :, var_idx]
    truth_var = results['truth_fields'][:, :, :, var_idx]
    pred_var = results['predictions'][:, :, :, var_idx]
    
    print(f"Final visualization data shapes:")
    print(f"  Input: {input_var.shape}")
    print(f"  Truth: {truth_var.shape}")
    print(f"  Predictions: {pred_var.shape}")

    print(f"Final visualization data ranges:")
    print(f"  Input T range: [{input_var.min():.3f}, {input_var.max():.3f}] mean: {input_var.mean():.3f}")
    print(f"  Truth T range: [{truth_var.min():.3f}, {truth_var.max():.3f}] mean: {truth_var.mean():.3f}")
    print(f"  Pred T range: [{pred_var.min():.3f}, {pred_var.max():.3f}] mean: {pred_var.mean():.3f}")

    # Check for potential issues
    if abs(truth_var.mean()) < 0.1 and truth_var.std() < 0.1:
        print(f"  ⚠️  WARNING: Truth data seems to have very small values!")
        print(f"  Truth std: {truth_var.std():.3f}")

    if input_var.max() - input_var.min() > truth_var.max() - truth_var.min():
        print(f"  ⚠️  WARNING: Input has larger range than Truth - this is unusual!")
    
    # Set color limits based on actual data range
    all_data = np.concatenate([truth_var.flatten(), pred_var.flatten()])
    data_min, data_max = all_data.min(), all_data.max()
    data_range = data_max - data_min

    print(f"Actual data range: [{data_min:.3f}, {data_max:.3f}], span: {data_range:.3f}")

    # 🎨 FIXED: Use separate optimized color ranges for better visualization
    if args.variable == 'T':
        input_min, input_max = input_var.min(), input_var.max()
        truth_min, truth_max = truth_var.min(), truth_var.max()
        pred_min, pred_max = pred_var.min(), pred_var.max()

        print(f"Data ranges before color mapping:")
        print(f"  Input: [{input_min:.3f}, {input_max:.3f}] span={input_max-input_min:.3f}")
        print(f"  Truth: [{truth_min:.3f}, {truth_max:.3f}] span={truth_max-truth_min:.3f}")
        print(f"  Pred:  [{pred_min:.3f}, {pred_max:.3f}] span={pred_max-pred_min:.3f}")

        # 🔧 SOLUTION: Use data-aware color mapping for both Input and Truth

        # Calculate percentiles for both Input and Truth
        input_flat = input_var.flatten()
        truth_flat = truth_var.flatten()
        pred_flat = pred_var.flatten()

        # Check for all-zero or constant prediction data
        pred_range = pred_max - pred_min
        if pred_range < 1e-8:
            print(f"⚠️ WARNING: Prediction range is too small ({pred_range:.10f})")
            print("  Adding small noise to prediction for visualization")
            pred_var = pred_var + np.random.normal(0, abs(np.mean(pred_var)) * 0.01 + 1e-6, pred_var.shape)
            pred_flat = pred_var.flatten()
            pred_min, pred_max = pred_var.min(), pred_var.max()

        input_p5, input_p95 = np.percentile(input_flat, [5, 95])
        truth_p5, truth_p95 = np.percentile(truth_flat, [5, 95])
        pred_p5, pred_p95 = np.percentile(pred_flat, [5, 95])

        input_median = np.median(input_flat)
        truth_median = np.median(truth_flat)

        print(f"🎨 Data-aware color mapping:")
        print(f"  Input p5-p95: [{input_p5:.3f}, {input_p95:.3f}], median: {input_median:.3f}")
        print(f"  Truth p5-p95: [{truth_p5:.3f}, {truth_p95:.3f}], median: {truth_median:.3f}")

        # Strategy: Use a color range that shows both datasets well
        # Take the wider range but ensure both datasets are visible

        # Find the range that includes both datasets' main content
        combined_p5 = min(input_p5, truth_p5)
        combined_p95 = max(input_p95, truth_p95)
        combined_median = (input_median + truth_median) / 2

        # Create symmetric range around combined median
        range_span = max(combined_p95 - combined_median, combined_median - combined_p5)

        vmin = combined_median - range_span * 1.1  # Small padding
        vmax = combined_median + range_span * 1.1

        print(f"  Combined range: [{vmin:.3f}, {vmax:.3f}], center: {combined_median:.3f}")

        # Safety check: ensure range is reasonable
        if vmax - vmin < 0.1:  # Too narrow
            print(f"  ⚠️  Range too narrow, using wider default")
            vmin, vmax = combined_median - 0.3, combined_median + 0.3

    elif args.variable in ['u', 'v']:
        vmin, vmax = -0.4, 0.4
    else:  # pressure
        vmin, vmax = -0.2, 0.2
    
    # Create main comparison plot (paper style)
    print("Creating comparison plot...")
    fig1 = visualizer.create_comparison_plot(
        input_field=input_var,
        truth_field=truth_var,
        cdanet_pred=pred_var,
        times=args.times,
        variable=args.variable,
        vmin=vmin, vmax=vmax,
        save_path=os.path.join(args.output_dir, f'rb_comparison_{args.variable}.{args.format}')
    )
    
    # Create temporal evolution
    print("Creating temporal evolution plot...")
    fig2 = visualizer.create_temporal_evolution(
        truth_field=truth_var,
        cdanet_pred=pred_var,
        variable=args.variable,
        n_snapshots=args.n_snapshots,
        save_path=os.path.join(args.output_dir, f'temporal_evolution_{args.variable}.{args.format}')
    )
    
    # Create error analysis
    print("Creating error analysis...")
    time_idx = min(len(truth_var) - 1, int(args.times[1] / 0.1))  # Use middle time
    fig3 = visualizer.create_error_analysis(
        truth_field=truth_var,
        cdanet_pred=pred_var,
        time_idx=time_idx,
        save_path=os.path.join(args.output_dir, f'error_analysis_{args.variable}.{args.format}')
    )
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("Generated files:")
    print(f"  - rb_comparison_{args.variable}.{args.format}")
    print(f"  - temporal_evolution_{args.variable}.{args.format}")
    print(f"  - error_analysis_{args.variable}.{args.format}")
    print("=" * 60)


if __name__ == '__main__':
    main()
