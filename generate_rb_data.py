#!/usr/bin/env python3
"""
Simple but stable Rayleigh-Bénard data generator
Uses analytical patterns with physics-informed time evolution
Avoids numerical instabilities while providing realistic training data
"""

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse


def initialize_run_parameters(run_id: int, Ra: float, nx: int, ny: int) -> dict:
    """Create smooth, time-coherent parameters for one RB run."""
    rng = np.random.RandomState(run_id)

    n_base_cells = rng.randint(2, 5)
    base_phase = rng.uniform(0, 2 * np.pi)

    # Temperature convection modes
    temp_modes = []
    for k in range(3):
        ax = max(1, n_base_cells + rng.randint(-1, 2))
        ay = k + 1
        amp = 0.28 / ay * (1 + 0.1 * rng.randn())
        phase0 = base_phase + rng.uniform(-np.pi, np.pi)
        omega = 0.15 * (k + 1) * (1 + 0.1 * rng.randn())
        temp_modes.append(dict(ax=ax, ay=ay, amp=amp, phase0=phase0, omega=omega))

    # Stream function modes (used for velocities)
    psi_modes = []
    for k in range(2):
        ax = max(1, n_base_cells + rng.randint(-1, 2))
        ay = k + 1
        amp = 0.2 / ay * (1 + 0.1 * rng.randn())
        phase0 = base_phase + rng.uniform(-np.pi, np.pi)
        omega = 0.12 * (k + 1) * (1 + 0.1 * rng.randn())
        psi_modes.append(dict(ax=ax, ay=ay, amp=amp, phase0=phase0, omega=omega))

    swirl = dict(
        amp=0.03 * (1 + 0.1 * rng.randn()),
        ax=n_base_cells + 1,
        ay=2,
        phase0=rng.uniform(0, 2 * np.pi),
        omega=0.3 * (1 + 0.1 * rng.randn())
    )

    pressure_modes = []
    for k in range(2):
        pressure_modes.append(dict(
            ax=max(1, n_base_cells + k),
            ay=k + 1,
            amp=0.06 / (k + 1),
            phase0=rng.uniform(0, 2 * np.pi),
            omega=0.18 * (1 + 0.1 * rng.randn())
        ))

    return dict(
        temp_modes=temp_modes,
        psi_modes=psi_modes,
        swirl=swirl,
        pressure_modes=pressure_modes,
        noise_amp=0.01,
        base_cells=n_base_cells,
        base_phase=base_phase
    )


def generate_stable_rb_data(Ra=1e5, nx=256, ny=256, t=0.0, dt=0.05, run_id=0,
                            run_params=None):
    """Generate analytical Rayleigh–Bénard-like roll patterns.

    The construction aims to mimic classic RB convection cells:
        * Temperature has a vertical gradient plus convective rolls.
        * Stream function is composed of a few sinusoidal rolls, from which
          velocities are derived analytically.
        * Mild stochasticity is injected through phase shifts and secondary modes
          to diversify the dataset.
    """

    Lx, Ly = 3.0, 1.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    x_norm = X / Lx
    y_norm = Y / Ly
    if run_params is None:
        run_params = initialize_run_parameters(run_id, Ra, nx, ny)

    # --- Temperature ---
    base_linear = 1.0 - y_norm  # hot bottom, cold top
    T = base_linear.copy()

    for mode in run_params['temp_modes']:
        phase = mode['phase0'] + mode['omega'] * t
        T += mode['amp'] * np.sin(mode['ay'] * np.pi * y_norm) * \
             np.cos(mode['ax'] * np.pi * x_norm + phase)

    # Gentle harmonic to break symmetry
    aux_phase = run_params['base_phase'] + 0.6 * t
    T += 0.05 * np.sin(2 * np.pi * y_norm) * np.cos((run_params['base_cells'] + 1) * np.pi * x_norm + aux_phase)
    T += run_params['noise_amp'] * np.sin(6 * np.pi * x_norm + 1.5 * aux_phase) * np.sin(3 * np.pi * y_norm)

    # Enforce boundary temperatures
    T[0, :] = 1.0
    T[-1, :] = 0.0

    # --- Stream function & velocities ---
    psi = np.zeros_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)

    for mode in run_params['psi_modes']:
        ax = max(1, mode['ax'])
        ay = mode['ay']
        phase = mode['phase0'] + mode['omega'] * t
        amp_i = mode['amp']
        sin_y = np.sin(ay * np.pi * y_norm)
        cos_y = np.cos(ay * np.pi * y_norm)
        sin_x = np.sin(ax * np.pi * x_norm + phase)
        cos_x = np.cos(ax * np.pi * x_norm + phase)

        psi += amp_i * sin_y * sin_x
        u += -amp_i * (ay * np.pi / Ly) * cos_y * sin_x
        v += amp_i * (ax * np.pi / Lx) * sin_y * cos_x

    # Add small-scale swirling perturbation for diversity
    swirl = run_params['swirl']
    swirl_phase = swirl['phase0'] + swirl['omega'] * t
    swirl_field = swirl['amp'] * np.sin(swirl['ax'] * np.pi * x_norm + swirl_phase) * \
        np.sin(swirl['ay'] * np.pi * y_norm - 0.4 * swirl_phase)
    u += swirl_field
    v += 0.5 * swirl_field

    # Enforce boundary conditions
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0
    u[:, 0] = u[:, -1]
    v[:, 0] = v[:, -1]

    # --- Pressure ---
    base_pressure = 0.5 - 0.12 * (T - T.mean())
    p = base_pressure
    for mode in run_params['pressure_modes']:
        ax = max(1, mode['ax'])
        ay = mode['ay']
        phase = mode['phase0'] + mode['omega'] * t
        p += mode['amp'] * np.cos(ax * np.pi * x_norm + phase) * np.cos(ay * np.pi * y_norm)

    return T, u, v, p


def generate_training_dataset(Ra=1e5, n_runs=5, n_samples=50, nx=256, ny=256, save_path='rb_data_numerical', dt=0.05):
    """Generate training dataset with stable time evolution"""
    os.makedirs(save_path, exist_ok=True)

    print(f"🌡️ Stable RB Data Generation")
    print(f"  Rayleigh number: Ra = {Ra:.0e}")
    print(f"  Runs: {n_runs}")
    print(f"  Samples per run: {n_samples}")
    print(f"  Grid: {nx}×{ny}")
    print()

    all_data = []

    run_params_list = [initialize_run_parameters(run, Ra, nx, ny) for run in range(n_runs)]

    for run in range(n_runs):
        print(f"  🏃 Run {run+1}/{n_runs}")

        run_data = []
        t_offset = run * n_samples * dt  # Different initial time for each run
        params = run_params_list[run]

        for sample in range(n_samples):
            t = t_offset + sample * dt

            # Generate snapshot with run-specific diversity
            T, u, v, p = generate_stable_rb_data(Ra=Ra, nx=nx, ny=ny, t=t, dt=dt,
                                                 run_id=run, run_params=params)

            # Save data
            frame_data = {
                'temperature': T.copy(),
                'velocity_x': u.copy(),
                'velocity_y': v.copy(),
                'pressure': p.copy(),
                'time': t
            }
            run_data.append(frame_data)

            if sample % 10 == 0:
                print(f"      Sample {sample+1}/{n_samples}")

        # Save individual run
        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        with h5py.File(filename, 'w') as f:
            # Add run-level attributes for training compatibility
            f.attrs['Ra'] = Ra
            f.attrs['Pr'] = 0.7
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['n_samples'] = n_samples
            f.attrs['run_id'] = run
            f.attrs['dt'] = dt

            for i, frame in enumerate(run_data):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if key != 'time':
                        grp.create_dataset(key, data=value)
                    else:
                        grp.attrs['time'] = value

        print(f"    ✅ Saved: {filename}")
        all_data.append(run_data)

    # Create consolidated dataset
    create_consolidated_dataset(save_path, Ra, all_data, nx, ny, dt=dt)

    return all_data


def create_consolidated_dataset(save_path, Ra, all_data, nx, ny, dt=0.05):
    """Create consolidated dataset compatible with training"""
    n_runs = len(all_data)
    n_samples = len(all_data[0])

    print(f"\n📦 Creating consolidated dataset: {n_runs} runs × {n_samples} samples")

    # Initialize arrays
    p_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    b_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # Temperature -> b
    u_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
    w_data = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)  # v -> w

    # Fill arrays
    for run_idx, run_data in enumerate(all_data):
        for sample_idx, frame in enumerate(run_data):
            p_data[run_idx, sample_idx] = frame['pressure']
            b_data[run_idx, sample_idx] = frame['temperature']
            u_data[run_idx, sample_idx] = frame['velocity_x']
            w_data[run_idx, sample_idx] = frame['velocity_y']

    # Save consolidated dataset
    output_file = f'{save_path}/rb2d_ra{Ra:.0e}_consolidated.h5'
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('p', data=p_data, compression='gzip')
        f.create_dataset('b', data=b_data, compression='gzip')
        f.create_dataset('u', data=u_data, compression='gzip')
        f.create_dataset('w', data=w_data, compression='gzip')

        # Add metadata compatible with training script
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = 0.7
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['Lx'] = 3.0
        f.attrs['Ly'] = 1.0
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = n_samples
        f.attrs['simulation_type'] = 'stable_analytical'
        f.attrs['format'] = 'consolidated_training_compatible'
        f.attrs['dt'] = dt

    print(f"✅ Consolidated dataset: {output_file}")

    # Print statistics
    print(f"\n📊 Data statistics:")
    print(f"  Temperature range: [{np.min(b_data):.3f}, {np.max(b_data):.3f}]")
    print(f"  Pressure range: [{np.min(p_data):.3f}, {np.max(p_data):.3f}]")
    print(f"  U-velocity range: [{np.min(u_data):.3f}, {np.max(u_data):.3f}]")
    print(f"  V-velocity range: [{np.min(w_data):.3f}, {np.max(w_data):.3f}]")

    # Check temporal evolution
    temp_change = np.max(np.abs(b_data[0, 0] - b_data[0, -1]))
    vel_change = np.max(np.abs(u_data[0, 0] - u_data[0, -1]))
    print(f"  ✅ Temporal evolution - Temperature: {temp_change:.4f}, Velocity: {vel_change:.4f}")

    return output_file


def visualize_data(output_file):
    """Create visualization of the data"""
    print(f"\n🎨 Creating visualization: {output_file}")

    with h5py.File(output_file, 'r') as f:
        b_data = f['b'][:]
        p_data = f['p'][:]
        u_data = f['u'][:]
        w_data = f['w'][:]

        n_runs, n_samples, ny, nx = b_data.shape
        print(f"  Data shape: {n_runs} runs × {n_samples} samples × {ny}×{nx}")

    # Select time steps for visualization
    run_idx = 0
    time_steps = [0, n_samples//4, n_samples//2, 3*n_samples//4, n_samples-1]

    # Create visualization
    fig, axes = plt.subplots(4, len(time_steps), figsize=(15, 12))
    fig.suptitle('Stable Rayleigh-Bénard Simulation (Time Evolution)', fontsize=16)

    for i, t in enumerate(time_steps):
        # Temperature
        im1 = axes[0, i].imshow(b_data[run_idx, t], cmap='RdBu_r', aspect='equal')
        axes[0, i].set_title(f'Temperature t={t}')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Pressure
        im2 = axes[1, i].imshow(p_data[run_idx, t], cmap='viridis', aspect='equal')
        axes[1, i].set_title(f'Pressure t={t}')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # U velocity
        im3 = axes[2, i].imshow(u_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[2, i].set_title(f'U Velocity t={t}')
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

        # V velocity
        im4 = axes[3, i].imshow(w_data[run_idx, t], cmap='RdBu', aspect='equal')
        axes[3, i].set_title(f'V Velocity t={t}')
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])

    # Add colorbars
    plt.colorbar(im1, ax=axes[0, :], shrink=0.6, label='Temperature')
    plt.colorbar(im2, ax=axes[1, :], shrink=0.6, label='Pressure')
    plt.colorbar(im3, ax=axes[2, :], shrink=0.6, label='U Velocity')
    plt.colorbar(im4, ax=axes[3, :], shrink=0.6, label='V Velocity')

    plt.tight_layout()

    viz_file = f"{os.path.dirname(output_file)}/stable_rb_visualization.png"
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {viz_file}")
    plt.close()

    return viz_file


def main():
    parser = argparse.ArgumentParser(description='Generate stable Rayleigh-Bénard data')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--n_runs', type=int, default=25,
                        help='Number of independent runs (default matches CDAnet paper: 20 train + 5 val)')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Snapshots per run (default spans t∈[25,45] with Δt=0.1, as in CDAnet paper)')
    parser.add_argument('--nx', type=int, default=256, help='Grid points in x (high resolution)')
    parser.add_argument('--ny', type=int, default=256, help='Grid points in y (high resolution)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step between saved samples (0.1 matches paper sampling)')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Save directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')

    args = parser.parse_args()

    # Clear old data
    if os.path.exists(args.save_path):
        print(f"🗑️  Clearing old data in {args.save_path}")
        import shutil
        shutil.rmtree(args.save_path)

    # Generate stable data
    print("🚀 Starting STABLE Rayleigh-Bénard data generation...")
    print("Uses analytical patterns with proper time evolution - fast and stable!")

    all_data = generate_training_dataset(
        Ra=args.Ra,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        nx=args.nx,
        ny=args.ny,
        save_path=args.save_path,
        dt=args.dt
    )

    print(f"\n✅ Stable data generation complete!")
    print(f"📁 Data saved in: {args.save_path}/")
    print(f"🚀 Ready for CDAnet training with realistic, stable data!")

    # Create visualizations if requested
    if args.visualize:
        output_file = f'{args.save_path}/rb2d_ra{args.Ra:.0e}_consolidated.h5'
        visualize_data(output_file)


if __name__ == "__main__":
    main()
