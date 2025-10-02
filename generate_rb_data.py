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

try:
    from sourcecodeCDAnet.sim import RBNumericalSimulation
except ImportError:
    RBNumericalSimulation = None


def initialize_run_parameters(run_id: int, Ra: float, nx: int, ny: int) -> dict:
    """Create smooth, time-coherent parameters for one RB run."""
    rng = np.random.RandomState(run_id)

    n_base_cells = rng.randint(2, 5)
    base_phase = rng.uniform(0, 2 * np.pi)

    temp_modes = []
    psi_modes = []
    for k in range(1, n_base_cells + 1):
        temp_modes.append(dict(
            ax=k,
            amp=0.25 * (1 + 0.05 * rng.randn()) / k,
            phase0=rng.uniform(-np.pi / 4, np.pi / 4),
            omega=0.08 * (1 + 0.1 * rng.randn())
        ))
        psi_modes.append(dict(
            ax=k,
            amp=0.18 * (1 + 0.05 * rng.randn()) / k,
            phase0=rng.uniform(-np.pi / 6, np.pi / 6),
            omega=0.07 * (1 + 0.1 * rng.randn())
        ))

    pressure_modes = []
    for k in range(1, min(3, n_base_cells + 1)):
        pressure_modes.append(dict(
            ax=k,
            amp=0.05 / k,
            phase0=rng.uniform(-np.pi / 3, np.pi / 3),
            omega=0.1 * (1 + 0.05 * rng.randn())
        ))

    # weak noise for slight asymmetry
    noise_amp = 0.005 * (1 + 0.05 * rng.randn())

    return dict(
        temp_modes=temp_modes,
        psi_modes=psi_modes,
        pressure_modes=pressure_modes,
        noise_amp=noise_amp,
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
        T += mode['amp'] * np.sin(np.pi * y_norm) * np.cos(mode['ax'] * np.pi * x_norm + phase)

    # boundary-layer enrichment for mushroom-shaped plumes
    aux_phase = run_params['base_phase'] + 0.4 * t
    T += 0.1 * np.sin(2 * np.pi * y_norm) * np.cos(run_params['base_cells'] * np.pi * x_norm + aux_phase)
    T += run_params['noise_amp'] * np.sin(4 * np.pi * x_norm + 0.7 * aux_phase) * np.sin(3 * np.pi * y_norm)

    # Enforce boundary temperatures
    T[0, :] = 1.0
    T[-1, :] = 0.0

    # --- Stream function & velocities ---
    psi = np.zeros_like(X)
    u = np.zeros_like(X)
    v = np.zeros_like(X)

    for mode in run_params['psi_modes']:
        ax = max(1, mode['ax'])
        phase = mode['phase0'] + mode['omega'] * t
        amp_i = mode['amp']
        sin_y = np.sin(np.pi * y_norm)
        cos_y = np.cos(np.pi * y_norm)
        sin_x = np.sin(ax * np.pi * x_norm + phase)
        cos_x = np.cos(ax * np.pi * x_norm + phase)

        psi += amp_i * sin_y * sin_x
        u += -amp_i * (np.pi / Ly) * cos_y * sin_x
        v += amp_i * (ax * np.pi / Lx) * sin_y * cos_x

    # Enforce boundary conditions
    u[0, :] = u[-1, :] = 0.0
    v[0, :] = v[-1, :] = 0.0
    u[:, 0] = u[:, -1]
    v[:, 0] = v[:, -1]

    # --- Pressure ---
    base_pressure = 0.5 - 0.1 * (T - T.mean())
    p = base_pressure
    for mode in run_params['pressure_modes']:
        ax = max(1, mode['ax'])
        ay = mode['ay']
        phase = mode['phase0'] + mode['omega'] * t
        p += mode['amp'] * np.cos(ax * np.pi * x_norm + phase) * np.cos(ay * np.pi * y_norm)

    return T, u, v, p


def generate_training_dataset_synthetic(Ra=1e5, n_runs=5, n_samples=50, nx=256, ny=256,
                                        save_path='rb_data_numerical', dt=0.05):
    """Generate dataset using the analytic synthetic generator (legacy)."""
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


def generate_paper_style_dataset(Ra=1e5, n_runs=5, nx=192, ny=64, save_path='rb_data_numerical',
                                dt=1e-3, t_start=5.0, t_end=15.0, sample_dt=0.1, pr=0.7):
    """Generate RB data via numerical integrator mimicking Hammoud et al. (2022)."""
    if RBNumericalSimulation is None:
        raise ImportError("RBNumericalSimulation not available. Ensure sourcecodeCDAnet is accessible.")

    os.makedirs(save_path, exist_ok=True)

    print("📐 Numerical RB Data Generation (paper-style)")
    print(f"  Ra = {Ra:.0e}, nx = {nx}, ny = {ny}")
    print(f"  dt = {dt}, warm-up until t = {t_start}, final time = {t_end}, sample_dt = {sample_dt}")

    total_samples = int(np.floor((t_end - t_start) / sample_dt))
    if total_samples <= 0:
        raise ValueError("Invalid sampling configuration: no samples to collect.")

    all_data = []
    warmup_steps = int(round(t_start / dt))
    stride = int(round(sample_dt / dt))
    if abs(stride * dt - sample_dt) > 1e-8:
        raise ValueError("sample_dt must be a multiple of dt for the integrator.")

    for run in range(n_runs):
        print(f"  🏃 Numerical run {run+1}/{n_runs}")
        np.random.seed(run)
        sim = RBNumericalSimulation(nx=nx, ny=ny, Lx=3.0, Ly=1.0, Ra=Ra, Pr=pr, dt=dt, save_path=save_path)

        for step in range(warmup_steps):
            sim.step(step)

        run_frames = []
        current_time = t_start
        for idx in range(total_samples):
            for local in range(stride):
                sim.step(idx * stride + local)
                current_time += dt
            run_frames.append({
                'temperature': sim.T.copy(),
                'velocity_x': sim.u.copy(),
                'velocity_y': sim.v.copy(),
                'pressure': sim.p.copy(),
                'time': current_time
            })

            if (idx + 1) % 10 == 0 or idx == total_samples - 1:
                print(f"      Saved frame {idx+1}/{total_samples} at t = {current_time:.2f}")

        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        with h5py.File(filename, 'w') as f:
            f.attrs['Ra'] = Ra
            f.attrs['Pr'] = pr
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['n_samples'] = total_samples
            f.attrs['run_id'] = run
            f.attrs['dt'] = dt
            for i, frame in enumerate(run_frames):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if key != 'time':
                        grp.create_dataset(key, data=value)
                    else:
                        grp.attrs['time'] = value
        print(f"    ✅ Saved numerical run: {filename}")
        all_data.append(run_frames)

    create_consolidated_dataset(save_path, Ra, all_data, nx, ny, dt=dt, pr=pr)
    return all_data


def create_consolidated_dataset(save_path, Ra, all_data, nx, ny, dt=0.05, pr=0.7):
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
        f.attrs['Pr'] = pr
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
    parser = argparse.ArgumentParser(description='Generate Rayleigh-Bénard datasets')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--n_runs', type=int, default=5, help='Number of runs')
    parser.add_argument('--n_samples', type=int, default=200, help='Snapshots per run (synthetic mode)')
    parser.add_argument('--nx', type=int, default=192, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=64, help='Grid points in y')
    parser.add_argument('--dt', type=float, default=1e-3, help='Integrator time step (numerical mode)')
    parser.add_argument('--t_start', type=float, default=5.0, help='Warm-up time before sampling (numerical mode)')
    parser.add_argument('--t_end', type=float, default=15.0, help='Final snapshot time (numerical mode)')
    parser.add_argument('--sample_dt', type=float, default=0.1, help='Interval between saved frames (numerical mode)')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Save directory')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--mode', type=str, default='paper', choices=['paper', 'synthetic'],
                        help='Generation mode: "paper" uses numerical solver, "synthetic" uses analytic approximation')

    args = parser.parse_args()

    # Clear old data
    if os.path.exists(args.save_path):
        print(f"🗑️  Clearing old data in {args.save_path}")
        import shutil
        shutil.rmtree(args.save_path)

    # Generate stable data
    if args.mode == 'paper':
        all_data = generate_paper_style_dataset(
            Ra=args.Ra,
            n_runs=args.n_runs,
            nx=args.nx,
            ny=args.ny,
            save_path=args.save_path,
            dt=args.dt,
            t_start=args.t_start,
            t_end=args.t_end,
            sample_dt=args.sample_dt
        )
    else:
        all_data = generate_training_dataset_synthetic(
            Ra=args.Ra,
            n_runs=args.n_runs,
            n_samples=args.n_samples,
            nx=args.nx,
            ny=args.ny,
            save_path=args.save_path,
            dt=args.sample_dt
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
