#!/usr/bin/env python3
"""Numerical Rayleigh–Bénard data generator (based on sourcecodeCDAnet/sim.py)."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, ifft2


class RBNumericalSimulation:
    def __init__(self, nx=128, ny=64, Lx=3.0, Ly=1.0, Ra=1e5, Pr=0.7,
                 dt=5e-4, save_path='rb_data_numerical'):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.Ra = Ra
        self.Pr = Pr
        self.dt = dt
        self.save_path = save_path

        self.dx = Lx / nx
        self.dy = Ly / ny

        self.T = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))

        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]

        self.setup_initial_conditions()
        os.makedirs(self.save_path, exist_ok=True)

    def setup_initial_conditions(self):
        self.T[-1, :] = 1.0
        self.T[0, :] = 0.0
        self.T += 1e-3 * np.random.randn(self.ny, self.nx)

        for i in range(2):
            self.T_prev[i] = self.T.copy()
            self.u_prev[i] = self.u.copy()
            self.v_prev[i] = self.v.copy()

    def solve_pressure_poisson(self):
        source = (
            (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2 * self.dx)
            + (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2 * self.dy)
        ) / self.dt

        source_fft = fft2(source)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx * Kx + Ky * Ky
        K2[0, 0] = 1.0

        p_fft = source_fft / (-K2)
        self.p = np.real(ifft2(p_fft))

    def adams_bashforth_step(self, f, f_prev1, f_prev2):
        return f + self.dt * (23 / 12 * f_prev1 - 16 / 12 * f_prev2 + 5 / 12 * f)

    def step(self, step_number):
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()

        self.solve_pressure_poisson()

        self.u = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2 * self.dx),
            self.u_prev[0],
            self.u_prev[1],
        )

        self.v = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2 * self.dy)
            + self.Ra * self.Pr * self.T,
            self.v_prev[0],
            self.v_prev[1],
        )

        T_update = self.adams_bashforth_step(
            self.Pr
            * (
                (np.roll(self.T, -1, axis=1) - 2 * self.T + np.roll(self.T, 1, axis=1))
                / (self.dx * self.dx)
                + (np.roll(self.T, -1, axis=0) - 2 * self.T + np.roll(self.T, 1, axis=0))
                / (self.dy * self.dy)
            )
            - (
                self.u * (np.roll(self.T, -1, axis=1) - np.roll(self.T, 1, axis=1)) / (2 * self.dx)
                + self.v * (np.roll(self.T, -1, axis=0) - np.roll(self.T, 1, axis=0)) / (2 * self.dy)
            ),
            self.T_prev[0],
            self.T_prev[1],
        )

        self.T = np.clip(T_update, 0, 1)
        self.T[0, :] = 0.0
        self.T[-1, :] = 1.0

        self.u[:, 0] = self.u[:, -1]
        self.u[:, -1] = self.u[:, 0]
        self.v[:, 0] = self.v[:, -1]
        self.v[:, -1] = self.v[:, 0]

    def plot_temperature(self, step_number):
        plt.imshow(self.T, cmap='hot', origin='lower', extent=[0, self.Lx, 0, self.Ly])
        plt.colorbar(label='Temperature')
        plt.title('Temperature Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{self.save_path}/temperature_step_{step_number}.png")
        plt.clf()


def generate_training_dataset(Ra=1e5, n_runs=25, save_path='rb_data_numerical',
                              nx=128, ny=64, dt=5e-4, t_start=25.0,
                              t_end=45.0, sample_dt=0.1, Pr=0.7, visualize=False):
    os.makedirs(save_path, exist_ok=True)

    total_samples = int(np.floor((t_end - t_start) / sample_dt))
    if total_samples <= 0:
        raise ValueError('No samples to collect; adjust time window.')

    warmup_steps = int(round(t_start / dt))
    stride = int(round(sample_dt / dt))
    if abs(stride * dt - sample_dt) > 1e-8:
        raise ValueError('sample_dt must be an integer multiple of dt')

    all_runs = []

    for run in range(n_runs):
        print(f"  🏃 Numerical run {run+1}/{n_runs}")
        sim = RBNumericalSimulation(nx=nx, ny=ny, Lx=3.0, Ly=1.0,
                                    Ra=Ra, Pr=Pr, dt=dt, save_path=save_path)

        for step in range(warmup_steps):
            sim.step(step)

        frames = []
        current_time = t_start
        for idx in range(total_samples):
            for local in range(stride):
                sim.step(idx * stride + local)
                current_time += dt
            frames.append({
                'temperature': sim.T.copy(),
                'velocity_x': sim.u.copy(),
                'velocity_y': sim.v.copy(),
                'pressure': sim.p.copy(),
                'time': current_time
            })
            if (idx + 1) % 10 == 0 or idx == total_samples - 1:
                print(f"      Frame {idx+1}/{total_samples} at t={current_time:.2f}")

        filename = f'{save_path}/rb_data_Ra_{Ra:.0e}_run_{run:02d}.h5'
        with h5py.File(filename, 'w') as f:
            f.attrs['Ra'] = Ra
            f.attrs['Pr'] = Pr
            f.attrs['nx'] = nx
            f.attrs['ny'] = ny
            f.attrs['n_samples'] = total_samples
            f.attrs['run_id'] = run
            f.attrs['dt'] = dt
            for i, frame in enumerate(frames):
                grp = f.create_group(f'frame_{i:03d}')
                for key, value in frame.items():
                    if key == 'time':
                        grp.attrs['time'] = value
                    else:
                        grp.create_dataset(key, data=value)
        print(f"    ✅ Saved: {filename}")
        all_runs.append(frames)

    create_consolidated_dataset(save_path, Ra, all_runs, nx, ny, dt=dt, pr=Pr)

    if visualize:
        viz_file = os.path.join(save_path, f'rb_snapshot_ra{Ra:.0e}.png')
        plot_snapshot(all_runs[0][0], viz_file)
        print(f"✅ Visualization saved: {viz_file}")

    return all_runs


def create_consolidated_dataset(save_path, Ra, all_data, nx, ny, dt=0.1, pr=0.7):
    n_runs = len(all_data)
    n_samples = len(all_data[0]) if all_data else 0
    output_file = os.path.join(save_path, f'rb2d_ra{Ra:.0e}_consolidated.h5')
    print(f"📦 Writing consolidated dataset: {output_file}")

    with h5py.File(output_file, 'w') as f:
        f.attrs['Ra'] = Ra
        f.attrs['Pr'] = pr
        f.attrs['nx'] = nx
        f.attrs['ny'] = ny
        f.attrs['n_runs'] = n_runs
        f.attrs['n_samples'] = n_samples
        f.attrs['dt'] = dt

        temp = np.zeros((n_runs, n_samples, ny, nx), dtype=np.float32)
        u = np.zeros_like(temp)
        v = np.zeros_like(temp)
        p = np.zeros_like(temp)

        for run_idx, run_frames in enumerate(all_data):
            for frame_idx, frame in enumerate(run_frames):
                temp[run_idx, frame_idx] = frame['temperature']
                u[run_idx, frame_idx] = frame['velocity_x']
                v[run_idx, frame_idx] = frame['velocity_y']
                p[run_idx, frame_idx] = frame['pressure']

        f.create_dataset('b', data=temp, compression='gzip')
        f.create_dataset('u', data=u, compression='gzip')
        f.create_dataset('w', data=v, compression='gzip')
        f.create_dataset('p', data=p, compression='gzip')


def plot_snapshot(frame, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    cm = axes[0, 0].imshow(frame['temperature'], origin='lower', aspect='auto')
    plt.colorbar(cm, ax=axes[0, 0]); axes[0, 0].set_title('Temperature')
    axes[0, 1].imshow(frame['pressure'], origin='lower', aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('Pressure')
    axes[1, 0].imshow(frame['velocity_x'], origin='lower', aspect='auto', cmap='coolwarm')
    axes[1, 0].set_title('Velocity X')
    axes[1, 1].imshow(frame['velocity_y'], origin='lower', aspect='auto', cmap='coolwarm')
    axes[1, 1].set_title('Velocity Y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Numerical Rayleigh–Bénard data generator')
    parser.add_argument('--Ra', type=float, default=1e5, help='Rayleigh number')
    parser.add_argument('--Pr', type=float, default=0.7, help='Prandtl number')
    parser.add_argument('--n_runs', type=int, default=25, help='Number of independent runs')
    parser.add_argument('--nx', type=int, default=128, help='Grid points in x')
    parser.add_argument('--ny', type=int, default=64, help='Grid points in y')
    parser.add_argument('--dt', type=float, default=5e-4, help='Integrator time step')
    parser.add_argument('--t_start', type=float, default=25.0, help='Warm-up time before sampling')
    parser.add_argument('--t_end', type=float, default=45.0, help='Final sampling time')
    parser.add_argument('--sample_dt', type=float, default=0.1, help='Interval between saved frames')
    parser.add_argument('--save_path', type=str, default='rb_data_numerical', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Save a sample snapshot figure')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('🚀 Starting numerical RB data generation...')
    generate_training_dataset(
        Ra=args.Ra,
        Pr=args.Pr,
        n_runs=args.n_runs,
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        t_start=args.t_start,
        t_end=args.t_end,
        sample_dt=args.sample_dt,
        save_path=args.save_path,
        visualize=args.visualize,
    )
