#!/usr/bin/env python3
"""Create side-by-side animations of low- and high-resolution RB data.

The animation shows:
    - Top row: temperature fields (low-res vs high-res)
    - Bottom row: velocity quiver plots (low-res vs high-res)

Usage example:
    python visualize_rb_data_animation.py \
        --input rb_data_numerical/rb2d_ra1e+05_consolidated.h5 \
        --run 0 \
        --frames 200 \
        --downsample 4 \
        --output rb_animation.mp4
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_args():
    parser = argparse.ArgumentParser(description='Animate RB dataset (low vs high resolution).')
    parser.add_argument('--input', type=str,
                        default='rb_data_numerical/rb2d_ra1e+05_consolidated.h5',
                        help='Path to consolidated RB dataset (HDF5).')
    parser.add_argument('--run', type=int, default=0,
                        help='Simulation run index inside the dataset.')
    parser.add_argument('--downsample', type=int, default=4,
                        help='Spatial downsample factor to create the low-resolution view.')
    parser.add_argument('--frames', type=int, default=200,
                        help='Number of frames to animate (default: 200, or shorter if dataset smaller).')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the animation.')
    parser.add_argument('--output', type=str, default='rb_data_animation.mp4',
                        help='Output animation file (mp4 or gif depending on extension).')
    parser.add_argument('--quiver_step_high', type=int, default=8,
                        help='Stride for high-resolution quiver arrows.')
    parser.add_argument('--quiver_step_low', type=int, default=2,
                        help='Stride for low-resolution quiver arrows (applied after downsampling).')
    parser.add_argument('--cmap', type=str, default='RdBu_r', help='Colormap for temperature fields.')
    parser.add_argument('--dpi', type=int, default=150, help='Output DPI for saved animation frames.')
    return parser.parse_args()


def load_dataset(path, run_idx):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with h5py.File(path, 'r') as f:
        if run_idx >= f['b'].shape[0]:
            raise IndexError(f"Run index {run_idx} out of range (available runs: {f['b'].shape[0]})")

        temperature = f['b'][run_idx]  # [time, ny, nx]
        u = f['u'][run_idx]
        v = f['w'][run_idx]

        dt = f.attrs.get('dt', 0.1)
        Lx = f.attrs.get('Lx', 3.0)
        Ly = f.attrs.get('Ly', 1.0)

    return temperature, u, v, dt, Lx, Ly


def create_animation(args):
    temp_high, u_high, v_high, dt, Lx, Ly = load_dataset(args.input, args.run)

    n_frames_total, ny, nx = temp_high.shape
    frames = min(args.frames, n_frames_total)

    ds = max(1, args.downsample)
    temp_low = temp_high[:, ::ds, ::ds]
    u_low = u_high[:, ::ds, ::ds]
    v_low = v_high[:, ::ds, ::ds]

    ny_low, nx_low = temp_low.shape[1:]

    # Coordinates for imshow extents
    x_high = np.linspace(0, Lx, nx)
    y_high = np.linspace(0, Ly, ny)
    x_low = np.linspace(0, Lx, nx_low)
    y_low = np.linspace(0, Ly, ny_low)

    # Precompute color limits for temperature to keep consistent scale
    temp_min = min(temp_high[:frames].min(), temp_low[:frames].min())
    temp_max = max(temp_high[:frames].max(), temp_low[:frames].max())

    # Create figure layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    (ax_low_temp, ax_high_temp), (ax_low_vel, ax_high_vel) = axes

    # Temperature imshow initial frames
    im_low = ax_low_temp.imshow(temp_low[0], origin='lower', extent=[0, Lx, 0, Ly],
                                vmin=temp_min, vmax=temp_max, cmap=args.cmap, aspect='auto')
    im_high = ax_high_temp.imshow(temp_high[0], origin='lower', extent=[0, Lx, 0, Ly],
                                  vmin=temp_min, vmax=temp_max, cmap=args.cmap, aspect='auto')

    cb_axes = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    cbar = fig.colorbar(im_high, cax=cb_axes)
    cbar.set_label('Temperature')

    ax_low_temp.set_title('Low-res Temperature')
    ax_high_temp.set_title('High-res Temperature')
    for ax in (ax_low_temp, ax_high_temp):
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    # Quiver plots
    qs_high = max(1, args.quiver_step_high)
    qs_low = max(1, args.quiver_step_low)

    X_high, Y_high = np.meshgrid(x_high[::qs_high], y_high[::qs_high])
    X_low, Y_low = np.meshgrid(x_low[::qs_low], y_low[::qs_low])

    quiver_low = ax_low_vel.quiver(X_low, Y_low,
                                   u_low[0, ::qs_low, ::qs_low],
                                   v_low[0, ::qs_low, ::qs_low],
                                   angles='xy', scale_units='xy', scale=None)
    quiver_high = ax_high_vel.quiver(X_high, Y_high,
                                     u_high[0, ::qs_high, ::qs_high],
                                     v_high[0, ::qs_high, ::qs_high],
                                     angles='xy', scale_units='xy', scale=None)

    ax_low_vel.set_title('Low-res Velocity')
    ax_high_vel.set_title('High-res Velocity')
    for ax in (ax_low_vel, ax_high_vel):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect('auto')

    time_text = fig.text(0.5, 0.95, '', ha='center', va='center', fontsize=12)

    def update(frame_idx):
        temp_low_frame = temp_low[frame_idx]
        temp_high_frame = temp_high[frame_idx]
        u_low_frame = u_low[frame_idx]
        v_low_frame = v_low[frame_idx]
        u_high_frame = u_high[frame_idx]
        v_high_frame = v_high[frame_idx]

        im_low.set_data(temp_low_frame)
        im_high.set_data(temp_high_frame)

        quiver_low.set_UVC(u_low_frame[::qs_low, ::qs_low],
                           v_low_frame[::qs_low, ::qs_low])
        quiver_high.set_UVC(u_high_frame[::qs_high, ::qs_high],
                            v_high_frame[::qs_high, ::qs_high])

        current_time = frame_idx * dt
        time_text.set_text(f'Time = {current_time:.2f}')
        return im_low, im_high, quiver_low, quiver_high, time_text

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)

    # Save animation
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"Saving animation to {output_path} (frames={frames}, fps={args.fps})")

    writer_kwargs = {'fps': args.fps, 'dpi': args.dpi}
    ext = os.path.splitext(output_path)[1].lower()

    if ext == '.gif':
        anim.save(output_path, writer='imagemagick', **writer_kwargs)
    else:
        anim.save(output_path, writer='ffmpeg', **writer_kwargs)

    plt.close(fig)
    print('✅ Animation complete!')


if __name__ == '__main__':
    arguments = parse_args()
    create_animation(arguments)
