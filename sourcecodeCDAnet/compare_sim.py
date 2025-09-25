#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from scipy.fft import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage
import warnings

###############################################################################
# 1. RBNumericalSimulation (老的雷利-贝纳德模拟器)
###############################################################################

class RBNumericalSimulation:
    def __init__(self, nx=128, ny=64, Lx=3.0, Ly=1.0, Ra=1e5, Pr=0.7, dt=1e-3, save_path='rb_data_numerical'):
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
        
        self.T = np.zeros((ny, nx))  # 温度场
        self.u = np.zeros((ny, nx))  # 水平速度
        self.v = np.zeros((ny, nx))  # 垂直速度
        self.p = np.zeros((ny, nx))  # 压力场
        
        # 存储前两步的历史数据，用于 Adams-Bashforth 时间推进
        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]
        
        # 默认设置：上下边界温度固定，并加小扰动
        self.setup_initial_conditions()
        
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)
    
    def setup_initial_conditions(self):
        """上下边界温度固定，并加随机扰动。"""
        self.T[-1, :] = 1.0
        self.T[0, :] = 0.0
        self.T += 1e-3 * np.random.randn(self.ny, self.nx)
        
        for i in range(2):
            self.T_prev[i] = self.T.copy()
            self.u_prev[i] = self.u.copy()
            self.v_prev[i] = self.v.copy()
    
    def solve_pressure_poisson(self):
        """使用 FFT 方法求解不可压缩条件下的压力泊松方程。"""
        # 构造 source 项 = div(u) / dt
        source = (
            (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2*self.dx) +
            (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2*self.dy)
        ) / self.dt
        
        # 进入频域计算
        source_fft = fft2(source)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx*Kx + Ky*Ky
        K2[0,0] = 1.0  # 防止除 0
        
        p_fft = source_fft / (-K2)
        self.p = np.real(ifft2(p_fft))
    
    def adams_bashforth_step(self, f, f_prev1, f_prev2):
        """Adams-Bashforth 三阶显式时间积分公式。"""
        return f + self.dt * (
            23/12 * f_prev1 - 16/12 * f_prev2 + 5/12 * f
        )
    
    def step(self, step_number):
        """进行一个时间步推进。"""
        # 更新历史数据
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()
        
        # 解压力场
        self.solve_pressure_poisson()
        
        # 更新速度 u
        self.u = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2*self.dx),
            self.u_prev[0],
            self.u_prev[1]
        )
        
        # 更新速度 v，包含浮力项 (Ra * Pr * T)
        self.v = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2*self.dy) +
            self.Ra * self.Pr * self.T,
            self.v_prev[0],
            self.v_prev[1]
        )
        
        # 更新温度 T，包含扩散和对流
        T_update = self.adams_bashforth_step(
            self.Pr * (
                (np.roll(self.T, -1, axis=1) - 2*self.T + np.roll(self.T, 1, axis=1)) / (self.dx*self.dx) +
                (np.roll(self.T, -1, axis=0) - 2*self.T + np.roll(self.T, 1, axis=0)) / (self.dy*self.dy)
            ) - (
                self.u * (np.roll(self.T, -1, axis=1) - np.roll(self.T, 1, axis=1)) / (2*self.dx) +
                self.v * (np.roll(self.T, -1, axis=0) - np.roll(self.T, 1, axis=0)) / (2*self.dy)
            ),
            self.T_prev[0],
            self.T_prev[1]
        )
        
        # 温度限制在 [0, 1]，并保持上下边界
        self.T = np.clip(T_update, 0, 1)
        self.T[0, :] = 0.0
        self.T[-1, :] = 1.0
        
        # 左右边界周期性
        self.u[:, 0] = self.u[:, -1]
        self.u[:, -1] = self.u[:, 0]
        self.v[:, 0] = self.v[:, -1]
        self.v[:, -1] = self.v[:, 0]
        
        # 如果想画图可取消注释
        # self.plot_temperature(step_number)

    def plot_temperature(self, step_number):
        plt.imshow(self.T, cmap='hot', origin='lower', extent=[0, self.Lx, 0, self.Ly])
        plt.colorbar(label='Temperature')
        plt.title('Temperature Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{self.save_path}/temperature_step_{step_number}.png")
        plt.clf()


###############################################################################
# 2. RB2DataLoader (从 .npz 文件加载 (p, b, u, w))
#    为了示例，保留基本结构，但将下采样参数设为默认 1，以“忽略下采样”。
###############################################################################

class RB2DataLoader(Dataset):
    """
    Pytorch Dataset instance for loading a 2D Rayleigh-Bénard dataset from .npz files.
    Channels: [p, b, u, w] => (pressure, buoyancy/temperature, velocity_x, velocity_z).
    """
    def __init__(self, data_dir, data_filename,
                 nx=128, nz=128, nt=16, n_samp_pts_per_crop=1024,
                 downsamp_xz=1, downsamp_t=1,  # 设置默认为 1，不做下采样
                 normalize_output=False, normalize_hres=False,
                 return_hres=False, lres_filter='none', lres_interp='linear', velOnly=False):
        
        self.data_dir = data_dir
        self.data_filename = data_filename if isinstance(data_filename, list) else [data_filename]

        self.nx_hres = nx
        self.nz_hres = nz
        self.nt_hres = nt
        self.n_samp_pts_per_crop = n_samp_pts_per_crop

        # 默认不做下采样
        self.downsamp_xz = downsamp_xz
        self.downsamp_t = downsamp_t
        self.nx_lres = int(nx/downsamp_xz)
        self.nz_lres = int(nz/downsamp_xz)
        self.nt_lres = int(nt/downsamp_t)

        self.normalize_output = normalize_output
        self.normalize_hres = normalize_hres
        self.return_hres = return_hres
        self.lres_filter = lres_filter
        self.lres_interp = lres_interp
        self.velOnly = velOnly

        if lres_filter == 'median':
            warnings.warn("the median filter is very slow...", RuntimeWarning)

        # 读取所有 npz 文件，将其合并为形如 [f, c, t, z, x]
        self.data = RB2DataLoader._read_data_files(self.data_dir, self.data_filename)
        print("Loaded data shape =", self.data.shape)  # [f, c, t, z, x]

        # (f, c, t, z, x) => 取出具体维度
        nf_data, nc_data, nt_data, nz_data, nx_data = self.data.shape

        # 简单检查：必须大于或等于我们需要的分辨率
        if (nx > nx_data) or (nz > nz_data) or (nt > nt_data):
            raise ValueError(
                f'Requested nx={nx}, nz={nz}, nt={nt} exceed actual data shape x={nx_data}, z={nz_data}, t={nt_data}'
            )
        if (nx % downsamp_xz != 0) or (nz % downsamp_xz != 0) or (nt % downsamp_t != 0):
            raise ValueError('nx, nz, nt 必须能被 downsamp_xz, downsamp_t 整除。')

        # 生成所有可能的起始位置
        self.nf_start_range = np.arange(nf_data)
        self.nx_start_range = np.arange(0, nx_data - nx + 1)
        self.nz_start_range = np.arange(0, nz_data - nz + 1)
        self.nt_start_range = np.arange(0, nt_data - nt + 1)

        # 形成网格后展平 => 每种组合对应一次裁剪
        self.rand_grid = np.stack(
            np.meshgrid(self.nf_start_range, self.nt_start_range, self.nz_start_range, self.nx_start_range, indexing='ij'),
            axis=-1
        )
        self.rand_start_id = self.rand_grid.reshape([-1, 4])

        # 记录高分辨率和低分辨率的 [t, z, x]
        self.scale_hres = np.array([self.nt_hres, self.nz_hres, self.nx_hres], dtype=np.int32)
        self.scale_lres = np.array([self.nt_lres, self.nz_lres, self.nx_lres], dtype=np.int32)

        # 统计数据集的均值和方差 (四个通道各自)
        self._mean = np.mean(self.data, axis=(0, 2, 3, 4))
        self._std  = np.std(self.data, axis=(0, 2, 3, 4))

    def __len__(self):
        """返回可裁剪的总数量。"""
        return self.rand_start_id.shape[0]

    def filter(self, signal):
        """
        根据 lres_filter 对 signal 做滤波。若 lres_filter='none'，则不做处理。
        signal: shape [c, t, z, x]
        """
        signal = signal.copy()
        filter_size = [1, self.downsamp_t*2-1, self.downsamp_xz*2-1, self.downsamp_xz*2-1]

        if self.lres_filter == 'none' or (not self.lres_filter):
            output = signal
        elif self.lres_filter == 'gaussian':
            sigma = [0, int(self.downsamp_t/2), int(self.downsamp_xz/2), int(self.downsamp_xz/2)]
            output = ndimage.gaussian_filter(signal, sigma=sigma)
        elif self.lres_filter == 'uniform':
            output = ndimage.uniform_filter(signal, size=filter_size)
        elif self.lres_filter == 'median':
            output = ndimage.median_filter(signal, size=filter_size)
        elif self.lres_filter == 'maximum':
            output = ndimage.maximum_filter(signal, size=filter_size)
        else:
            raise NotImplementedError("Unsupported lres_filter option.")
        return output

    def __getitem__(self, idx):
        """
        返回一个裁剪块 (高分辨率/低分辨率)，以及随机采样的点坐标+对应的物理量。
        注意：在本示例中，我们主要关心高分辨率数据 (p, b, u, w)。
        """
        f_id, t_id, z_id, x_id = self.rand_start_id[idx]
        # 截取一个 [c, t, z, x] 的高分辨率块
        space_time_crop_hres = self.data[f_id, :,
                                         t_id:t_id+self.nt_hres,
                                         z_id:z_id+self.nz_hres,
                                         x_id:x_id+self.nx_hres]  # shape = [c, nt_hres, nz_hres, nx_hres]
        
        # 这一步执行滤波，再用 RegularGridInterpolator 插值到低分辨
        # 如果你不需要低分辨数据，可直接不返回或简化
        space_time_crop_hres_fil = self.filter(space_time_crop_hres)

        interp = RegularGridInterpolator(
            (np.arange(self.nt_hres), np.arange(self.nz_hres), np.arange(self.nx_hres)),
            values=space_time_crop_hres_fil.transpose(1, 2, 3, 0),  # 转置到 [nt, nz, nx, c]
            method=self.lres_interp
        )
        
        # 生成低分辨目标网格坐标
        lres_coord = np.stack(np.meshgrid(
            np.linspace(0, self.nt_hres-1, self.nt_lres),
            np.linspace(0, self.nz_hres-1, self.nz_lres),
            np.linspace(0, self.nx_hres-1, self.nx_lres),
            indexing='ij'
        ), axis=-1)  # shape [nt_lres, nz_lres, nx_lres, 3]
        
        # 插值得到低分辨数据
        space_time_crop_lres = interp(lres_coord).transpose(3, 0, 1, 2)  # [c, nt_lres, nz_lres, nx_lres]

        # 在这个裁剪块内，随机采集 n_samp_pts_per_crop 个点
        # 同样通过插值拿到它们的物理量
        point_coord = np.random.choice(np.linspace(0., 1., 2000), (self.n_samp_pts_per_crop, 3)) \
                      * (self.scale_hres - 1)
        point_value = interp(point_coord)  # shape [n_samp_pts_per_crop, c]
        point_coord = point_coord / (self.scale_hres - 1)  # 归一化到 0~1

        # 如需要对输出做归一化
        if self.normalize_output:
            space_time_crop_lres = self.normalize_grid(space_time_crop_lres)
            point_value = self.normalize_points(point_value)
        if self.normalize_hres:
            space_time_crop_hres = self.normalize_grid(space_time_crop_hres)
        
        if self.velOnly:
            # 如果只关心速度通道
            space_time_crop_lres = space_time_crop_lres[2:4, :, :, :]

        # 最终组织返回
        # 默认返回: (低分辨块, 随机点坐标, 随机点物理量)
        return_tensors = [space_time_crop_lres.astype(np.float32),
                          point_coord.astype(np.float32),
                          point_value.astype(np.float32)]
        
        # 如果需要高分辨的数据一起返回
        if self.return_hres:
            return_tensors = [space_time_crop_hres.astype(np.float32)] + return_tensors
        
        return tuple(return_tensors)

    @staticmethod
    def _read_data_files(data_folder, data_filenames):
        """将多个 .npz 文件中的 (p, b, u, w) 通道拼接为 [f, c, t, z, x]。"""
        all_data = []
        for data_filename in data_filenames:
            fullpath = os.path.join(data_folder, data_filename)
            print("\nLoading data from:", fullpath)
            if not os.path.exists(fullpath):
                raise FileNotFoundError(f"File not found: {fullpath}")

            # 读取 .npz
            npdata = np.load(fullpath)
            # 通道顺序: p, b, u, w
            data = np.stack([npdata['p'], npdata['b'], npdata['u'], npdata['w']], axis=0)  # [4, t, z, x]
            # 变为 [4, t, z, x] => 满足数据格式
            data = data.transpose(0, 1, 2, 3)  # 这里可能不需要变动
            all_data.append(data)

        # 堆叠 => [f, c, t, z, x]
        all_data = np.stack(all_data, axis=0).astype(np.float32)
        return all_data

    # 以下是归一化相关工具
    @property
    def channel_mean(self):
        return self._mean

    @property
    def channel_std(self):
        return self._std

    @staticmethod
    def _normalize_array(array, mean, std):
        return (array - mean) / std

    @staticmethod
    def _denormalize_array(array, mean, std):
        return array * std + mean

    def normalize_grid(self, grid):
        """对 [c, ...] 的网格做通道级归一化。"""
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)] 
        std_bc  = self.channel_std[(...,)+(None,)*(g_dim-1)]
        return self._normalize_array(grid, mean_bc, std_bc)

    def normalize_points(self, points):
        """对 [n_pts, c] 的点数据做通道级归一化。"""
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]
        std_bc  = self.channel_std[(None,)*(g_dim-1)]
        return self._normalize_array(points, mean_bc, std_bc)

    def denormalize_grid(self, grid):
        g_dim = len(grid.shape)
        mean_bc = self.channel_mean[(...,)+(None,)*(g_dim-1)]
        std_bc  = self.channel_std[(...,)+(None,)*(g_dim-1)]
        return self._denormalize_array(grid, mean_bc, std_bc)

    def denormalize_points(self, points):
        g_dim = len(points.shape)
        mean_bc = self.channel_mean[(None,)*(g_dim-1)]
        std_bc  = self.channel_std[(None,)*(g_dim-1)]
        return self._denormalize_array(points, mean_bc, std_bc)


###############################################################################
# 3. 演示：从已有 npz 数据加载前几帧，初始化并继续模拟，然后对比误差
###############################################################################
def main():
    """
    使用 RB2DataLoader 加载一个 .npz 文件的 (p, b, u, w) 数据。
    从中选取若干帧，作为 RBNumericalSimulation 的初始条件继续演算。
    并在每个时间步比较模拟结果与原数据的差距。
    """
    #========== 1) 用户自行修改的参数区域 ==========#
    data_dir = "data"         # 你的 npz 文件所在目录
    data_file = "rb2d_uniform_ra1e6_S_10_chopped.npz"  # 你的 npz 文件名
    nx, nz, nt = 128, 128, 16  # 数据在 x, z, t 维度的裁剪大小
    #============================================#

    # 实例化 DataLoader：默认不做下采样
    dataset = RB2DataLoader(data_dir=data_dir,
                            data_filename=data_file,
                            nx=nx,
                            nz=nz,
                            nt=nt,
                            downsamp_xz=1,
                            downsamp_t=1,
                            return_hres=True,       # 我们想获取高分辨率块
                            normalize_output=False, # 不做归一化
                            normalize_hres=False)

    # 这里只做演示，取索引=0 的裁剪块
    sample = dataset[0]
    # sample => [ space_time_crop_hres, space_time_crop_lres, point_coord, point_value ]
    # 因为 return_hres=True，所以 sample[0] = space_time_crop_hres
    space_time_crop_hres = sample[0]  # shape = [4, nt, nz, nx]

    print("High-res crop shape:", space_time_crop_hres.shape, "(channels, time, z, x)")

    # 通道含义：p=0, b=1, u=2, w=3
    p_data = space_time_crop_hres[0]  # [nt, nz, nx]
    b_data = space_time_crop_hres[1]  # [nt, nz, nx] => 温度场
    u_data = space_time_crop_hres[2]  # [nt, nz, nx] => 水平速度
    w_data = space_time_crop_hres[3]  # [nt, nz, nx] => 垂直速度

    #========== 2) 创建老模拟器并使用 t=2 帧作为初始条件 ==========#
    # RBNumericalSimulation 中 y 对应 nz， x 对应 nx
    sim = RBNumericalSimulation(nx=nx, ny=nz, Lx=3.0, Ly=1.0, Ra=1e6, Pr=0.7, dt=5e-4)

    # 1) 用 data 中的第2帧 作为当前时刻 (index = 2)
    sim.T = b_data[2].copy()
    sim.u = u_data[2].copy()
    sim.v = w_data[2].copy()
    sim.p = p_data[2].copy()

    # 2) 前两步 (prev1, prev2) 分别用 data 的第1帧, 第0帧
    sim.T_prev[0] = b_data[1].copy()
    sim.T_prev[1] = b_data[0].copy()
    sim.u_prev[0] = u_data[1].copy()
    sim.u_prev[1] = u_data[0].copy()
    sim.v_prev[0] = w_data[1].copy()
    sim.v_prev[1] = w_data[0].copy()


    #========== 3) 连续演算并与原数据对比 ==========#
    n_compare_steps = min(nt, 10)  # 比较前 10 步(或不超过 nt)
    print(f"\nStart simulating from loaded data, compare up to {n_compare_steps-1} steps...\n")

    for step_idx in tqdm(range(3, n_compare_steps), desc="Simulating"):
        # 让模拟器前进一步
        sim.step(step_idx)

        # 取原数据中同一时刻 t=step_idx 的真值
        T_true = b_data[step_idx]
        U_true = u_data[step_idx]
        W_true = w_data[step_idx]
        P_true = p_data[step_idx]

        # 计算误差 (MSE)
        mse_T = np.mean((sim.T - T_true)**2)
        mse_U = np.mean((sim.u - U_true)**2)
        mse_V = np.mean((sim.v - W_true)**2)
        mse_P = np.mean((sim.p - P_true)**2)

        print(f"Step={step_idx:02d} | MSE => T:{mse_T:.6f}, U:{mse_U:.6f}, V:{mse_V:.6f}, P:{mse_P:.6f}")

    print("\nSimulation and comparison complete.")


if __name__ == "__main__":
    main()
