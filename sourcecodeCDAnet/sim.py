import numpy as np
from scipy.fft import fft2, ifft2
import h5py
import os
import matplotlib.pyplot as plt

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
        
        self.T = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        
        self.T_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.u_prev = [np.zeros((ny, nx)) for _ in range(2)]
        self.v_prev = [np.zeros((ny, nx)) for _ in range(2)]
        
        self.setup_initial_conditions()
        
        # Ensure the directory exists
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
            (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1)) / (2*self.dx) +
            (np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / (2*self.dy)
        ) / self.dt
        
        source_fft = fft2(source)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dy)
        Kx, Ky = np.meshgrid(kx, ky)
        K2 = Kx*Kx + Ky*Ky
        K2[0,0] = 1.0
        
        p_fft = source_fft / (-K2)
        self.p = np.real(ifft2(p_fft))
    
    def adams_bashforth_step(self, f, f_prev1, f_prev2):
        return f + self.dt * (
            23/12 * f_prev1 - 16/12 * f_prev2 + 5/12 * f
        )
    
    def step(self, step_number):
        self.T_prev[1] = self.T_prev[0].copy()
        self.T_prev[0] = self.T.copy()
        self.u_prev[1] = self.u_prev[0].copy()
        self.u_prev[0] = self.u.copy()
        self.v_prev[1] = self.v_prev[0].copy()
        self.v_prev[0] = self.v.copy()
        
        self.solve_pressure_poisson()
        
        self.u = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=1) - np.roll(self.p, 1, axis=1)) / (2*self.dx),
            self.u_prev[0],
            self.u_prev[1]
        )
        
        self.v = self.adams_bashforth_step(
            -(np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)) / (2*self.dy) +
            self.Ra * self.Pr * self.T,
            self.v_prev[0],
            self.v_prev[1]
        )
        
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
        
        self.T = np.clip(T_update, 0, 1)
        self.T[0, :] = 0.0
        self.T[-1, :] = 1.0
        
        self.u[:, 0] = self.u[:, -1]
        self.u[:, -1] = self.u[:, 0]
        self.v[:, 0] = self.v[:, -1]
        self.v[:, -1] = self.v[:, 0]
        
        #self.plot_temperature(step_number)

    def plot_temperature(self, step_number):
        plt.imshow(self.T, cmap='hot', origin='lower', extent=[0, self.Lx, 0, self.Ly])
        plt.colorbar(label='Temperature')
        plt.title('Temperature Field')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{self.save_path}/temperature_step_{step_number}.png")
        plt.clf()

def generate_training_dataset(Ra=1e5, n_runs=5, save_path='rb_data_numerical'):
    os.makedirs(save_path, exist_ok=True)
    
    t_start = 5
    t_end = 10
    dt = 1e-3
    delta_t = 0.1
    
    n_steps = int((t_end - t_start) / delta_t)
    
    for run in range(n_runs):
        sim = RBNumericalSimulation(Ra=Ra, save_path=save_path)
        
        n_startup = int(t_start / dt)
        for _ in range(n_startup):
            sim.step(_)
        
        data = []
        steps_per_save = int(delta_t / dt)
        for step_number in range(n_steps):
            for _ in range(steps_per_save):
                sim.step(step_number)
            data.append({
                'temperature': sim.T.copy(),
                'velocity_x': sim.u.copy(),
                'velocity_y': sim.v.copy(),
                'pressure': sim.p.copy()
            })
        
        filename = f'{save_path}/ra_{Ra}_run_{run}.h5'
        with h5py.File(filename, 'w') as f:
            for i, frame in enumerate(data):
                grp = f.create_group(f'frame_{i}')
                for key, value in frame.items():
                    grp.create_dataset(key, data=value)

if __name__ == "__main__":
    Ra_numbers = [1e5]
    
    for Ra in Ra_numbers:
        print(f"Generating numerical simulation data for Ra = {Ra}")
        generate_training_dataset(Ra=Ra, n_runs=5, save_path='rb_data_numerical') 