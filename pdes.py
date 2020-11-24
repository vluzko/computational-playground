import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def fcts_heat_equation_2d(steps: int=100, diff_coeff: float=4.0,
                     width: float=10.0, height: float=10.0,
                     dx: float=0.05, dy: float=0.05,
                     t_cool: float=300.0, t_hot: float=700.0):
    """FTCS heat equation.

    Attributes:
        steps: Number of steps
        diff_coeff: The diffusion coefficient of the material
        width: Width of the object
        height: Height of the object
        dx: The size of the increment on the x-axis
        dy: The size of the increment on the y-axis
        t_cool: Initial cool temperature.
        t_hot: Initial hot temperature
    """

    # Number of divisions
    nx = int(width / dx)
    ny = int(height / dy)

    # The heat equation is second-order in the spatial axes
    dx2, dy2 = dx*dx, dy*dy

    # The size of a time increment
    dt = dx2 * dy2 / (2 * diff_coeff * (dx2 + dy2))

    # Update coefficients
    x_update = dt * diff_coeff / dx2
    y_update = dt * diff_coeff / dy2

    u0 = t_cool * np.ones((nx, ny))
    u = u0.copy()

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r, cx, cy = 2, 5, 5
    r2 = r**2
    for i in range(nx):
        for j in range(ny):
            p2 = (i*dx-cx)**2 + (j*dy-cy)**2
            if p2 < r2:
                u0[i,j] = t_hot

    def do_timestep(u_prev: np.ndarray):
        # Propagate with forward-difference in time, central-difference in space

        # Set it to the current value
        u = u0.copy()

        # x-axis update
        u[1:-1, 1:-1] += (u_prev[2:, 1:-1] - 2 * u_prev[1:-1, 1:-1] + u_prev[:-2, 1:-1]) * x_update
        # y-axis update
        u[1:-1, 1:-1] += (u_prev[1:-1, 2:] - 2 * u_prev[1:-1, 1:-1] + u_prev[1:-1, :-2]) * y_update

        return u

    # Number of timesteps
    # Output 4 figures at these timesteps
    for m in range(steps):
        u = do_timestep(u0)
        u0 = u.copy()
        fig, ax = plt.subplots()

        im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=t_cool, vmax=t_hot)
        ax.set_axis_off()
        ax.set_title('{:.1f} ms'.format(m*dt*1000))
        plt.savefig('plots/pdes/fcts_heat/step_{}.png'.format(m))
        plt.close()


def poissons_equation():
    raise NotImplementedError


def wave_equation():
    raise NotImplementedError


if __name__ == '__main__':
    fcts_heat_equation_2d()
