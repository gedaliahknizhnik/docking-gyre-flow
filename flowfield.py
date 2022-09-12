from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class GyreFlow:
    """Defines a gyre-like flow field that provides velocities for simulation"""

    def __init__(self, *, flow_model: callable, **kwargs) -> None:
        """
        Creates a flow_func function with the keyword arguments passed along.
        We assume that flow_func() takes a set of points and some parameters, and
        create a partial function that takes only the points.

        Inputs:
            flow_model: callable function model
            **kwargs: any parameters for the flow_model.

        Usage:
            g = GyreFlow(flow_model=SOMEMODEL, param1=1, param2=2, ...)
            dpts = g.flow_func(pts)

        """

        # Pull the center out of the kwargs dictionary
        #   [x, y]
        self.center = kwargs.pop("center", np.array((0, 0)))

        # Pull the noise parameters out of the kwargs dictionary
        #   [mu, sigma]
        self.noise_params = kwargs.pop("noise", np.array((0, 0)))

        # All other kwargs go to flow function
        self.flow_func_before_noise = partial(flow_model, **kwargs)

    def flow_func(self, pts: np.ndarray) -> np.ndarray:
        """
        Calls the flow function and adds noise.

        Inputs:
            pts: numpy array containing the points at which to evaluate.
                Supports single points, vectors, or meshgrid, assuming that
                pts[0] contains x1, pts[1] contains x2

        Outputs:
            dxs: numpy array of same shape as pts containing [dx1, dx2, dtheta]
                with noise added in
        """

        dxs = self.flow_func_before_noise(pts)
        dxs += self.noise(dxs)

        return dxs

    def noise(self, dxs: np.ndarray) -> np.ndarray:
        """
        Returns Gaussian noise on the flow parameters

        Inputs:
            dxs: numpy array containing [dx1, dx2, dtheta]

        Outputs:
            noise: numpy array of the same shape as dxs to be added to it.
        """

        return np.random.normal(self.noise_params[0], self.noise_params[1], dxs.shape)

    def plot(self, step: float, lims: np.ndarray, ax, *args, **kwargs) -> None:
        """
        Plots a flow-field for the internal flow_func on a single plot.

        Inputs:
            step: the discretization of the plotted flow-field.
            lims: 4-long numpy array of plotting limits [xmin, xmax, ymin, ymax]
            ax: matplotlib axes object for plotting on
            *args, **kwargs: matplotlib plotting parameters passed
                             directly along to the plot command
        """

        x1s = np.arange(lims[0], lims[1], step)
        x2s = np.arange(lims[2], lims[3], step)

        x1mesh, x2mesh = np.meshgrid(x1s, x2s)
        dxs = self.flow_func(np.array((x1mesh, x2mesh)))

        # Exclude exclusion region
        exclusion = kwargs.pop("exclusion_region", 0)
        for ii in range(len(x1s)):
            for jj in range(len(x2s)):
                dist = (x1mesh[ii, jj] ** 2 + x2mesh[ii, jj] ** 2) ** (1 / 2)
                if dist < exclusion:
                    dxs[:, ii, jj] = 0

        ax.quiver(x1mesh, x2mesh, dxs[0], dxs[1], *args, **kwargs)
        plt.draw()

    def get_state(self, pos: np.ndarray) -> Tuple[float, float]:
        """
        Gets the state (r, theta) for positions in the given flow,
        assuming it is centered at 0.

        Inputs:
            pos: numpy array [x,y,theta] or N x [x,y,theta]

        Outputs:
            r: radii from center
            theta: phase angles in flow

        """

        if len(pos.shape) == 1:
            pos_from_center = pos[0:2] - self.center
            r = np.sqrt(pos_from_center[0] ** 2 + pos_from_center[1] ** 2)
            theta = np.arctan2(pos_from_center[1], pos_from_center[0])
        else:
            pos_from_center = pos[:, 0:2] - self.center
            # r = np.sqrt(pos_from_center[:, 0] ** 2 + pos_from_center[:, 1] ** 2)
            r = np.squeeze(np.linalg.norm(pos_from_center, axis=1))
            theta = np.squeeze(np.arctan2(pos_from_center[:, 1], pos_from_center[:, 0]))

        return r, theta


def double_gyre(pts: np.ndarray, A: float, s: float, mu: float) -> np.ndarray:
    """
    Models the classical wind-driven double-gyre flow model.

    Inputs:
        pts: numpy array containing the points at which to evaluate.
             Supports single points, vectors, or meshgrid, assuming that
             pts[0] contains x1, pts[1] contains x2
        A: circulation amplitude
        s: circulation dimension
        mu: dissipation parameter

    Outputs:
        dxs: flow-field at each point in pts, with shape matching pts.
             dxs[0] contains dx1, dxs[1] contains dx2
    """

    x1s, x2s = pts[0], pts[1]

    factor1 = np.pi * x1s / s
    factor2 = np.pi * x2s / s

    dx1s = -np.pi * A * np.sin(factor1) * np.cos(factor2) - mu * x1s
    dx2s = np.pi * A * np.cos(factor1) * np.sin(factor2) - mu * x2s

    dxs = np.array((dx1s, dx2s, 0 * dx1s))

    return dxs


def single_vortex(pts: np.ndarray, Omega: callable, mu: float) -> np.ndarray:
    """
    Models a vortex flow generated by a spinning blade at a fixed location.
    Inputs:
        pts: numpy array containing the points at which to evaluate.
             Supports single points, vectors, or meshgrid, assuming that
             pts[0] contains x1, pts[1] contains x2
        Omega: angular velocity function of pts # TODO: Make this a function of radius
        mu: dissipation parameter

    Outputs:
        dxs: flow-field at each point in pts, with shape matching pts.
             dxs[0] contains dx1, dxs[1] contains dx2
    """

    x1s, x2s = pts[0], pts[1]

    omega = Omega(x1s, x2s)

    dx1s = -omega * x2s - mu * x1s
    dx2s = omega * x1s - mu * x2s

    dxs = np.array((dx1s, dx2s, 0 * dx1s))

    return dxs


def rankine_velocity(
    x1s: np.ndarray, x2s: np.ndarray, Gamma: float, a: float
) -> np.ndarray:
    """Rankine velocity field - just omegas"""

    rs = np.sqrt(x1s**2 + x2s**2)

    omegas = Gamma / (2 * np.pi) * ((rs / (a**2)) * (rs <= a) + (1 / rs) * (rs > a))

    return omegas


def rankine_vortex(pts: np.ndarray, Gamma: float, a: float) -> np.ndarray:
    """Rankine vortex model - complete"""

    omegas = rankine_velocity(pts[0], pts[1], Gamma, a)

    ths = np.arctan2(pts[1], pts[0])

    dx1s = -omegas * np.sin(ths)
    dx2s = omegas * np.cos(ths)

    dxs = np.array((dx1s, dx2s, 0 * dx1s))

    return dxs


def main():
    f, ax = plt.subplots()

    # g = GyreFlow(flow_model=double_gyre, A=1, s=5, mu=0.001)
    # g.plot(0.1, np.array((0, 5, 0, 5)), ax)
    # plt.show()

    # g = GyreFlow(
    #     flow_model=single_vortex,
    #     **{"Omega": partial(rankine_velocity, Gamma=0.0565, a=0.05), "mu": 0.000}
    # )
    g = GyreFlow(flow_model=rankine_vortex, Gamma=0.0565, a=0.05)
    g.plot(0.1, np.array((-1.5, 1.5, -1.5, 1.5)), ax)
    # plt.show()

    xs = np.arange(-1, 1, 0.01).reshape(-1, 1)
    ys = 0 * xs

    dx1s, dx2s, dths = g.flow_func([xs, ys])
    vels = np.sqrt(dx1s**2 + dx2s**2)

    print(np.hstack((xs, vels)))

    f, ax = plt.subplots()
    plt.plot(xs, vels)
    plt.show()


if __name__ == "__main__":
    main()
