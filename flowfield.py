from functools import partial

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

        self.flow_func = partial(double_gyre, **kwargs)

    def plot(self, step: float, lims: np.ndarray, ax) -> None:
        """
        Plots a flow-field for the internal flow_func on a single plot.

        Inputs:
            step: the discretization of the plotted flow-field.
            lims: 4-long numpy array of plotting limits [xmin, xmax, ymin, ymax]
        """

        x1s = np.arange(lims[0], lims[1], step)
        x2s = np.arange(lims[2], lims[3], step)

        x1mesh, x2mesh = np.meshgrid(x1s, x2s)
        dxs = self.flow_func(np.array((x1mesh, x2mesh)))

        ax.quiver(x1mesh, x2mesh, dxs[0], dxs[1])
        plt.draw()


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

    dxs = np.array((dx1s, dx2s))

    return dxs


def main():
    f, ax = plt.subplots()

    g = GyreFlow(flow_model=double_gyre, A=1, s=1, mu=1)
    g.plot(0.1, np.array((-1, 1, -1, 1)), ax)
    plt.show()


if __name__ == "__main__":
    main()
