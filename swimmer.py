from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from flowfield import GyreFlow


class Swimmer:
    """Defines a point-mass swimmer with 2D position and orientation"""

    def __init__(self, pose: np.ndarray, lifespan: int) -> None:
        """Creates a swimmer.

        Inputs:
            pose: numpy 2D pose [x,y,theta]
            lifespan: int number of simulation steps that will be taken

        """

        self.life = 0
        self.lifespan = lifespan

        self.pose_hist = np.zeros((self.lifespan, 4))  # [t, x, y, theta]
        self.pose_hist[0, 1:] = pose

        self.vel_flow_hist = np.zeros((self.lifespan, 4))  # [t, dx, dy, dtheta]
        self.vel_comd_hist = np.zeros((self.lifespan, 4))

    def get_pose(self) -> np.ndarray:
        """Returns the current pose of the swimmer"""

        return self.pose_hist[self.life, 1:]

    def update(
        self,
        t: float,
        vel: np.ndarray,
        cont_vel: Optional[np.ndarray] = np.array((0, 0, 0)),
    ) -> None:
        """
        Updates the position of the swimmer given a velocity and time.
            Assumes given velocity applied from the last time to the given time.

        Inputs:
            t: time at the update step
            vel: numpy 2D velocity [xdot, ydot, thetadot]
            cont_vel: optional velocity imparted by a controller [xdot, ydot, thetadot]

        """

        if self.life >= self.lifespan - 1:
            raise IndexError("Simulation has gone on too long...")

        # If vel = [dx, dy] -> use vel = [dx, dy, 0]
        if len(vel) == 2:
            vel = np.hstack((vel, 0))

        # Save velocities
        self.vel_flow_hist[self.life, 0] = t
        self.vel_flow_hist[self.life, 1:] = vel
        self.vel_comd_hist[self.life, 0] = t
        self.vel_comd_hist[self.life, 1:] = cont_vel

        # Time difference
        dt = t - self.pose_hist[self.life, 0]

        newPose = self.pose_hist[self.life, 1:] + (vel + cont_vel) * dt

        self.life += 1
        self.pose_hist[self.life, 0] = t
        self.pose_hist[self.life, 1:] = newPose

    def plot(self, ax: plt.Axes, *args, **kwargs) -> None:
        """
        Plots the trajectory of the swimmer on the given axes

        Inputs:
            ax: matplotlib axes object for plotting on
            *args, **kwargs: matplotlib plotting parameters passed
                             directly along to the plot command
        """

        traj = self.pose_hist[: self.life + 1, 1:3]
        ax.plot(traj[:, 0], traj[:, 1], *args, **kwargs)
        plt.draw()

    def plotPhase(self, ax: plt.Axes, *args, **kwargs) -> None:
        """
        Plots the phase of the swimmer on the given axes

        Inputs:
            ax: matplotlib axes object for plotting on
            *args, **kwargs: matplotlib plotting parameters passed
                             directly along to the plot command
        """

        flow_model: GyreFlow = kwargs.pop("flow_model")
        
        ts = self.pose_hist[: self.life + 1, 0]
        traj = self.pose_hist[: self.life + 1, 1:3]
        _, phases = flow_model.get_state(traj)

        ax.plot(ts, phases, *args, **kwargs)
        plt.draw()
        
    def plotRadii(self, ax: plt.Axes, *args, **kwargs) -> None:
        """
        Plots the radius of the swimmer on the given axes

        Inputs:
            ax: matplotlib axes object for plotting on
            *args, **kwargs: matplotlib plotting parameters passed
                             directly along to the plot command
        """

        flow_model: GyreFlow = kwargs.pop("flow_model")
        
        ts = self.pose_hist[: self.life + 1, 0]
        traj = self.pose_hist[: self.life + 1, 1:3]
        rs, _ = flow_model.get_state(traj)

        ax.plot(ts, rs, *args, **kwargs)
        plt.draw()


def main():

    f, ax = plt.subplots()

    s = Swimmer(np.array((0, 0, 0)), 10)
    for i in range(9):
        s.update(i + 1, np.array((0, 1, 0)))

    s.plot(ax, "r-")
    plt.show()
    pass


if __name__ == "__main__":
    main()
