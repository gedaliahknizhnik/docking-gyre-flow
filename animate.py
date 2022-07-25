import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import flowfield
import swimmer


def animate_simulation(
    flow_model: flowfield.GyreFlow,
    swimmers: List[swimmer.Swimmer],
    axis_limits: np.ndarray,
) -> None:

    arrow_scale = 5

    f, ax = plt.subplots()

    # Plot flow
    flow_model.plot(0.1, axis_limits, ax)

    iterations = swimmers[0].life
    traj_handles, vel_handles, vel_comd_handles = [], [], []

    for swimmer in swimmers:
        traj_handles.append(
            ax.plot(swimmer.pose_hist[0, 1], swimmer.pose_hist[0, 2])[0]
        )

        # vel_handles.append(
        #     ax.quiver(
        #         swimmer.pose_hist[0, 1],
        #         swimmer.pose_hist[0, 2],
        #         swimmer.vel_flow_hist[0, 1],
        #         swimmer.vel_flow_hist[0, 2],
        #         color="blue",
        #     )
        # )
        vel_comd_handles.append(
            ax.plot(
                swimmer.pose_hist[0, 1],
                swimmer.pose_hist[0, 2],
                swimmer.pose_hist[0, 1] + arrow_scale * swimmer.vel_comd_hist[0, 1],
                swimmer.pose_hist[0, 2] + arrow_scale * swimmer.vel_comd_hist[0, 2],
                color="green",
            )[0]
        )

    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("$t = 0$ [s]")
    plt.show(block=False)

    t_start = time.perf_counter()

    ii = 1
    while ii < iterations:

        t = swimmers[0].pose_hist[ii, 0]

        if time.perf_counter() - t_start < t:
            continue

        ax.set_title(f"$t = {t:0.2}$ [s]")

        for swimmer, traj, vel_comd in zip(swimmers, traj_handles, vel_comd_handles):
            traj.set_xdata(swimmer.pose_hist[0:ii, 1])
            traj.set_ydata(swimmer.pose_hist[0:ii, 2])

            vel_comd.set_xdata(
                np.array(
                    (
                        swimmer.pose_hist[ii, 1],
                        swimmer.pose_hist[ii, 1]
                        + arrow_scale * swimmer.vel_comd_hist[ii, 1],
                    )
                )
            )
            vel_comd.set_ydata(
                np.array(
                    (
                        swimmer.pose_hist[ii, 2],
                        swimmer.pose_hist[ii, 2]
                        + arrow_scale * swimmer.vel_comd_hist[ii, 2],
                    )
                )
            )
        f.canvas.draw()
        f.canvas.flush_events()

        ii += 1

    pass
