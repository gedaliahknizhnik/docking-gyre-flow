import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import animate
from controller import ApproachController
from flowfield import GyreFlow, rankine_velocity, single_vortex
from swimmer import Swimmer


def main():

    # Simulation Parameters
    TOTAL_TIME = 180
    TIMESTEP = 0.01
    INITIAL_POS_MODBOAT = np.array((0.5, 0, 0))
    INITIAL_POS_STRUCTURE = np.array((0, 0.5, 0))
    # Flow field parameter
    FLOW_MODEL = single_vortex
    FLOW_PARAMS = {"Omega": partial(rankine_velocity, Gamma=0.1, a=0.05), "mu": 0.01}

    # Plotting Parameters
    LIMITS = np.array((-1, 1, -1, 1))
    PLOT_STEP = 0.1

    # Simulation setup
    iters = int(TOTAL_TIME / TIMESTEP)
    boat = Swimmer(INITIAL_POS_MODBOAT, iters)
    strc = Swimmer(INITIAL_POS_STRUCTURE, iters)
    flow = GyreFlow(flow_model=FLOW_MODEL, **FLOW_PARAMS)
    cont = ApproachController(flow)

    # Run simulation
    for ii in range(1, iters):
        t = TIMESTEP * ii

        pos_modboat = boat.get_pose()[0:2]
        pos_structure = strc.get_pose()[0:2]

        vel_modboat = flow.flow_func(pos_modboat)
        vel_structure = flow.flow_func(pos_structure)

        vel_control = cont.get_control_vel(pos_modboat, pos_structure)

        boat.update(t, vel_modboat, vel_control)
        strc.update(t, vel_structure)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2)
    flow.plot(PLOT_STEP, LIMITS, ax1)
    boat.plot(ax1)
    strc.plot(ax1, "g")
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect("equal")

    plt.show(block=False)

    boat.plotPhase(ax2)
    strc.plotPhase(ax2, "g")
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("$\\theta$ [rad]")
    plt.show(block=False)

    # animate.animate_simulation(flow, [boat, strc], LIMITS)
    plt.show()


if __name__ == "__main__":
    main()
