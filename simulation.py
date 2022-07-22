import time

import matplotlib.pyplot as plt
import numpy as np

from controller import ApproachController
from flowfield import GyreFlow, single_vortex
from swimmer import Swimmer


def main():

    # Simulation Parameters
    TOTAL_TIME = 30
    TIMESTEP = 0.01
    INITIAL_POS_MODBOAT = np.array((0.5, 0, 0))
    INITIAL_POS_STRUCTURE = np.array((0, 0.5, 0))
    # Flow field parameter
    FLOW_MODEL = single_vortex
    FLOW_PARAMS = {"Omega": 1, "mu": 0.1}

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
    fig, ax = plt.subplots()
    flow.plot(PLOT_STEP, LIMITS, ax)
    boat.plot(ax)
    strc.plot(ax, "g")
    plt.show()


if __name__ == "__main__":
    main()
