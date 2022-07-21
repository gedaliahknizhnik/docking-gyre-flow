import time

import matplotlib.pyplot as plt
import numpy as np

import flowfield
from swimmer import Swimmer


def main():

    # Simulation Parameters
    TOTAL_TIME = 30
    TIMESTEP = 0.01
    INITIAL_POS_MODBOAT = np.array((0.5, 0, 0))
    INITIAL_POS_STRUCTURE = np.array((0, 0.5, 0))
    # Flow field parameter
    FLOW_MODEL = flowfield.single_vortex
    FLOW_PARAMS = {"Omega": 1, "mu": 0.1}

    # Plotting Parameters
    LIMITS = np.array((-1, 1, -1, 1))
    PLOT_STEP = 0.1

    # Simulation setup
    iters = int(TOTAL_TIME / TIMESTEP)
    modboat = Swimmer(INITIAL_POS_MODBOAT, iters)
    structure = Swimmer(INITIAL_POS_STRUCTURE, iters)
    flow = flowfield.GyreFlow(flow_model=FLOW_MODEL, **FLOW_PARAMS)

    # Run simulation
    for ii in range(1, iters):
        t = TIMESTEP * ii

        pos_modboat = modboat.get_pose()[0:2]
        pos_structure = structure.get_pose()[0:2]

        vel_modboat = flow.flow_func(pos_modboat)
        vel_structure = flow.flow_func(pos_structure)

        modboat.update(t, vel_modboat)
        structure.update(t, vel_structure)

    # Plot results
    fig, ax = plt.subplots()
    flow.plot(PLOT_STEP, LIMITS, ax)
    modboat.plot(ax)
    structure.plot(ax, "g")
    plt.show()


if __name__ == "__main__":
    main()
