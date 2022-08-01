import time
from functools import partial
from msilib.schema import Control

import matplotlib.pyplot as plt
import numpy as np

import animate
import controller
from controller import ApproachController, FlowDirection, FlowOrientation
from flowfield import GyreFlow, double_gyre, rankine_velocity, single_vortex
from swimmer import Swimmer


def main():

    # Simulation Parameters
    TOTAL_TIME_S = 300  # [s]
    TIMESTEP_S = 0.01
    INITIAL_POS_MODBOAT = np.array((3.5, 3.5, 0))
    INITIAL_POS_STRUCTURE = np.array((3.6, 3.5, 0))
    GYRE_CENTER = np.array((2.5, 2.5))

    # Flow field parameter
    # FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
    #     single_vortex,
    #     FlowDirection.IN,
    #     FlowOrientation.CW,
    #     {"Omega": partial(rankine_velocity, Gamma=0.1, a=0.05), "mu": 0.001},
    # )
    FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
        double_gyre,
        FlowDirection.IN,
        FlowOrientation.CW,
        {"A": 1, "s": 5, "mu": 0.001, "center": GYRE_CENTER},
    )

    # Plotting Parameters
    # LIMITS = np.array((-1, 1, -1, 1))
    LIMITS = np.array((0, 5, 0, 5))
    PLOT_STEP = 0.1

    # Simulation setup
    iters = int(TOTAL_TIME_S / TIMESTEP_S)
    boat = Swimmer(INITIAL_POS_MODBOAT, iters)
    strc = Swimmer(INITIAL_POS_STRUCTURE, iters)
    flow = GyreFlow(flow_model=FLOW_MODEL, **FLOW_PARAMS)
    cont = ApproachController(flow, FLOW_ORI, FLOW_DIR)

    # Run simulation
    for ii in range(1, iters):
        t = TIMESTEP_S * ii

        pos_modboat = boat.get_pose()[0:2]
        pos_structure = strc.get_pose()[0:2]

        vel_modboat = flow.flow_func(pos_modboat)
        vel_structure = flow.flow_func(pos_structure)

        vel_control = cont.get_control_vel(pos_modboat, pos_structure)

        boat.update(t, vel_modboat, vel_control)
        strc.update(t, vel_structure)

        if controller.evaluate_convergence(boat, strc):
            break

        # input()

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    flow.plot(PLOT_STEP, LIMITS, ax1)

    boat.plot(ax1)
    strc.plot(ax1, "g")
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect("equal")
    plt.show(block=False)

    boat.plotPhase(ax2, flow_model=flow)
    strc.plotPhase(ax2, "g", flow_model=flow)
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("$\\theta$ [rad]")
    plt.show(block=False)

    boat.plotRadii(ax4, flow_model=flow)
    strc.plotRadii(ax4, "g", flow_model=flow)
    ax4.set_xlabel("$t$ [s]")
    ax4.set_ylabel("$r$ [rm]")
    plt.show(block=False)

    # animate.animate_simulation(flow, [boat, strc], LIMITS)
    plt.show()


if __name__ == "__main__":
    main()
