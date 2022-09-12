import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import anglefunctions
from simulation import SimulationOutput, SimulationParams, plot_result

_FILENAME_PREFIX = "rankine_vortex_sim"
_FILENAME_IDENTIFIER = None  # "with_noise"
_FILENAME_QTY = None  # "100"


def main():

    text_to_add = f"_{_FILENAME_IDENTIFIER}" if _FILENAME_IDENTIFIER is not None else ""
    qty_to_add = f"_{_FILENAME_QTY}" if _FILENAME_QTY is not None else ""
    filename_data = f"{_FILENAME_PREFIX}{text_to_add}{qty_to_add}_data.pickle"
    filename_params = f"{_FILENAME_PREFIX}{text_to_add}{qty_to_add}_params.pickle"

    print(f"Reading data from {filename_data}...")
    with open(filename_data, "rb") as f:
        simulations = pickle.load(f)
    print(f"\tData imported.")

    print(f"Reading data from {filename_params}...")
    with open(filename_params, "rb") as f:
        params = pickle.load(f)
    print(f"\tData imported.")

    print("")

    data = np.zeros((len(simulations), 4))

    """Assemble desired results"""
    for ind, simulation in enumerate(simulations):

        # Initial phase difference
        _, modboat_init_phase = simulation.flow_model.get_state(
            simulation.modboat.pose_hist[0, 1:3]
        )
        _, struct_init_phase = simulation.flow_model.get_state(
            simulation.struct.pose_hist[0, 1:3]
        )
        init_phase_diff = anglefunctions.wrap_to_pi(
            struct_init_phase - modboat_init_phase
        )
        init_dist = np.linalg.norm(
            simulation.modboat.pose_hist[0, 1:3] - simulation.struct.pose_hist[0, 1:3]
        )
        data[ind] = np.array(
            (
                simulation.success,
                init_phase_diff,
                init_dist,
                simulation.success_time,
            )
        )

    """Plot trajectories one by one"""
    for ind in [12]:  # range(len(simulations)):
        print(f"Simulation {ind}")
        plot_result(
            simulations[ind], params, block=True, traj_only=True, exclusion_region=0.4
        )

    plt.show()


if __name__ == "__main__":
    main()
