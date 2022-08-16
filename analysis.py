import pickle

import matplotlib.pyplot as plt
import numpy as np

from simulation import SimulationOutput

_FILENAME = "single_gyre_sim_data.pickle"


def main():

    print(f"Reading data from {_FILENAME}...")
    with open(_FILENAME, "rb") as f:
        simulations = pickle.load(f)
    print(f"Data imported.")

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
        init_phase_diff = struct_init_phase - modboat_init_phase
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

    """Overall success percentage"""
    successes = np.sum(data[:, 0] > 0)
    print(f"{successes}/{data.shape[0]} = {100*successes/data.shape[0]:0.2f} % success")
    print(f"\tFailed tests: {np.where(data[:,0] == 0)[0]}")

    """Average success time vs phase difference"""
    fig, ax = plt.subplots()
    plt.scatter(data[:, 1], data[:, 3])
    plt.xlabel("$\Delta\phi_0$ [rad]")
    plt.ylabel("$t_{converge}$ [s]")
    plt.show(block=False)

    fig, ax = plt.subplots()
    plt.scatter(data[:, 2], data[:, 3])
    plt.xlabel("$\Delta x_0$ [rad]")
    plt.ylabel("$t_{converge}$ [s]")
    plt.show(block=False)

    plt.show()

    i = 1


if __name__ == "__main__":
    main()
