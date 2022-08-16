import pickle

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

import anglefunctions
from simulation import SimulationOutput, SimulationParams, plot_result

_FILENAME_DATA = "rankine_vortex_sim_data.pickle"
_FILENAME_PARAMS = "rankine_vortex_sim_params.pickle"


def main():

    print(f"Reading data from {_FILENAME_DATA}...")
    with open(_FILENAME_DATA, "rb") as f:
        simulations = pickle.load(f)
    print(f"\tData imported.")

    print(f"Reading data from {_FILENAME_PARAMS}...")
    with open(_FILENAME_PARAMS, "rb") as f:
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

    """Overall success percentage"""
    successes = np.sum(data[:, 0] > 0)
    success_inds = np.where(data[:, 0] == 1)[0]
    failure_inds = np.where(data[:, 0] == 0)[0]

    print(f"{successes}/{data.shape[0]} = {100*successes/data.shape[0]:0.2f} % success")
    print(f"\tFailed tests: {failure_inds}")

    """Average success time vs phase difference"""
    fig, ax = plt.subplots()
    plt.scatter(data[success_inds, 1], data[success_inds, 3])
    plt.xlabel("$\Delta\phi_0$ [rad]")
    plt.ylabel("$t_{converge}$ [s]")
    plt.show(block=False)

    """Average success time vs position difference"""
    fig, ax = plt.subplots()
    plt.scatter(data[success_inds, 2], data[success_inds, 3])
    plt.xlabel("$\Delta x_0$ [rad]")
    plt.ylabel("$t_{converge}$ [s]")
    plt.show(block=False)

    """3D view"""
    ax = plt.axes(projection="3d")
    xdata = data[success_inds, 1]
    ydata = data[success_inds, 2]
    zdata = data[success_inds, 3]
    ax.scatter3D(xdata, ydata, zdata, c=xdata)
    ax.set_xlabel("$\Delta\phi_0$ [rad]")
    ax.set_ylabel("$\Delta x_0$ [rad]")
    ax.set_zlabel("$t_{converge}$ [s]")

    """Binned"""
    bins = np.linspace(-np.pi / 2, np.pi / 2, 6)
    digitized = np.digitize(xdata, bins)
    fig, ax = plt.subplots()
    c = np.random.rand(len(bins))

    fitydata = np.sort(ydata)

    for ii in range(len(bins)):
        sc = ax.scatter(ydata[digitized == ii], zdata[digitized == ii])
        p = np.polyfit(ydata[digitized == ii], zdata[digitized == ii], 1)
        predict = np.poly1d(p)
        fitzdata = predict(fitydata)
        ax.plot(fitydata, fitzdata, color=np.squeeze(sc._facecolors))

    ax.set_xlabel("$\Delta x_0$ [rad]")
    ax.set_ylabel("$t_{converge}$ [s]")
    plt.show(block=False)

    """Plot failed trajectories"""
    for failure_ind in failure_inds[: max(10, len(failure_inds))]:
        plot_result(simulations[failure_ind], params, block=False)
        plt.suptitle(f"Failed simulation {failure_ind}")

    plt.show()


if __name__ == "__main__":
    main()
