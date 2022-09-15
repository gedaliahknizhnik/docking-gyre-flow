import pickle

import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True

import numpy as np
from mpl_toolkits import mplot3d

import anglefunctions
from simulation import SimulationOutput, SimulationParams, plot_result

_FILENAME_PREFIX = "rankine_vortex_sim"
_FILENAME_IDENTIFIER = "changing_radius"
_FILENAME_QTY = "50"


def main():

    text_to_add = f"_{_FILENAME_IDENTIFIER}" if _FILENAME_IDENTIFIER is not None else ""
    filename_output_FBRD = (
        f"{_FILENAME_PREFIX}{text_to_add}_FBRD_{_FILENAME_QTY}_outputs.pickle"
    )
    filename_output_Naive = (
        f"{_FILENAME_PREFIX}{text_to_add}_Naive_{_FILENAME_QTY}_outputs.pickle"
    )
    filename_params = (
        f"{_FILENAME_PREFIX}{text_to_add}_FBRD_{_FILENAME_QTY}_params.pickle"
    )

    print(f"Reading data from {filename_output_FBRD}...")
    with open(filename_output_FBRD, "rb") as f:
        outputs_FBRD = pickle.load(f)
    print(f"\tData imported.")

    print(f"Reading data from {filename_output_Naive}...")
    with open(filename_output_Naive, "rb") as f:
        outputs_Naive = pickle.load(f)
    print(f"\tData imported.")

    print(f"Reading data from {filename_params}...")
    with open(filename_params, "rb") as f:
        params = pickle.load(f)
    print(f"\tData imported.")

    print("")

    """Analyze"""
    radii = np.unique(outputs_FBRD[:, 2])
    print(radii)

    f, ax = plt.subplots()

    FBRDs = np.zeros((radii.shape[0], 3))
    Naives = np.zeros((radii.shape[0], 3))

    for ii in range(radii.shape[0]):
        # for radius in radii:
        radius = radii[ii]

        inds = np.where(outputs_FBRD[:, 2] * outputs_FBRD[:, 0] == radius)[0]
        FBRDs[ii] = [
            radius,
            np.mean(outputs_FBRD[inds, 3]),
            np.std(outputs_FBRD[inds, 3]),
        ]

        # ax.scatter(outputs_FBRD[inds, 2], outputs_FBRD[inds, 3], color="blue")

        inds = np.where(outputs_Naive[:, 2] * outputs_Naive[:, 0] == radius)[0]
        Naives[ii] = [
            radius,
            np.mean(outputs_Naive[inds, 3]),
            np.std(outputs_Naive[inds, 3]),
        ]
        # ax.scatter(outputs_Naive[inds, 2], outputs_Naive[inds, 3], color="red")

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    ax.errorbar(
        FBRDs[:, 0],
        FBRDs[:, 1],
        yerr=FBRDs[:, 2],
        color="blue",
        label="FBRD",
        linewidth=2.0,
        capsize=2,
    )
    ax.errorbar(
        Naives[:, 0],
        Naives[:, 1],
        yerr=Naives[:, 2],
        color="red",
        label="Naive",
        linewidth=2.0,
        capsize=2,
    )
    ax.set_xlabel("Initial Radius [m]")
    ax.set_ylabel("Cost [m]")
    ax.legend()
    ax.grid()
    ax.set_aspect('auto')
    plt.show()

    # """Assemble desired results"""
    # for ind, simulation in enumerate(simulations):

    #     # Initial phase difference
    #     _, modboat_init_phase = simulation.flow_model.get_state(
    #         simulation.modboat.pose_hist[0, 1:3]
    #     )
    #     _, struct_init_phase = simulation.flow_model.get_state(
    #         simulation.struct.pose_hist[0, 1:3]
    #     )
    #     init_phase_diff = anglefunctions.wrap_to_pi(
    #         struct_init_phase - modboat_init_phase
    #     )
    #     init_dist = np.linalg.norm(
    #         simulation.modboat.pose_hist[0, 1:3] - simulation.struct.pose_hist[0, 1:3]
    #     )
    #     data[ind] = np.array(
    #         (
    #             simulation.success,
    #             init_phase_diff,
    #             init_dist,
    #             simulation.success_time,
    #         )
    #     )


if __name__ == "__main__":
    main()
