import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True
import numpy as np

import animate
from controller import FBRDController, FlowDirection, FlowOrientation, NaiveController
from flowfield import (
    GyreFlow,
    double_gyre,
    rankine_velocity,
    rankine_vortex,
    single_vortex,
)
from swimmer import Swimmer


@dataclass
class SimulationParams:
    """Class for setting up a convergence simulation"""

    total_time_s: float
    timestep_s: float
    flow_model: callable
    flow_dir: FlowDirection
    flow_ori: FlowOrientation
    flow_params: dict
    gyre_center: np.ndarray = np.array((0, 0))
    plot_limits: np.ndarray = np.array((-1, 1, -1, 1))
    plot_step: float = 0.1


@dataclass
class SimulationProblem:
    """Class for storing complete simulation problem"""

    simulation_params: SimulationParams
    modboat_start_pos: np.ndarray
    struct_start_pos: np.ndarray
    id_number: int
    radius: Optional[float] = 0


@dataclass
class SimulationOutput:
    """Class for storing simulation results"""

    modboat: Swimmer
    struct: Swimmer
    flow_model: GyreFlow
    success: bool
    success_time: float
    id_number: int
    control_cost: float


def main():

    run_simulation_many_rankine_vortex_changing_radius(
        output_name="changing_radius_FBRD"
    )

    return

    # Simulation Parameters
    TOTAL_TIME_S = 60000  # [s]
    TIMESTEP_S = 0.1
    # GYRE_CENTER = np.array((2.5, 2.5))

    # Plotting Parameters
    LIMITS = np.array((-1.5, 1.5, -1.5, 1.5))
    # LIMITS = np.array((0, 5, 0, 5))
    PLOT_STEP = 0.1

    # Flow field parameter
    FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
        rankine_vortex,
        FlowDirection.IN,
        FlowOrientation.CCW,
        {"Gamma": 1.0, "a": 0.05, "noise": np.array([0, 0.001])}
        # {"Gamma": 0.0565, "a": 0.05, "noise": np.array([0, 0.001])},
    )

    RADIUS = 32
    # I want 18cm/s at the radius the boats are at:
    GammaVal = 0.18 * (2 * np.pi) * RADIUS

    # Flow field parameter
    FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
        rankine_vortex,
        FlowDirection.IN,
        FlowOrientation.CCW,
        {"Gamma": GammaVal, "a": 0.05, "noise": np.array([0, 0.00])},
    )

    # FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
    #     single_vortex,
    #     FlowDirection.IN,
    #     FlowOrientation.CCW,
    #     {"Omega": partial(rankine_velocity, Gamma=0.1, a=0.05), "mu": 0.001},
    # )
    # FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
    #     double_gyre,
    #     FlowDirection.IN,
    #     FlowOrientation.CW,
    #     {"A": 1, "s": 5, "mu": 0.001, "center": GYRE_CENTER},
    # )

    sim_params = SimulationParams(
        TOTAL_TIME_S, TIMESTEP_S, FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS
    )

    INITIAL_POS_MODBOAT = np.array((32, 0, 0))
    INITIAL_POS_STRUCTURE = np.array((-32, 0, 0))

    sim_problem = SimulationProblem(
        sim_params, INITIAL_POS_MODBOAT, INITIAL_POS_STRUCTURE, 0
    )

    sim_results, duration, _ = run_simulation(sim_problem)
    print(f"Cost is: {sim_results.control_cost:0.2}")
    plot_result(sim_results, sim_params)


def run_simulation_many_rankine_vortex(output_name: Optional[str] = None) -> None:
    """Runs many simulations on a single gyre"""

    # Simulation Parameters
    TOTAL_TIME_S = 2000  # [s]
    TIMESTEP_S = 0.1
    ITERS = 50

    # Plotting Parameters
    LIMITS = np.array((-1.5, 1.5, -1.5, 1.5))
    PLOT_STEP = 0.1

    # Flow field parameter
    FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
        rankine_vortex,
        FlowDirection.IN,
        FlowOrientation.CCW,
        {"Gamma": 0.0565, "a": 0.05, "noise": np.array([0, 0.00])},
    )

    # Simulation parameters
    sim_params = SimulationParams(
        TOTAL_TIME_S, TIMESTEP_S, FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS
    )

    # Assemble problems15
    print("Assembling simulations...")
    sim_problems = []
    random.seed(5)
    for ii in range(1, ITERS + 1):
        modboat_pt = np.array(
            (
                random.uniform(LIMITS[0], LIMITS[1]),
                random.uniform(LIMITS[2], LIMITS[3]),
                0,
            )
        )
        struct_pt = np.array(
            (
                random.uniform(LIMITS[0], LIMITS[1]),
                random.uniform(LIMITS[2], LIMITS[3]),
                0,
            )
        )
        sim_problems.append(SimulationProblem(sim_params, modboat_pt, struct_pt, ii))

    print("Simulations assembled...")

    print("")
    print("Running simulations...")
    time_start = time.perf_counter()

    sim_results = []
    sim_outputs = np.zeros((ITERS, 2))
    with Pool() as pool:
        results = pool.imap_unordered(run_simulation, sim_problems)

        for simulation_output, duration in results:

            sim_results.append(simulation_output)
            sim_outputs[simulation_output.id_number - 1] = [
                simulation_output.success,
                simulation_output.success_time,
            ]
            print(
                f"\tSimulation {simulation_output.id_number:03} finished in {duration:5.2f} s. Result: {simulation_output.success}"
            )

    print(f"Finished in {time.perf_counter() - time_start:0.2f} s")

    print("")
    print("Saving data to pickle file...")

    file_name_prefix = "rankine_vortex_sim"
    text_to_add = f"_{output_name}" if output_name is not None else ""
    file_name = f"{file_name_prefix}{text_to_add}_{ITERS}"

    with open(f"{file_name}_data.pickle", "wb") as f:
        pickle.dump(sim_results, f)

    with open(f"{file_name}_params.pickle", "wb") as f:
        pickle.dump(sim_params, f)

    print("Data pickled.")


def run_simulation_many_rankine_vortex_changing_radius(
    output_name: Optional[str] = None,
) -> None:
    """Runs many simulations on a single gyre"""

    # Simulation Parameters
    TOTAL_TIME_S = 200000  # [s]
    TIMESTEP_S = 0.1
    ITERS = 50

    # Plotting Parameters
    RADII = np.array((0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0))
    LIMITS = np.array((-1.5, 1.5, -1.5, 1.5))
    PLOT_STEP = 0.1

    TOTAL_ITERS = ITERS * RADII.shape[0]

    # Flow field parameter
    FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
        rankine_vortex,
        FlowDirection.IN,
        FlowOrientation.CCW,
        {"Gamma": 1.0, "a": 0.05, "noise": np.array([0, 0.00])},
    )

    # Simulation parameters
    sim_params = SimulationParams(
        TOTAL_TIME_S, TIMESTEP_S, FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS
    )

    # Assemble problems
    print("Assembling simulations...")
    sim_problems = []
    random.seed(5)

    r_count = 0
    for RADIUS in RADII:
        for ii in range(1, ITERS + 1):
            modboat_r = random.uniform(RADIUS - 0.1, RADIUS + 0.1)
            modboat_th = random.uniform(-np.pi, np.pi)
            modboat_pt = np.array(
                (modboat_r * np.cos(modboat_th), modboat_r * np.sin(modboat_th), 0)
            )

            struct_r = random.uniform(RADIUS - 0.1, RADIUS + 0.1)
            struct_th = random.uniform(-np.pi, np.pi)
            struct_pt = np.array(
                (struct_r * np.cos(struct_th), struct_r * np.sin(struct_th), 0)
            )

            # I want 18cm/s at the radius the boats are at:
            GammaVal = 0.18 * (2 * np.pi) * RADIUS

            # Flow field parameter
            FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS = (
                rankine_vortex,
                FlowDirection.IN,
                FlowOrientation.CCW,
                {"Gamma": GammaVal, "a": 0.05, "noise": np.array([0, 0.00])},
            )

            # Simulation parameters
            sim_params = SimulationParams(
                TOTAL_TIME_S, TIMESTEP_S, FLOW_MODEL, FLOW_DIR, FLOW_ORI, FLOW_PARAMS
            )

            sim_problems.append(
                SimulationProblem(
                    sim_params, modboat_pt, struct_pt, r_count * ITERS + ii, RADIUS
                )
            )

        r_count += 1

    print("Simulations assembled...")

    print("")
    print("Running simulations...")
    time_start = time.perf_counter()

    # sim_results = []
    sim_outputs = np.zeros((TOTAL_ITERS, 4))

    with Pool() as pool:
        results = pool.imap_unordered(run_simulation, sim_problems)

        for simulation_output, duration, radius in results:

            # sim_results.append(simulation_output)
            sim_outputs[simulation_output.id_number - 1] = [
                simulation_output.success,
                simulation_output.success_time,
                radius,
                simulation_output.control_cost,
            ]
            print(
                f"\tSimulation {simulation_output.id_number:03}"
                f" from radius: {radius:0.2}"
                f" finished in {duration:5.2f} s. Result: {simulation_output.success}"
            )

    print(f"Finished in {time.perf_counter() - time_start:0.2f} s")

    print("")
    print("Saving data to pickle file...")

    file_name_prefix = "rankine_vortex_sim"
    text_to_add = f"_{output_name}" if output_name is not None else ""
    file_name = f"{file_name_prefix}{text_to_add}_{ITERS}"

    # with open(f"{file_name}_data.pickle", "wb") as f:
    # pickle.dump(sim_results, f)

    with open(f"{file_name}_outputs.pickle", "wb") as f:
        pickle.dump(sim_outputs, f)

    with open(f"{file_name}_params.pickle", "wb") as f:
        pickle.dump(sim_params, f)

    print("Data pickled.")


def run_simulation(sim_problem: SimulationProblem) -> Tuple[SimulationOutput, float]:

    time_start = time.perf_counter()

    # Pull out input values
    sim_params = sim_problem.simulation_params
    initial_modboat_pos = sim_problem.modboat_start_pos
    initial_structure_pos = sim_problem.struct_start_pos

    # Simulation setup
    iters = int(sim_params.total_time_s / sim_params.timestep_s)
    boat = Swimmer(initial_modboat_pos, iters)
    strc = Swimmer(initial_structure_pos, iters)
    flow = GyreFlow(flow_model=sim_params.flow_model, **sim_params.flow_params)
    cont = FBRDController(
        flow, sim_params.flow_ori, sim_params.flow_dir, sim_params.timestep_s
    )

    result = False
    control_cost = 0

    # Run simulation
    for ii in range(1, iters):
        t = sim_params.timestep_s * ii

        pos_modboat = boat.get_pose()[0:2]
        pos_structure = strc.get_pose()[0:2]

        vel_modboat = flow.flow_func(pos_modboat)
        vel_structure = flow.flow_func(pos_structure)

        vel_control = cont.get_control_vel(pos_modboat, pos_structure)

        boat.update(t, vel_modboat, vel_control)
        strc.update(t, vel_structure)

        # Sum up the cost as the velocity applied perpendicular to the flow.
        new_cost = (
            np.linalg.norm(vel_control - np.dot(vel_control, vel_modboat) * vel_control)
            * sim_params.timestep_s
        )
        control_cost += new_cost

        if cont.evaluate_convergence():
            result = True
            break

    return (
        SimulationOutput(
            boat, strc, flow, result, t, sim_problem.id_number, control_cost
        ),
        time.perf_counter() - time_start,
        sim_problem.radius,
    )


def plot_result(
    sim_out: SimulationOutput,
    sim_params: SimulationParams,
    block: Optional[bool] = True,
    traj_only: Optional[bool] = False,
    exclusion_region: Optional[float] = 0,
) -> None:

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    if traj_only:
        fig, ax = plt.subplots()

        max_x_modboat = np.max(np.abs(sim_out.modboat.pose_hist[:, 1]))
        max_y_modboat = np.max(np.abs(sim_out.modboat.pose_hist[:, 2]))
        max_x_lattice = np.max(np.abs(sim_out.struct.pose_hist[:, 1]))
        max_y_lattice = np.max(np.abs(sim_out.struct.pose_hist[:, 2]))

        maxCoord = np.max(
            np.array((max_x_modboat, max_y_modboat, max_x_lattice, max_y_lattice))
        )

        sim_out.flow_model.plot(
            sim_params.plot_step,
            np.array((-maxCoord, maxCoord, -maxCoord, maxCoord)),
            ax,
            exclusion_region=exclusion_region,
        )

        sim_out.modboat.plot(ax, "b", label="Module")
        sim_out.struct.plot(ax, "g", label="Lattice")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")
        ax.set_aspect("equal")
        ax.legend()

        if block:
            plt.show()
        return

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    sim_out.flow_model.plot(sim_params.plot_step, sim_params.plot_limits, ax1)

    sim_out.modboat.plot(ax1)
    sim_out.struct.plot(ax1, "g")
    ax1.set_xlabel("$x$ [m]")
    ax1.set_ylabel("$y$ [m]")
    ax1.set_aspect("equal")
    plt.show(block=False)

    sim_out.modboat.plotPhase(ax2, flow_model=sim_out.flow_model)
    sim_out.struct.plotPhase(ax2, "g", flow_model=sim_out.flow_model)
    ax2.set_xlabel("$t$ [s]")
    ax2.set_ylabel("$\\theta$ [rad]")
    plt.show(block=False)

    sim_out.modboat.plotRadii(ax4, flow_model=sim_out.flow_model)
    sim_out.struct.plotRadii(ax4, "g", flow_model=sim_out.flow_model)
    ax4.set_xlabel("$t$ [s]")
    ax4.set_ylabel("$r$ [rm]")
    plt.show(block=False)

    # animate.animate_simulation(flow, [boat, strc], LIMITS)
    if block:
        plt.show()


if __name__ == "__main__":
    main()
