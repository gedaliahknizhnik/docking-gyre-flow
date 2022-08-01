from enum import Enum

import numpy as np

import anglefunctions
import flowfield
import swimmer


class FlowDirection(Enum):
    IN = -1
    OUT = 1


class FlowOrientation(Enum):
    CW = -1
    CCW = 1


class ApproachController:
    """
    Models a controller that drives a mobile boat towards a target boat using the flow.
    """

    def __init__(
        self,
        flow_model: flowfield.GyreFlow,
        flowOri: FlowOrientation,
        flowDir: FlowDirection,
    ) -> None:
        """Creates a controller object by storing the flowfield and setting params"""

        # Control params
        self.Kp = 0.5
        self.vel_from_thrusters = 0.01

        # Flow model
        self.flow_model = flow_model
        self.flowOri = flowOri
        self.flowDir = flowDir

    def get_control_vel(self, pos: np.ndarray, targ: np.ndarray) -> np.ndarray:
        """
        Gets the velocity to apply to the mobile boat. The controller looks like:

            r_des = r_targ + Kp*phase_error

        and the applied velocity is either out or in along the phase direction.

        Inputs:
            pos: numpy [x, y, theta] pose of the mobile swimmer
            targ: numpy [x, y, theta] pose of the target swimmer

        Outputs:
            control_vel: [dx, dy, dtheta] velocity to apply to the mobile boat.

        """

        r_mob, theta_mob = self.flow_model.get_state(pos)
        r_targ, theta_targ = self.flow_model.get_state(targ)

        err = anglefunctions.wrap_to_pi(theta_targ - theta_mob) * self.flowOri.value

        r_des = r_targ + self.Kp * err * self.flowDir.value
        dir = 1 if r_des > r_mob else -1

        # print(f"{theta_mob=: 0.3} {theta_targ=: 0.3} {err=:0.3}")
        # print(f"{dir=} {r_targ=} {r_des=: 0.3} {r_mob=: 0.3}")
        # print()

        control_vel = (
            dir
            * self.vel_from_thrusters
            * np.array((np.cos(theta_mob), np.sin(theta_mob), 0))
        )

        return control_vel


def evaluate_convergence(swimmer1: swimmer.Swimmer, swimmer2: swimmer.Swimmer) -> bool:

    time_to_convergence = 2  # [s]
    convergence_threshold = 0.01  # [m]

    # Find the index of time_to_convergence ago (last one where this is true)
    inds = np.where(
        swimmer1.pose_hist[: swimmer1.life, 0]
        - (swimmer1.pose_hist[swimmer1.life, 0] - time_to_convergence)
        <= 0
    )[0]

    # If there is no such index - not enough time has passed yet.
    if len(inds) == 0:
        return False

    # Evaluate the average distance over the last time_to_convergence seconds
    convergence_ind = inds[-1]
    pose_diff = (
        swimmer1.pose_hist[convergence_ind : swimmer1.life, 1:3]
        - swimmer2.pose_hist[convergence_ind : swimmer2.life, 1:3]
    )
    avg_distance = np.mean(np.linalg.norm(pose_diff, axis=1))

    return avg_distance <= convergence_threshold

    pass
