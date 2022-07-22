import numpy as np

import anglefunctions
import flowfield


class ApproachController:
    """
    Models a controller that drives a mobile boat towards a target boat using the flow.
    """

    def __init__(self, flow_model: flowfield.GyreFlow) -> None:
        """Creates a controller object by storing the flowfield and setting params"""

        # Control params
        self.Kp = 2
        self.vel_from_thrusters = 0.01

        # Flow model
        self.flow_model = flow_model

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

        err = anglefunctions.wrap_to_pi(theta_targ - theta_mob)

        r_des = r_targ + self.Kp * err
        dir = 1 if r_des > r_mob else -1

        control_vel = (
            dir
            * self.vel_from_thrusters
            * np.array((np.cos(theta_mob), np.sin(theta_mob), 0))
        )

        return control_vel
