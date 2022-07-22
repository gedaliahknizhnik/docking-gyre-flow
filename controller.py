import numpy as np

import anglefunctions
import flowfield


class ApproachController:
    def __init__(self, flow_model: flowfield.GyreFlow) -> None:

        # Control params
        self.Kp = 2
        self.vel_from_thrusters = 0.01

        # Flow model
        self.flow_model = flow_model

        pass

    def get_control_vel(self, pos: np.ndarray, targ: np.ndarray) -> np.ndarray:

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
