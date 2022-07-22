import numpy as np


def wrap_to_pi(angle):
    # Helper function that wraps angles to the range (-pi,pi]
    #
    # INPUTS:
    #     angle: single angle in radians

    # OUTPUTS:
    #     angle: single angle in radians, but wrapped

    newAngle = (angle + np.pi) % (2 * np.pi) - np.pi
    return newAngle
