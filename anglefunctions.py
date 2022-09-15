""" 
Provides helper functions for angle operations not supported by numpy
"""

from typing import Union

import numpy as np


def wrap_to_pi(angle: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Wraps angles to the range (-pi,pi]

    INPUTS:
        angle: either single angle or numpy array of angles (in radians)

    OUTPUTS:
        angle: either single angle or numpy array of angles (in radians), but wrapped

    But this will wrap the value pi to -pi.
    """

    newAngle = (angle + np.pi) % (2 * np.pi) - np.pi
    return newAngle
