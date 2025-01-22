"""
This is the high-level module for the multi-objective control toolbox (MOCT) of the TWAIN environment.
"""

from typing import TypeVar, TypeAlias
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.typing as npt

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

# ------ CLASSES ------


def Optimizer():
    def __init__(self):
        pass


# ------ FUNCTIONS ------


def open(connect: dict = None, dir_local: Path | str = None):
    """
    This function launches the GUI for the multi-objective control toolbox (MOCT) of the TWAIN environment.

    Parameters
    ----------
    connect : dict, optional
        The connection details to the TWAIN environment, by default None
    """
    pass


# FIXME: What we actually want is a separate subdirectory with a module called 'optimize.py' that contains the following function, such that we can call 'moct.optimize.yaw_angles' in the example_usage.py script.
def yaw_angles(wind_farm_layout: NPArray[float], wind_direction: float) -> NPArray[float]:
    """
    Given a wind farm layout and a wind direction, this function solves for the optimal yaw angles of the wind turbines.

    Parameters
    ----------
    wind_farm_layout :
        The layout of the wind farm, where each row is a wind turbine and the columns are the x and y coordinates, respectively.
    wind_direction :
        The wind direction for which to optimize the yaw angles.

    Returns
    -------
    optimal_yaw_angles :
        The optimal yaw angles of the wind turbines.
    """
    # FIXME: Perform some check on the input
    n_wt = wind_farm_layout.shape[0]
    optimal_yaw_angles = np.random.rand(n_wt)
    return optimal_yaw_angles

        
