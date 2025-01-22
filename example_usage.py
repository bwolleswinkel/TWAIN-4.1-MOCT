"""
This script explains how the multi-objective control toolbox can interact with the TWAIN environment and returns the result to the users.
"""

import pandas as pd
import numpy as np

import twain.moct as moct

# Set the parameters
wind_farm_layout = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
wind_direction = 270  # In degrees

# Solve for optimal yaw angles
yaw_angles_opt = moct.yaw_angles(wind_farm_layout, wind_direction)