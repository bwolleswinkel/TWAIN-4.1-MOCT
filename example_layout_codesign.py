"""
This is a script for offline co-design optimization.

# FIXME: Not working, FLORIS does not do any optimization, and no wind-rose is provided.
"""

# Import packages
from pathlib import Path

import numpy as np
import scipy as sp
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot

# ------------ PARAMETERS ------------

# Select the scenario parameters
U_inf = 12.0  # Mean wind speed, in m/s
theta = 210  # Wind direction, in degrees
turb_intensity = 0.06  # Turbulence intensity

# ------------ SCRIPT ------------

# Create the scenario
scenario = moct.Scenario(U_inf=U_inf, theta=theta, n_wt=10, perimeter=np.array([[0, 0], [500, 0], [500, 500], [0, 500]]))

# Construct an optimization problem
problem = moct.OptProblem(scenario, metrics=['aep'], opt_type='layout', opt_method='scipy')

# Solve the problem
optimal_control_setpoints, scenario = problem.solve()

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, ax_exist=ax_layout)
fig_layout.suptitle("Wind-farm layout")

# Plot the noise field
fig_noise, ax_noise = plot.noise_field(scenario, optimal_control_setpoints)
ax_noise.set_xlim([0, 500])
ax_noise.set_ylim([0, 500])
fig_noise.suptitle("Noise field")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_noise)
plt.show()
