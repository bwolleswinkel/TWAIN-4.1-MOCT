"""
Example script for yaw steering co-design.
"""

# Import packages
from pathlib import Path

import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/example_site_1/wf_layout.csv'

# Select the scenario parameters
U_inf = 12.0  # Mean wind speed, in m/s
theta = 270  # Wind direction, in degrees
turb_intensity = 0.06  # Turbulence intensity

# ------------ SCRIPT ------------

# Convert to path
path_layout = Path(layout)

# Load the wind-farm layout
match path_layout.suffix:
    case '.yaml':
        # FIXME: Not working
        with open(layout, 'r') as file:
            df_layout = pd.json_normalize(safe_load(file))
    case '.csv':
        df_layout = pd.read_csv(layout, sep=';')
    case _:
        raise ValueError(f"Unrecognized file format '{path_layout.suffix}")
    
# Convert to numpy array
array_layout = df_layout[['x', 'y']].to_numpy()

# Create the scenario
scenario = moct.Scenario(wf_layout=array_layout, U_inf=U_inf, theta=theta, TI=turb_intensity)

# Construct an optimization problem
problem = moct.OptProblem(scenario, metrics=['aep'], opt_type='wake_steering', opt_method='pso', params={'N_swarm': 20, 'N_iter': 20})

# Solve the problem
optimal_control_setpoints = problem.solve(verbose=1)

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, optimal_control_setpoints, ax_exist=ax_layout)
fig_layout.suptitle("Wind-farm layout")

# Plot the noise field
fig_noise, ax_noise = plot.noise_field(scenario, optimal_control_setpoints)
ax_noise.set_xlim([0, 500])
ax_noise.set_ylim([0, 500])
fig_noise.suptitle("Noise field")

# Plot the flow field
# FIXME: The yaw offset seems to have no effect on the wake?
fig_flow, ax_flow = plot.flow_field(scenario, optimal_control_setpoints, clip=False)
ax_flow.set_xlim([0, 500])
ax_flow.set_ylim([0, 500])
fig_flow.suptitle("Flow field")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_noise)
# plt.close(fig_flow)
plt.show()
