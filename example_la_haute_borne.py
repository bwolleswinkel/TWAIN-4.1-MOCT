"""
Example script for yaw steering co-design.
"""

# Import packages
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/la_haute_borne/wf_layout.csv'

# Select the building plan
buildings = 'data/la_haute_borne/buildings.npy'

# Set a new datum
datum = [621400, 6181000]

# Select the scenario parameters
U_inf = 12.0  # Mean wind speed, in m/s
theta = 270  # Wind direction, in degrees
turb_intensity = 0.16  # Turbulence intensity

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
    
# Extract the noise mask
noise_mask = np.load(buildings)
# FIXME: Add X and Y to dataset itself
X_mask, Y_mask = np.meshgrid(np.linspace(0, 2500, 2500), np.linspace(0, 2500, 2500), indexing='xy')
    
# Convert to numpy array
array_layout = df_layout[['x', 'y']].to_numpy() - datum

# Create the scenario
scenario = moct.Scenario(wf_layout=array_layout, U_inf=U_inf, theta=theta, TI=turb_intensity, wt_names=df_layout['name'])

# Set the greedy control setpoints
greedy_control_setpoints = moct.ControlSetpoints(np.zeros(scenario.n_wt), np.array([1., 0.9, 0.2, 0.3]))

# ------------ PLOTTING ------------

# Set the viewing range
xy_range = [[0, 2500], [0, 2500]]

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, greedy_control_setpoints, ax_exist=ax_layout)
ax_layout.set_xlim(xy_range[0])
ax_layout.set_ylim(xy_range[1])
ax_layout.set_aspect('equal')
fig_layout.suptitle("Wind-farm layout")

# Plot the noise field
fig_noise, ax_noise = plot.noise_field(scenario, greedy_control_setpoints, xy_range=xy_range, noise_mask=[X_mask, Y_mask, noise_mask], N_points=20, clip=False)
ax_noise.set_xlim(xy_range[0])
ax_noise.set_ylim(xy_range[1])
ax_noise.set_aspect('equal')
fig_noise.suptitle("Noise field")

# Plot the flow field
fig_flow, ax_flow = plot.flow_field(scenario, greedy_control_setpoints, xy_range=xy_range, clip=False)
ax_flow.set_xlim(xy_range[0])
ax_flow.set_ylim(xy_range[1])
ax_flow.set_aspect('equal')
fig_flow.suptitle("Flow field")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_noise)
# plt.close(fig_flow)
plt.show()
