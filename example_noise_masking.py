"""Ths is an example script on noise masking

"""

from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/example_site_1/wf_layout.csv'

# Select the building plan
buildings = 'data/example_site_1/buildings.npy'

# Select the scenario parameters
U_inf = 12.0  # Mean wind speed, in m/s
theta = 210  # Wind direction, in degrees
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
scenario = moct.Scenario(wf_layout=array_layout, U_inf=U_inf, theta=theta)

# Get the greedy control setpoints
greedy_control_setpoints = moct.ControlSetpoints(np.zeros(scenario.n_wt), np.ones(scenario.n_wt))

# Extract the noise mask
noise_mask = np.load(buildings)
# FIXME: Add X and Y to dataset itself
X, Y = np.meshgrid(np.linspace(0, 500, 500), np.linspace(0, 500, 500), indexing='xy')

# Construct an optimization problem
problem = moct.OptProblem(scenario, metrics=['noise'], opt_type='downregulation', params=[(X, Y, noise_mask)])

# Solve the problem
optimal_control_setpoints = problem.solve()

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, ax_exist=ax_layout)
fig_layout.suptitle("Wind-farm layout")

# Plot the noise field (greedy)
fig_noise_greedy, ax_noise_greedy = plot.noise_field(scenario, greedy_control_setpoints)
ax_noise_greedy.set_xlim([0, 500])
ax_noise_greedy.set_ylim([0, 500])
fig_noise_greedy.suptitle("Noise field (greedy)")

# Plot the noise field (optimal)
fig_noise_optimal, ax_noise_optimal = plot.noise_field(scenario, optimal_control_setpoints)
ax_noise_optimal.set_xlim([0, 500])
ax_noise_optimal.set_ylim([0, 500])
fig_noise_optimal.suptitle("Noise field (optimal)")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_noise_greedy)
plt.show()
