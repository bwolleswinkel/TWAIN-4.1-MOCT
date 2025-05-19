"""Example script of optimization of power lines.

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

# Construct an optimization problem
problem = moct.OptProblem(scenario, metrics=['aep'], opt_type='power_lines', params=[['000', '008', '006', '003'], ['001', '005', '007', '002', '009', '004']])

# Solve the problem
_, scenario = problem.solve()

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, ax_exist=ax_layout)
fig_layout.suptitle("Wind-farm layout")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_noise)
plt.show()
