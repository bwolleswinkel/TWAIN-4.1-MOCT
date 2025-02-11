"""
This is an example script to create a downregulation scenario, with a fixed .
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

# Select the wind-farm layout
layout = 'data/example_site_1/wf_layout.csv'

# Select the scenario parameters
U_inf = 10.0  # In m/s
theta = 210  # In degrees

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
problem = moct.OptProblem(scenario, metrics=['aep'], opt_type='downregulation')

# Solve the problem
solution = problem.solve()

# TEMP
#
print(solution.yaw_angles)
print(solution.power_setpoints)
#

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
plot.layout(scenario, ax_exist=ax_layout)
fig_layout.suptitle("Wind-farm layout")

# Show the plots
# plt.close(fig_layout)
plt.show()
