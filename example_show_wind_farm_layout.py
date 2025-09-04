"""A simple script to visualize the layout of a wind farm

"""

# Import packages
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from twain import moct, plot
from twain.utils import gen_gaussian_process as gp
from twain.moct import WindFarmModel

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/la_prevoterie/wf_layout.csv'

# Set a new datum
# TODO: Make it such that this is automatic
datum = [5458500, 602000]

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
    
# Convert layout to numpy array
array_layout = df_layout[['x', 'y']].to_numpy() - datum

# Create the scenario
scenario = moct.Scenario(wf_layout=array_layout, wt_names=df_layout['name'])

# ------------ PLOTTING ------------

# Set the viewing range
xy_range = [[0, 2500], [0, 3500]]
cut_off_threshold = 0

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, ax_exist=ax_layout)
ax_layout.set_xlim(xy_range[0])
ax_layout.set_ylim(xy_range[1])
ax_layout.set_aspect('equal')
fig_layout.suptitle("Wind-farm layout")

# Show the plots
# plt.close(fig_layout)
plt.show()