"""
Example script to plot a wind rose
"""

# Import packages
from pathlib import Path

import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot

# ------------ PARAMETERS ------------

# Select the directory with the wind-rose data
wind_rose = 'data/example_site_1/wind_rose.json'

# ------------ SCRIPT ------------

# Convert to path
path_layout = Path(wind_rose)

# Load the wind-farm layout
match path_layout.suffix:
    case '.yaml':
        raise NotImplementedError("YAML format not implemented yet")
    case '.csv':
        raise NotImplementedError("CSV format not implemented yet")
    case '.json':
        with open(path_layout, 'r') as file:
            wind_rose = safe_load(file)
    case _:
        raise ValueError(f"Unrecognized file format '{path_layout.suffix}")

# Construct the wind-rose object
wind_rose = moct.WindRose(wind_rose)

# ------------ PRINTING ------------

# Print the wind-rose data
print(wind_rose)

# ------------ PLOTTING ------------

# Plot the wind-farm layout
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose)
fig_wind_rose.suptitle("Wind rose")

# Show the plots
# plt.close(fig_wind_rose)
plt.show()
