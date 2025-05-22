"""A very high level example of how the bats detection could be used as a constraint

"""

# Import packages
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from twain import moct, plot
from twain.moct import WindFarmModel

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/la_haute_borne/wf_layout.csv'

# Set a new datum
datum = [621400, 6181000]

# Select the wind-data as timeseries
wind_data = 'data/example_site_1/wind_data.csv'

# Set the resampling factor
N_bins_wd, N_bins_ws = 5, 10

# ------------ SCRIPT ------------

# Convert to path
path_layout, path_wind_data = Path(layout), Path(wind_data)

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
    
# Load the time-series data
match path_wind_data.suffix:
    case '.yaml':
        # FIXME: Not working
        with open(wind_data, 'r') as file:
            df_wind_data = pd.json_normalize(safe_load(file))
    case '.csv':
        df_wind_data = pd.read_csv(wind_data, sep=';')
    case _:
        raise ValueError(f"Unrecognized file format '{path_wind_data.suffix}")
    
# Convert layout to numpy array
array_layout = df_layout[['x', 'y']].to_numpy() - datum

# Convert wind data to numpy array
array_wind_data = df_wind_data[['DDVEC', 'FHVEC']].to_numpy()

# FIXME: We 'inflate' the data by taking data from multiple years, and acting like its the same year
array_wd = array_wind_data[:(365 * 5), 0].reshape((365, 5), order='F').flatten(order='C')
array_ws = array_wind_data[:(365 * 5), 1].reshape((365, 5), order='F').flatten(order='C')
array_wind_data = np.column_stack((array_wd, array_ws))

# Create a measurement time array
times = np.arange(array_wd.size)

# Convert the wind data to a wind-rose object
wind_rose = moct.WindRose.from_ts(array_wd, array_ws, n_bins_wd=20, n_bins_ws=25)

# Create the scenario
scenario = moct.Scenario(wf_layout=array_layout, wt_names=df_layout['name'])

# ------------ PLOTTING ------------

# Set the viewing range
xy_range = [[0, 2500], [0, 2500]]
cut_off_threshold = 0

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(scenario, ax_exist=ax_layout)
ax_layout.set_xlim(xy_range[0])
ax_layout.set_ylim(xy_range[1])
ax_layout.set_aspect('equal')
fig_layout.suptitle("Wind-farm layout")

# Plot the timeseries data
fig_ts, (ax_ts_wd, ax_ts_ws) = plt.subplots(2, 1)
ax_ts_wd.plot(times, array_wd, 'o', markersize=2, label='Wind direction', color='blue')
ax_ts_ws.plot(times, array_ws, 'o', markersize=2, label='Wind speed', color='orange')
ax_ts_wd.set_ylabel("Direction (°)")
ax_ts_wd.set_ylim([0, 360])
ax_ts_wd.set_yticks(np.arange(0, 360 + 1, 45))
ax_ts_wd.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
ax_ts_ws.set_ylabel("Speed (m/s)")
ax_ts_ws.set_ylim([0, 25])
ax_ts_ws.set_yticks(np.linspace(0, 25, 5))
ax_ts_ws.set_xlabel("Time (quarter days)")
ax_ts_wd.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd.size / 12), offset=(array_wd.size / 24)))
ax_ts_ws.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd.size / 12), offset=(array_wd.size / 24)))
ax_ts_wd.set_xticklabels([])
ax_ts_ws.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
fig_ts.suptitle("Time-series")

# Plot the timesries data based on time-of-day
fig_tod, ax_tod = plt.subplots()
col = ax_tod.imshow(array_ws.reshape((365, 5), order='C').T, aspect='auto', cmap='inferno', interpolation='nearest')
fig_tod.suptitle("Wind speed based on time-of-day")

# Plot the wind rose
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose, threshold=cut_off_threshold)
fig_wind_rose.suptitle("Wind rose")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_ts)
# plt.close(fig_tod)
# plt.close(fig_wind_rose)
plt.show()