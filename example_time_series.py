"""Example script that works with time series data.

"""

import numpy as np
import matplotlib.pyplot as plt

from twain.moct import WindRose
from twain import plot
from twain.utils import gen_gaussian_process, remap

# ------ PARAMETERS ------

# Set the random seed
seed_value = 45538371

# Set the time series length
T_range = [0, 365]  # Unit of length
N_T = 365 * 24 # Number of time steps

# ------ SCRIPT ------

# Set the random seed
if seed_value is not None:
    np.random.seed(seed_value)

# Generate the time vector
T = np.linspace(T_range[0], T_range[1], N_T)
Delta_T = T[1] - T[0]  # Time step size

# Generate the wind-direction time series
WD = gen_gaussian_process(T, kernel='periodic', params={'var': 0.3, 'length_scale': 20, 'period': 30})
WD += gen_gaussian_process(T, kernel='RBF', params={'var': 0.1, 'length_scale': 25})
WD += gen_gaussian_process(T, kernel='white', params={'var': 0.01})
WD = np.random.uniform(0, 180) + remap(WD, [0, 180])  # Wrap the wind direction to [0, 360]

# Generate the wind-speed time series
WS = gen_gaussian_process(T, kernel='periodic', params={'var': 0.1, 'length_scale': 2.5, 'period': 30})
WS += gen_gaussian_process(T, kernel='RBF', params={'var': 0.1, 'length_scale': 10})
WS += gen_gaussian_process(T, kernel='white', params={'var': 0.02})
WS = remap(WS, [0, 10])  # Wrap the wind speed to [0, 10]

# Convert the time-series to distributional data
wind_rose = WindRose.from_ts(WD, WS, n_bins_wd=20, n_bins_ws=25)

# ------ PLOTTING ----

# Plot the time-series data
fig_ts, ax_ts_wd = plt.subplots()
ax_ts_ws = ax_ts_wd.twinx()
ax_ts_wd.plot(T, WD, color='blue', label=r"Wind direction $\phi$")
ax_ts_wd.set_xlabel("Time (in days)")
ax_ts_wd.set_ylabel("Wind direction (in °)")
ax_ts_wd.set_yticks(45 * np.arange(0, 8))
ax_ts_wd.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax_ts_wd.set_ylim(0, 360)
ax_ts_ws.plot(T, WS, color='red', label=r"Wind speed $U_{\infty}$")
ax_ts_ws.set_xlabel("Time (in days)")
ax_ts_ws.set_ylabel("Wind speed (in m/s)")
ax_ts_ws.set_ylim(0, 25)
fig_ts.suptitle("Time series data")
#: Add the legend to both axes
lines, labels = ax_ts_wd.get_legend_handles_labels()
lines2, labels2 = ax_ts_ws.get_legend_handles_labels()
ax_ts_ws.legend(lines + lines2, labels + labels2, loc='upper right')

# Plot the wind rose
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose)
fig_wind_rose.suptitle("Wind rose")

# Show the plots
# plt.close(fig_ts)
# plt.close(fig_wind_rose)
plt.show()
