"""A very high level example of how the bats detection could be used as a constraint

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

# Set the random seed
seed_value = 45538371

# Select the wind-farm layout
layout = 'data/la_haute_borne/wf_layout.csv'

# Set a new datum
datum = [621400, 6181000]

# Select the wind-data as timeseries
wind_data = 'data/example_site_1/wind_data.csv'

# ------------ SCRIPT ------------

# Set the random seed
if seed_value is not None:
    np.random.seed(seed_value)

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

# Create a random temperature timeseries
array_temp = np.roll(20 + gp(times, kernel='periodic', params={'var': 4, 'length_scale': 0.1, 'period': 4}) + 3 * gp(times, kernel='RBF', params={'var': 10, 'length_scale': 300}) + gp(times, kernel='RBF', params={'var': 5, 'length_scale': 8}), -200)

# Select the indices for which there is bat activity
summer_months = np.zeros_like(times, dtype=bool)
summer_months[500:1500] = True
evening_night_hours = np.zeros_like(times, dtype=bool)
evening_night_hours[::5] = True
evening_night_hours[1::5] = True
evening_night_hours[4::5] = True
indices_bat_activity = np.bitwise_and(np.bitwise_and(array_temp > 20.5, summer_months), evening_night_hours)

# Select the indices for which bats are actively spotted
prob_bat_spotted = np.clip(array_temp, 15, None) / array_temp.max() - 0.5
prob_bat_spotted[2::5] = 0
prob_bat_spotted[3::5] = 0
prob_bat_spotted = np.clip(1000 / (200 * np.sqrt(2 * np.pi)) * np.exp( - (times - 1000) ** 2 / (2 * 200 ** 2)), 0, 1) * prob_bat_spotted
indices_bat_spotted = np.random.choice(np.arange(times.size, dtype=int), 50, p=prob_bat_spotted / prob_bat_spotted.sum())

# Convert the wind speed data into time-of day
array_ws_per_tod = array_ws.reshape((365, 5), order='C').T
array_ws_per_tod_no_bats = array_ws_per_tod.copy()
array_ws_per_tod_no_bats[indices_bat_activity.reshape((365, 5), order='C').T] = np.nan
array_ws_per_tod_bats = array_ws_per_tod.copy()
array_ws_per_tod_bats[~indices_bat_activity.reshape((365, 5), order='C').T] = np.nan

# Convert the wind data to a wind-rose object
wind_rose = moct.WindRose.from_ts(array_wd, array_ws, n_bins_wd=20, n_bins_ws=25)
wind_rose_coarse = moct.WindRose.from_ts(array_wd, array_ws, n_bins_wd=20, n_bins_ws=5)

# Convert the wind data to a wind-rose object, separated by bat activity
wind_rose_no_bats = moct.WindRose.from_ts(array_wd[~indices_bat_activity], array_ws[~indices_bat_activity], n_bins_wd=20, n_bins_ws=25)
wind_rose_bats = moct.WindRose.from_ts(array_wd[indices_bat_activity], array_ws[indices_bat_activity], n_bins_wd=20, n_bins_ws=25)
wind_rose_bats.data = wind_rose_bats.data / (wind_rose_bats.data.sum() + wind_rose_no_bats.data.sum())

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
fig_ts, (ax_ts_wd, ax_ts_ws, ax_ts_temp) = plt.subplots(3, 1)
ax_ts_wd.plot(times[~indices_bat_activity], array_wd[~indices_bat_activity], 'o', markersize=3, label='Wind direction', color='blue')
ax_ts_wd.plot(times[indices_bat_activity], array_wd[indices_bat_activity], 'o', markersize=3, markeredgecolor=colors.to_rgba('blue', 0.2), markerfacecolor=colors.to_rgba('blue', 0.2))
ax_ts_wd.plot(times[indices_bat_spotted], array_wd[indices_bat_spotted], 'x', markersize=4, color='purple')
ax_ts_ws.plot(times, array_ws, 'o', markersize=2, label='Wind speed', color='orange')
ax_ts_temp.plot(times, array_temp, label='Temperature', color='red')
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
ax_ts_temp.set_ylabel("Temperature (in °C)")
ax_ts_temp.set_ylim([-5, 35])
fig_ts.suptitle("Time-series")

# Plot the timeseries data based on time-of-day
X, Y = np.meshgrid(np.arange(array_ws_per_tod.shape[1]), np.arange(array_ws_per_tod.shape[0]), indexing='ij')
fig_tod, ax_tod = plt.subplots()
ax_tod.imshow(array_ws_per_tod_bats, aspect='auto', vmin=0, vmax=np.nanmax(array_ws), alpha=0.5, cmap='inferno', interpolation='nearest')
col = ax_tod.imshow(array_ws_per_tod_no_bats, aspect='auto', vmin=0, vmax=np.nanmax(array_ws), cmap='inferno', interpolation='nearest')
ax_tod.plot(X.flatten()[indices_bat_spotted], Y.flatten()[indices_bat_spotted], 'x', markersize=4, color='red', label='Bats spotted')
plt.colorbar(col, ax=ax_tod, label='Wind speed (m/s)')
ax_tod.set_yticklabels(reversed(['Night', 'Morning', 'Midday', 'Afternoon', 'Evening', '']))
ax_tod.xaxis.set_major_locator(ticker.IndexLocator(base=(365 / 12), offset=(365 / 24)))
ax_tod.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax_tod.set_xlabel("Time (quarter days)")
ax_tod.set_ylabel("Time of day")
ax_tod.legend(loc='upper right')
fig_tod.suptitle("Wind speed based on time-of-day")

# Plot the wind rose
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose, threshold=cut_off_threshold)
fig_wind_rose.suptitle("Wind rose")

# Plot the wind rose, but as found normally
fig_wind_rose_conv, ax_wind_rose_conv = plot.wind_rose_conv(wind_rose_coarse)
fig_wind_rose_conv.suptitle("Wind rose, conventional")

# Plot the wind rose, but with explanations
fig_wind_rose_exp,  (ax_wind_rose_direction, ax_wind_rose_speed)  = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax_wind_rose_direction.set_theta_direction('clockwise')
ax_wind_rose_speed.set_theta_direction('clockwise')
ax_wind_rose_direction.set_theta_zero_location('N')
ax_wind_rose_speed.set_theta_zero_location('N')
T, R = np.meshgrid(np.linspace(0, 2 * np.pi, wind_rose.n_bins_wd), np.linspace(0, 25, wind_rose.n_bins_ws))
#: Construct the first masks, based on wind direction
data_mask = wind_rose.data.copy()
data_mask[~(np.bitwise_and(T > np.pi, T < 4/3 * np.pi))] = np.nan
ax_wind_rose_direction.pcolormesh(T, R, wind_rose.data * 100, edgecolors='none', cmap='inferno', alpha=0.5)
cs = ax_wind_rose_direction.pcolormesh(T, R, data_mask * 100, edgecolors='face', cmap='inferno')
cbar = plt.colorbar(cs, ax=ax_wind_rose_direction)
cbar.set_label('Prevalence (%)')
ax_wind_rose_direction.set_title(f"Prevalence: {np.nansum(data_mask) * 100:.2f}%")
ax_wind_rose_direction.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax_wind_rose_direction.tick_params(axis='y', colors='green')
#: Construct the second masks, based on wind speed
data_mask = wind_rose.data.copy()
data_mask[~(R[:, 0] > 10)] = np.nan
ax_wind_rose_speed.pcolormesh(T, R, wind_rose.data * 100, edgecolors='none', cmap='inferno', alpha=0.5)
cs = ax_wind_rose_speed.pcolormesh(T, R, data_mask * 100, edgecolors='face', cmap='inferno')
cbar = plt.colorbar(cs, ax=ax_wind_rose_speed)
cbar.set_label('Prevalence (%)')
ax_wind_rose_speed.set_title(f"Prevalence: {np.nansum(data_mask) * 100:.2f}%")
ax_wind_rose_speed.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
ax_wind_rose_speed.tick_params(axis='y', colors='green')
fig_wind_rose_exp.suptitle("Wind rose (explained)")

# Plot the wind rose, separated by bat activity
fig_wind_rose_bat_activity, (ax_wind_rose_no_bats, ax_wind_rose_bats)  = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
ax_wind_rose_no_bats = plot.wind_rose(wind_rose_no_bats, threshold=cut_off_threshold, ax_exist=ax_wind_rose_no_bats)
ax_wind_rose_bats = plot.wind_rose(wind_rose_bats, threshold=cut_off_threshold, ax_exist=ax_wind_rose_bats)
ax_wind_rose_no_bats.set_title("Wind rose (no bat activity)")
ax_wind_rose_bats.set_title("Wind rose (bat activity)")
fig_wind_rose_bat_activity.suptitle("Wind rose (separated by bat activity)")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_ts)
# plt.close(fig_tod)
# plt.close(fig_wind_rose)
# plt.close(fig_wind_rose_conv)
# plt.close(fig_wind_rose_exp)
# plt.close(fig_wind_rose_bat_activity)
plt.show()