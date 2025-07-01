"""A script to demonstrate that optimization is agnostic to time-series data or distributions

"""

# Import packages
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

from twain import moct, plot, utils
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

# Select the number of bins for the data
N_bins_wd, N_bins_ws = 36, 25
N_bins_wd_coarse, N_bins_ws_coarse = 18, 5

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
# FIXME: I have to delete certain entries, otherwise I get 'nan' result
array_wd = np.delete(array_wind_data[:(365 * 5), 0].reshape((365, 5), order='F').flatten(order='C'), range(620, 630))
array_ws = np.delete(array_wind_data[:(365 * 5), 1].reshape((365, 5), order='F').flatten(order='C'), range(620, 630))
array_wind_data = np.column_stack((array_wd, array_ws))

# TEMP
print(365 * 5)
print(array_wd.size)

# Create a measurement time array
times = np.arange(array_wd.size)

# Create a random temperature timeseries
array_electricity = np.roll(20 + gp(times, kernel='periodic', params={'var': 4, 'length_scale': 0.1, 'period': 4}) + 3 * gp(times, kernel='RBF', params={'var': 10, 'length_scale': 300}) + gp(times, kernel='RBF', params={'var': 5, 'length_scale': 8}), -200) / 100 + 0.02

# Perform the 'binning' of the wind speeds and directions
array_wd_binned = np.round(array_wd / (360 / N_bins_wd)) * (360 / N_bins_wd)
array_ws_binned = np.round(array_ws / (25 / N_bins_ws)) * (25 / N_bins_ws)
array_wd_binned_coarse = np.round(array_wd / (360 / N_bins_wd_coarse)) * (360 / N_bins_wd_coarse)
array_ws_binned_coarse = np.round(array_ws / (25 / N_bins_ws_coarse)) * (25 / N_bins_ws_coarse)

# Convert the wind data to a wind-rose object
wind_rose = moct.WindRose.from_ts(array_wd, array_ws, n_bins_wd=N_bins_wd, n_bins_ws=N_bins_ws)
wind_rose_coarse = moct.WindRose.from_ts(array_wd, array_ws, n_bins_wd=N_bins_wd_coarse, n_bins_ws=N_bins_ws_coarse)

# Create the scenario
empty_scenario = moct.Scenario(wf_layout=array_layout, wt_names=df_layout['name'])

# Create a full scenario for time-series, and distributions
full_scenario_ts = moct.Scenario(wf_layout=array_layout, U_inf=array_ws, theta=array_wd, TI=np.full(times.size, 0.06),  wt_names=df_layout['name'])
full_scenario_dist = moct.Scenario(wf_layout=array_layout, U_inf=wind_rose.wind_rose['wind_speed'].to_numpy(), theta=wind_rose.wind_rose['wind_dir'].to_numpy(), TI=np.full(wind_rose.data.size, 0.06),  wt_names=df_layout['name'])
full_scenario_dist_coarse = moct.Scenario(wf_layout=array_layout, U_inf=wind_rose_coarse.wind_rose['wind_speed'].to_numpy(), theta=wind_rose_coarse.wind_rose['wind_dir'].to_numpy(), TI=np.full(wind_rose_coarse.data.size, 0.06),  wt_names=df_layout['name'])

# Construct an optimization problem
problem_ts = moct.OptProblem(full_scenario_ts, metrics=['aep'], opt_type='wake_steering', opt_method='geometric')
# FIXME: Again, if I choose 'serial-refine,' I get NaN slice encountered in the evaluation, but not with 'geometric'
problem_dist = moct.OptProblem(full_scenario_dist, metrics=['aep'], opt_type='wake_steering', opt_method='geometric')
problem_dist_coarse = moct.OptProblem(full_scenario_dist_coarse, metrics=['aep'], opt_type='wake_steering', opt_method='geometric')

# Solve the problem
control_setpoints_ts = problem_ts.solve()
control_setpoints_dist = problem_dist.solve()
control_setpoints_dist_coarse = problem_dist_coarse.solve()

# Create a wind farm model
wf_model_ts = WindFarmModel(full_scenario_ts)
wf_model_dist = WindFarmModel(full_scenario_dist)
wf_model_dist_coarse = WindFarmModel(full_scenario_dist_coarse)

# Set the greedy control setpoints
greedy_control_setpoints_ts = moct.ControlSetpoints(np.zeros((empty_scenario.n_wt, times.size)), np.ones((empty_scenario.n_wt, times.size)))
greedy_control_setpoints_dist = moct.ControlSetpoints(np.zeros((empty_scenario.n_wt, wind_rose.data.size)), np.ones((empty_scenario.n_wt, wind_rose.data.size)))
greedy_control_setpoints_dist_coarse = moct.ControlSetpoints(np.zeros((empty_scenario.n_wt, wind_rose_coarse.data.size)), np.ones((empty_scenario.n_wt, wind_rose_coarse.data.size)))

# Calculate the wind farm power
greedy_power_ts, *_ = wf_model_ts.impact_control_variables(greedy_control_setpoints_ts)
power_ts, *_ = wf_model_ts.impact_control_variables(control_setpoints_ts)
greedy_power_dist, *_ = wf_model_dist.impact_control_variables(greedy_control_setpoints_dist)
power_dist, *_ = wf_model_dist.impact_control_variables(control_setpoints_dist)
greedy_power_dist_coarse, *_ = wf_model_dist_coarse.impact_control_variables(greedy_control_setpoints_dist_coarse)
power_dist_coarse, *_ = wf_model_dist_coarse.impact_control_variables(control_setpoints_dist_coarse)

# Reshape back into an array
greedy_power_dist = greedy_power_dist.reshape((N_bins_wd, N_bins_ws, full_scenario_dist.n_wt))
power_dist = power_dist.reshape((N_bins_wd, N_bins_ws, full_scenario_dist.n_wt))
greedy_power_dist_coarse = greedy_power_dist_coarse.reshape((N_bins_wd_coarse, N_bins_ws_coarse, full_scenario_dist_coarse.n_wt))
power_dist_coarse_coarse = power_dist_coarse.reshape((N_bins_wd_coarse, N_bins_ws_coarse, full_scenario_dist_coarse.n_wt))

# Compute the AEP
metrics = moct.Metrics()
greedy_aep_ts = metrics.compute_aep(greedy_power_ts)
geometric_aep_ts = metrics.compute_aep(power_ts)
greedy_aep_dist = metrics.compute_aep(greedy_power_dist, params={'prevalence': wind_rose.wind_rose['prevalence'].to_numpy().reshape((N_bins_wd, N_bins_ws))})
geometric_aep_dist = metrics.compute_aep(power_dist, params={'prevalence': wind_rose.wind_rose['prevalence'].to_numpy().reshape((N_bins_wd, N_bins_ws))})
greedy_aep_dist_coarse = metrics.compute_aep(greedy_power_dist_coarse, params={'prevalence': wind_rose_coarse.wind_rose['prevalence'].to_numpy().reshape((N_bins_wd_coarse, N_bins_ws_coarse))})
geometric_aep_dist_coarse = metrics.compute_aep(power_dist_coarse_coarse, params={'prevalence': wind_rose_coarse.wind_rose['prevalence'].to_numpy().reshape((N_bins_wd_coarse, N_bins_ws_coarse))})

# ------------ PRINTING ------------

# Print the total AEP
print(f"\n--- AEP ---")
print(f"Greedy AEP time-series: {greedy_aep_ts * 1E-9 / 3600:.2f} GWh")
print(f"AEP time-series (serial-refine): {geometric_aep_ts * 1E-9 / 3600:.2f} GWh (\033[34m{'+' if np.sign((geometric_aep_ts - greedy_aep_ts) / greedy_aep_ts) >= 0 else '-'}{(geometric_aep_ts - greedy_aep_ts) / greedy_aep_ts * 100:.2f}%\033[0m)")
print(f"Greedy AEP distribution: {greedy_aep_dist * 1E-9 / 3600:.2f} GWh")
print(f"AEP distribution (geometric): {geometric_aep_dist * 1E-9 / 3600:.2f} GWh (\033[34m{'+' if np.sign((geometric_aep_dist - greedy_aep_dist) / greedy_aep_dist) >= 0 else '-'}{(geometric_aep_dist - greedy_aep_dist) / greedy_aep_dist * 100:.2f}%\033[0m)")
print(f"Greedy AEP distribution, coarse: {greedy_aep_dist_coarse * 1E-9 / 3600:.2f} GWh")
print(f"AEP distribution (geometric), coarse: {geometric_aep_dist_coarse * 1E-9 / 3600:.2f} GWh (\033[34m{'+' if np.sign((geometric_aep_dist_coarse - greedy_aep_dist_coarse) / greedy_aep_dist_coarse) >= 0 else '-'}{(geometric_aep_dist_coarse - greedy_aep_dist_coarse) / greedy_aep_dist_coarse * 100:.2f}%\033[0m)")

# ------------ PLOTTING ------------

# Set the viewing range
xy_range = [[0, 2500], [0, 2500]]
cut_off_threshold = 0

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(empty_scenario, ax_exist=ax_layout)
ax_layout.set_xlim(xy_range[0])
ax_layout.set_ylim(xy_range[1])
ax_layout.set_aspect('equal')
fig_layout.suptitle("Wind-farm layout")

# Plot the timeseries data
fig_ts, (ax_ts_wd, ax_ts_ws, ax_ts_price) = plt.subplots(3, 1, sharex=True)
ax_ts_wd.plot(times, array_wd, 'o', markersize=2, label='Wind direction', color='blue')
ax_ts_ws.plot(times, array_ws, 'o', markersize=2, label='Wind speed', color='orange')
ax_ts_price.plot(times, array_electricity, 'o', markersize=2, label='Electricity price', color='red')
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
ax_ts_price.set_ylabel("Price (in €/kWh)")
ax_ts_price.set_ylim([0, 0.35])
fig_ts.suptitle("Time-series")

# Plot the timeseries data, 'binned'
fig_ts_binned, ((ax_ts_wd_binned, ax_ts_wd_binned_coarse), (ax_ts_ws_binned, ax_ts_ws_binned_coarse)) = plt.subplots(2, 2, sharex=True)
# Fine
for idx in range(N_bins_wd):
    ax_ts_wd_binned.axhline(y=(360 / N_bins_wd) * idx, color='gray', linewidth=0.2, alpha=0.5)
for idx in range(N_bins_ws):
    ax_ts_ws_binned.axhline(y=(25 / N_bins_ws) * idx, color='gray', linewidth=0.2, alpha=0.5)
# Coarse
for idx in range(N_bins_wd_coarse):
    ax_ts_wd_binned_coarse.axhline(y=(360 / N_bins_wd_coarse) * idx, color='gray', linewidth=0.2, alpha=0.5)
for idx in range(N_bins_ws_coarse):
    ax_ts_ws_binned_coarse.axhline(y=(25 / N_bins_ws_coarse) * idx, color='gray', linewidth=0.2, alpha=0.5)
# Fine
ax_ts_wd_binned.plot(times, array_wd_binned, 'o', markersize=2, label='Wind direction', color='blue')
ax_ts_ws_binned.plot(times, array_ws_binned, 'o', markersize=2, label='Wind speed', color='orange')
ax_ts_wd_binned.set_ylabel("Direction (°)")
ax_ts_wd_binned.set_ylim([0, 360])
ax_ts_wd_binned.set_yticks(np.arange(0, 360 + 1, 45))
ax_ts_wd_binned.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
ax_ts_ws_binned.set_ylabel("Speed (m/s)")
ax_ts_ws_binned.set_ylim([0, 25])
ax_ts_ws_binned.set_yticks(np.linspace(0, 25, 5))
ax_ts_ws_binned.set_xlabel("Time (quarter days)")
ax_ts_wd_binned.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd_binned.size / 12), offset=(array_wd_binned.size / 24)))
ax_ts_ws_binned.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd_binned.size / 12), offset=(array_wd_binned.size / 24)))
ax_ts_wd_binned.set_xticklabels([])
ax_ts_ws_binned.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# Coarse
ax_ts_wd_binned_coarse.plot(times, array_wd_binned_coarse, 'o', markersize=2, label='Wind direction', color='blue')
ax_ts_ws_binned_coarse.plot(times, array_ws_binned_coarse, 'o', markersize=2, label='Wind speed', color='orange')
ax_ts_wd_binned_coarse.set_ylabel("Direction (°)")
ax_ts_wd_binned_coarse.set_ylim([0, 360])
ax_ts_wd_binned_coarse.set_yticks(np.arange(0, 360 + 1, 45))
ax_ts_wd_binned_coarse.set_yticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
ax_ts_ws_binned_coarse.set_ylabel("Speed (m/s)")
ax_ts_ws_binned_coarse.set_ylim([0, 25])
ax_ts_ws_binned_coarse.set_yticks(np.linspace(0, 25, 5))
ax_ts_ws_binned_coarse.set_xlabel("Time (quarter days)")
ax_ts_wd_binned_coarse.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd_binned_coarse.size / 12), offset=(array_wd_binned_coarse.size / 24)))
ax_ts_ws_binned_coarse.xaxis.set_major_locator(ticker.IndexLocator(base=(array_wd_binned_coarse.size / 12), offset=(array_wd_binned_coarse.size / 24)))
ax_ts_wd_binned_coarse.set_xticklabels([])
ax_ts_ws_binned_coarse.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
fig_ts_binned.suptitle("Time-series, binned")

# Plot the wind rose
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose, threshold=0)
fig_wind_rose.suptitle("Wind rose")

# Plot the wind rose, 'unwrapped'
fig_wind_rose_unwrapped, ax_wind_rose_unwrapped = plt.subplots()
cs = ax_wind_rose_unwrapped.imshow(np.flipud(wind_rose.data) * 100, cmap='inferno', extent=[0 - (1/2 * (360 / N_bins_wd)), 360 + (1/2 * (360 / N_bins_wd)), 0 - (1/2 * (25 / N_bins_ws)), 25 + (1/2 * (25 / N_bins_ws))], aspect='auto')
cbar = fig_wind_rose_unwrapped.colorbar(cs, ax=ax_wind_rose_unwrapped)
cbar.set_label("Prevalence (%)")
ax_wind_rose_unwrapped.set_xticks(np.linspace(0, 360, np.max(utils.devisors(N_bins_wd)[utils.devisors(N_bins_wd) <= 10])))
ax_wind_rose_unwrapped.set_yticks(np.linspace(0, 25, np.max(utils.devisors(N_bins_ws)[utils.devisors(N_bins_ws) <= 10])))
ax_wind_rose_unwrapped.set_xlabel("Wind direction (in °)")
ax_wind_rose_unwrapped.set_ylabel("Wind speed (in m/s)")
fig_wind_rose_unwrapped.suptitle("Wind rose, unwrapped")

# Plot the wind rose, coarse
fig_wind_rose_coarse, ax_wind_rose_coarse = plot.wind_rose(wind_rose_coarse, threshold=0)
fig_wind_rose_coarse.suptitle("Wind rose, coarse")

# Plot the wind rose, but as found normally
fig_wind_rose_conv, ax_wind_rose_conv = plot.wind_rose_conv(wind_rose)
fig_wind_rose_conv.suptitle("Wind rose, conventional")

# Plot the wind rose, but as found normally
fig_wind_rose_conv_coarse, ax_wind_rose_conv_coarse = plot.wind_rose_conv(wind_rose_coarse)
fig_wind_rose_conv_coarse.suptitle("Wind rose (coarse), conventional")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_ts)
# plt.close(fig_tod)
# plt.close(fig_wind_rose)
# plt.close(fig_wind_rose_coarse)
# plt.close(fig_wind_rose_conv)
# plt.close(fig_wind_rose_conv_coarse)
plt.show()