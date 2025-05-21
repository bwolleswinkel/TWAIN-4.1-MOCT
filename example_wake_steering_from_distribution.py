"""Script to demonstrate wake steering optimization from distributional data with geometric yaw control.

"""

# Import packages
from pathlib import Path

import numpy as np
import pandas as pd
from yaml import safe_load
import matplotlib.pyplot as plt

from twain import moct, plot, utils
from twain.moct import WindFarmModel

# ------------ PARAMETERS ------------

# Select the wind-farm layout
layout = 'data/la_haute_borne/wf_layout.csv'

# Select the directory with the wind-rose data
wind_rose = 'data/la_haute_borne/wind_rose.csv'

# Set a new datum
datum = [621400, 6181000]

# Set the resampling factor
N_bins_wd, N_bins_ws = None, None

# ------------ SCRIPT ------------

# Convert to path
path_layout = Path(layout)
path_wind_rose = Path(wind_rose)

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
    
# Load the wind rose data
match path_wind_rose.suffix:
    case '.yaml':
        raise NotImplementedError("YAML format not implemented yet")
    case '.csv':
        with open(path_wind_rose, 'r') as file:
            df_wind_rose = pd.read_csv(file, sep=';')
    case '.json':
        with open(path_wind_rose, 'r') as file:
            wind_rose = safe_load(file)
    case _:
        raise ValueError(f"Unrecognized file format '{path_layout.suffix}")
    
# Convert to numpy array
array_layout = df_layout[['x', 'y']].to_numpy() - datum

# Construct the wind-rose object
wind_rose = moct.WindRose(df_wind_rose)

# Resample the wind-rose data
wind_rose.resample(N_bins_wd, N_bins_ws, mode='resample')
N_bins_ws, N_bins_wd = wind_rose.data.shape

# Create an empty scenario
empty_scenario = moct.Scenario(wf_layout=array_layout, wt_names=df_layout['name'])

# Create a full scenario
full_scenario = moct.Scenario(wf_layout=array_layout, U_inf=wind_rose.wind_rose['wind_speed'].to_numpy(), theta=wind_rose.wind_rose['wind_dir'].to_numpy(), TI=moct.ti_from_ws(wind_rose.wind_rose['wind_speed'].to_numpy()),  wt_names=df_layout['name'])

# Construct an optimization problem
problem = moct.OptProblem(full_scenario, metrics=['aep'], opt_type='wake_steering', opt_method='geometric')

# Solve the problem
geometric_control_setpoints = problem.solve()

# Create a wind farm model
wf_model = WindFarmModel(full_scenario)

# Set the greedy control setpoints
greedy_control_setpoints = moct.ControlSetpoints(np.zeros((empty_scenario.n_wt, wind_rose.data.size)), np.ones((empty_scenario.n_wt, wind_rose.data.size)))

# Calculate the wind farm power
greedy_power, *_ = wf_model.impact_control_variables(greedy_control_setpoints)
geometric_power, *_ = wf_model.impact_control_variables(geometric_control_setpoints)

# Reshape back into an array
greedy_power = greedy_power.reshape((N_bins_wd, N_bins_ws, full_scenario.n_wt))
geometric_power = geometric_power.reshape((N_bins_wd, N_bins_ws, full_scenario.n_wt))

# Extract the prevalence
prevalence = wind_rose.wind_rose['prevalence'].to_numpy().reshape((N_bins_wd, N_bins_ws))

# Compute the AEP
metrics = moct.Metrics()
greedy_aep = metrics.compute_aep(greedy_power, params={'prevalence': prevalence})
geometric_aep = metrics.compute_aep(geometric_power, params={'prevalence': prevalence})

# Compute the differences
greedy_power_diff = np.nansum(geometric_power - greedy_power, axis=2)

# ------------ PRINTING ------------

# Print the wind-rose data
print(f"--- WIND ROSE ---")
print(wind_rose)

# Print the total AEP
print(f"\n--- AEP ---")
print(f"Greedy AEP: {greedy_aep * 1E-9 / 3600:.2f} GWh")
print(f"Geometric AEP: {geometric_aep * 1E-9 / 3600:.2f} GWh (\033[34m{'+' if np.sign((geometric_aep - greedy_aep) / greedy_aep) >= 0 else '-'}{(geometric_aep - greedy_aep) / greedy_aep * 100:.2f}%\033[0m)")

# ------------ PLOTTING ------------

# Set the viewing range
xy_range = [[0, 2500], [0, 2500]]

# Plot the wind-farm layout
fig_layout, ax_layout = plt.subplots()
ax_layout = plot.layout(empty_scenario, None, ax_exist=ax_layout)
ax_layout.set_xlim(xy_range[0])
ax_layout.set_ylim(xy_range[1])
ax_layout.set_aspect('equal')
fig_layout.suptitle("Wind-farm layout")

# Plot the wind rose
fig_wind_rose, ax_wind_rose = plot.wind_rose(wind_rose, threshold=0.001)
fig_wind_rose.suptitle("Wind rose")

# TEMP
print(utils.devisors(N_bins_wd))

# Plot the difference in power production
fig_diff_power, ax_diff_power = plt.subplots()
cs = ax_diff_power.imshow(np.rot90(greedy_power_diff, -1) * 1E-3, cmap='coolwarm', vmin=-np.nanmax(np.abs(greedy_power_diff)) * 1E-3, vmax=np.nanmax(np.abs(greedy_power_diff)) * 1E-3, extent=[0 - (1/2 * (360 / N_bins_wd)), 360 + (1/2 * (360 / N_bins_wd)), 0 - (1/2 * (25 / N_bins_ws)), 25 + (1/2 * (25 / N_bins_ws))], aspect='auto')
cbar = fig_diff_power.colorbar(cs, ax=ax_diff_power)
cbar.set_label('Power difference (in kW)')
ax_diff_power.set_xticks(np.linspace(0, 360, np.max(utils.devisors(N_bins_wd)[utils.devisors(N_bins_wd) <= 10])))
ax_diff_power.set_yticks(np.linspace(0, 25, np.max(utils.devisors(N_bins_ws)[utils.devisors(N_bins_ws) <= 10])))
ax_diff_power.set_xlabel("Wind direction (in °)")
ax_diff_power.set_ylabel("Wind speed (in m/s)")
fig_diff_power.suptitle("Power difference")

# Show the plots
# plt.close(fig_layout)
# plt.close(fig_wind_rose)
# plt.close(fig_diff_power)
plt.show()