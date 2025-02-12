"""
Auxiliary function for plotting the results of the twain module.
"""

# Import packages
from pathlib import Path
from typing import TypeVar, TypeAlias

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path as mpltPath
import matplotlib.tri as tri

from twain.moct import Scenario, ControlSetpoints, WindFarmModel, WindRose

# ------ STATIC ------

PATH_WT_TOP_ICON = Path(r'./assets/icons/icon_wt_top.png')

# ------ METHODS ------

def layout(scenario: Scenario, control_setpoints: ControlSetpoints = None, ax_exist: Axes = None) -> tuple[Figure, Axes]:
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig, ax = plt.subplots()
    #: Set text offset
    TEXT_OFFSET = np.array([10, 10])
    #: Check if control setpoints are provided
    if control_setpoints is None:
        control_setpoints = ControlSetpoints(yaw_angles=np.zeros(scenario.n_wt), power_setpoints=np.random.uniform(0, 1, scenario.n_wt))
    #: Convert the wind-farm layout to a numpy array
    array_layout = scenario.wf_layout.to_numpy()
    #: Plot the power lines if provided
    if scenario.power_lines is not None:
        for group in scenario.power_lines:
            ax.plot(group[0][:, 0], group[0][:, 1], '--', color='gray', linewidth=2)  # Main power line
            for line in group[1]:
                ax.plot(line[:, 0], line[:, 1], '--', color='gray', linewidth=1)  # Secondary power line
    #: Plot the turbines as images
    for wt_idx in range(array_layout.shape[0]):
        #: Extract and rotate the image
        # NOTE: 'sp.ndimage.rotate' uses clockwise rotation, so that's why the minus sign is used
        ab = AnnotationBbox(OffsetImage(sp.ndimage.rotate(plt.imread(PATH_WT_TOP_ICON), -(scenario.theta + control_setpoints.yaw_angles[wt_idx])), zoom=0.1), array_layout[wt_idx, :], frameon=False)
        ax.add_artist(ab)
        ax.annotate(scenario.wt_names[wt_idx], array_layout[wt_idx, :] + TEXT_OFFSET, fontsize=8)
    #: Plot the actual center points
    ax.plot(array_layout[:, 0], array_layout[:, 1], 'bx', markersize=2, zorder=3)
    #: Plot the contour
    ax.plot(np.append(scenario.perimeter[:, 0], scenario.perimeter[0, 0]), np.append(scenario.perimeter[:, 1], scenario.perimeter[0, 1]), 'r--', lw=2)
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax


def noise_field(scenario: Scenario, control_setpoints: ControlSetpoints, ax_exist: Axes = None, clip: bool = True) -> tuple[Figure, Axes]:
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig, ax = plt.subplots()
    #: Compute the noise from the wind farm
    wf_model = WindFarmModel(scenario)
    _, _, (X, Y, noise_field) = wf_model.impact_control_variables(control_setpoints)
    #: Convert the wind-farm layout to a numpy array
    array_layout = scenario.wf_layout.to_numpy()
    #: Clip the noise outside the perimeter
    if clip:
        for idx, _ in np.ndenumerate(noise_field):
            if not mpltPath(scenario.perimeter, closed=False).contains_point([X[idx], Y[idx]]):
                noise_field[idx] = np.nan
    #: Plot the noise field
    cs = ax.imshow(noise_field, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', aspect='auto')
    #: Add a colorbar
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Noise level (in dB)')
    #: Plot the actual center points
    ax.plot(array_layout[:, 0], array_layout[:, 1], 'x', markersize=4, color='blue', linewidth=16, zorder=3)
    #: Plot the contour
    ax.plot(np.append(scenario.perimeter[:, 0], scenario.perimeter[0, 0]), np.append(scenario.perimeter[:, 1], scenario.perimeter[0, 1]), 'r--', lw=2)
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax
    

def flow_field(scenario: Scenario, control_setpoints: ControlSetpoints, ax_exist: Axes = None, clip: bool = True) -> tuple[Figure, Axes]:
    # TODO: Merge this with the previous method, to a unifying method 'plot_field'
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig, ax = plt.subplots()
    #: Compute the noise from the wind farm
    wf_model = WindFarmModel(scenario)
    X, Y, flow_field = wf_model.get_flow_field(control_setpoints)
    #: Convert the wind-farm layout to a numpy array
    array_layout = scenario.wf_layout.to_numpy()
    #: Clip the noise outside the perimeter
    if clip:
        for idx, _ in np.ndenumerate(flow_field):
            if not mpltPath(scenario.perimeter, closed=False).contains_point([X[idx], Y[idx]]):
                flow_field[idx] = np.nan
    #: Plot the noise field
    cs = ax.imshow(flow_field, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', aspect='auto')
    #: Add a colorbar
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Wind speed (in m/s)')
    cs.set_clim([0, flow_field.max()])
    #: Plot the actual center points
    ax.plot(array_layout[:, 0], array_layout[:, 1], 'x', markersize=4, color='blue', linewidth=16, zorder=3)
    #: Plot the contour
    ax.plot(np.append(scenario.perimeter[:, 0], scenario.perimeter[0, 0]), np.append(scenario.perimeter[:, 1], scenario.perimeter[0, 1]), 'r--', lw=2)
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax
    

def wind_rose(wind_rose: WindRose, ax_exist: Axes = None) -> tuple[Figure, Axes]:
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='polar')
    #: Set oriantation parameters
    # FROM: https://gist.github.com/phobson/41b41bdd157a2bcf6e14  # nopep8
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')
    #: Unravel the data
    # TEMP
    #
    print(wind_rose.wind_rose)
    print(wind_rose.wind_speeds)
    #
    X, Y = np.meshgrid(wind_rose.wind_rose['direction'], wind_rose.wind_speeds, indexing='xy')
    Z = np.zeros(X.shape)
    for idx, _ in np.ndenumerate(Z):
        Z[idx] = wind_rose.wind_rose.iloc[idx[1]]['frequencies'][idx[0]]
    #: Create a contour plot
    ax.tricontourf(X.flatten(order='F'), Y.flatten(order='F'), Z.flatten(order='F'), origin='lower')
    #: Set the labels
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax