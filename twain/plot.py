"""
Auxiliary function for plotting the results of the twain module.
"""

# Import packages
from pathlib import Path
from typing import TypeVar, TypeAlias

import numpy as np
import scipy as sp
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path as mpltPath
import matplotlib.tri as tri

from twain.moct import Scenario, ControlSetpoints, WindFarmModel, WindRose, SpatialArray

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

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


def noise_field(scenario: Scenario, control_setpoints: ControlSetpoints, xy_range: tuple[tuple[float, float], tuple[float, float]] = None, N_points: int = 1000, noise_mask: tuple[NPArray[float], NPArray[float], NPArray[float]] = None, ax_exist: Axes = None, clip: bool = True) -> tuple[Figure, Axes]:
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig, ax = plt.subplots()
    #: Compute the noise from the wind farm
    wf_model = WindFarmModel(scenario)
    _, _, (X, Y, noise_field) = wf_model.impact_control_variables(control_setpoints, xy_range, N_points)
    #: Convert the wind-farm layout to a numpy array
    array_layout = scenario.wf_layout.to_numpy()
    #: Clip the noise outside the perimeter
    if clip:
        for idx, _ in np.ndenumerate(noise_field):
            if not mpltPath(scenario.perimeter, closed=False).contains_point([X[idx], Y[idx]]):
                noise_field[idx] = np.nan
    #: Check if a noise mask is provided
    if noise_mask is not None:
        #: Unpack the noise mask
        X_mask, Y_mask, noise_mask = noise_mask
        #: Convert to SpatialArray
        noise_mask = SpatialArray((X_mask, Y_mask), noise_mask)
        #: Loop over all values in the noise field
        for idx, _ in np.ndenumerate(noise_field):
            #: Retrieve the x- and y-values
            x, y = X[idx], Y[idx]
            #: Check if the point is within the mask
            noise_field[idx] = noise_field[idx] if noise_mask[x, y] else np.nan
        # FIXME: Something is wrong, we have to 'flip' the matrix
        noise_field = np.rot90(noise_field, k=1)
        #: Compute the maximum noise
        idx_max_noise = np.nanargmax(noise_field)
        x_max_noise, y_max_noise, max_noise = X[np.unravel_index(idx_max_noise, noise_field.shape)], Y[np.unravel_index(idx_max_noise, noise_field.shape)], noise_field[np.unravel_index(idx_max_noise, noise_field.shape)]
    else: 
        max_noise = None
    #: Plot the noise field
    cs = ax.imshow(noise_field, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', aspect='auto')
    #: Plot the maximum noise value
    if max_noise is not None:
        ax.plot(x_max_noise, y_max_noise, 'x', markersize=2, color='red', markeredgewidth=12, zorder=3)
        ax.annotate(f'{max_noise:.2f} dB', xy=(x_max_noise, y_max_noise), xytext=(x_max_noise + 50, y_max_noise + 50), fontsize=8)
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
    

def flow_field(scenario: Scenario, control_setpoints: ControlSetpoints, xy_range: tuple[tuple[float, float], tuple[float, float]], N_points: int = 1000, ax_exist: Axes = None, clip: bool = True) -> tuple[Figure, Axes]:
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
    X, Y, flow_field = wf_model.get_flow_field(control_setpoints, xy_range, N_points)
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
    

def wind_rose(wind_rose: WindRose, threshold: float = None, v_cutin_cutout: tuple[float, float] = None, ax_exist: Axes = None) -> tuple[Figure, Axes]:
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
    T, R = np.meshgrid(np.linspace(0, 2 * np.pi, wind_rose.n_bins_wd), np.linspace(0, 25, wind_rose.n_bins_ws))
    #: Filter the data with a threshold
    if threshold is not None:
        wind_rose.data[wind_rose.data < threshold] = np.nan
    #: Clip the data with a cut-in and cut-out speed
    if v_cutin_cutout is not None:
        wind_rose.data[R[:, 0] < v_cutin_cutout[0], :] = np.nan
        wind_rose.data[R[:, 0] > v_cutin_cutout[1], :] = np.nan
    #: Create a contour plot
    cs = ax.pcolormesh(T, R, wind_rose.data, edgecolors='face', cmap='inferno')
    #: Add a colorbar
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Prevalence (%)')
    #: Set the labels
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    #: Change the color of the y-labels
    ax.tick_params(axis='y', colors='green')
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax