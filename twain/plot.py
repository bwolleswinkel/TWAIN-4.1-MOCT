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

from twain.moct import Scenario, ControlSetpoints, WindFarmModel

# ------ STATIC ------

PATH_WT_TOP_ICON = Path(r'./assets/icons/icon_wt_top.png')

# ------ METHODS ------

def layout(scenario: Scenario, ax_exist: Axes = None) -> tuple[Figure, Axes]:
    #: Add to existing axes (if they exist)
    if ax_exist is not None:
        #: Set equal
        fig, ax = None, ax_exist
    else:
        #: Create a figure
        fig, ax = plt.subplots()
    #: Convert the wind-farm layout to a numpy array
    array_layout = scenario.wf_layout.to_numpy()
    #: Plot the turbines as images
    for wt_loc in array_layout:
        #: Extract and rotate the image
        ab = AnnotationBbox(OffsetImage(sp.ndimage.rotate(plt.imread(PATH_WT_TOP_ICON), np.random.uniform(-20, 20)), zoom=0.1), wt_loc, frameon=False)
        ax.add_artist(ab)
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