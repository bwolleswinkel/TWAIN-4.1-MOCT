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

from twain.moct import Scenario

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
    #: Extract a convex hull
    hull_layout = sp.spatial.ConvexHull(array_layout)
    #: Plot the turbines as images
    for wt_loc in array_layout:
        #: Extract and rotate the image
        ab = AnnotationBbox(OffsetImage(sp.ndimage.rotate(plt.imread(PATH_WT_TOP_ICON), np.random.uniform(-20, 20)), zoom=0.1), wt_loc, frameon=False)
        ax.add_artist(ab)
    #: Plot the actual center points
    ax.plot(array_layout[:, 0], array_layout[:, 1], 'bx', markersize=2, zorder=3)
    #: Plot the contour
    ax.plot(np.append(array_layout[hull_layout.vertices, 0], array_layout[hull_layout.vertices[0], 0]), np.append(array_layout[hull_layout.vertices, 1], array_layout[hull_layout.vertices[0], 1]), 'r--', lw=2)
    #: Return the axis
    if ax_exist is not None:
        return ax
    else:
        return fig, ax
