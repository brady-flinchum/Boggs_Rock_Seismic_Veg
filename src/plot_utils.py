"""
plot_utils.py
-------------
Shared plotting helpers used across all figure scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def panel_label(ax, label, color="w", fontsize=18, loc="top-left"):
    """Add a bold panel label (e.g. '(a)') to an axes corner."""
    x_pos = 0.02 if "left" in loc else 0.98
    y_pos = 0.98 if "top"  in loc else 0.02
    ha    = "left" if "left" in loc else "right"
    va    = "top"  if "top"  in loc else "bottom"
    ax.text(x_pos, y_pos, label,
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va=va, ha=ha, color=color)


def build_forest_colormap():
    """Brown-to-light-green colormap representing canopy height."""
    c2 = np.array((170, 125,  96)) / 255
    c6 = np.array(( 55,  78,  64)) / 255
    c7 = np.array(( 73, 106,  80)) / 255
    c8 = np.array((103, 150, 110)) / 255
    return LinearSegmentedColormap.from_list(
        "BrownToLightGreen", [c2, c6, c7, c8], N=256
    )


def set_times_font(size=14):
    """Switch matplotlib to Times New Roman."""
    plt.rc("font", family="Times New Roman", weight="regular", size=size)


VELOCITY_CMAP = "turbo_r"
V_MIN, V_MAX  = 300, 4500   # seismic velocity colour scale limits
MASK_THRESH   = 0.2          # ray-coverage threshold for masking
