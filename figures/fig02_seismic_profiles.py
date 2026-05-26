"""
fig02_seismic_profiles.py
-------------------------
Figure 2: Seismic P-wave velocity cross-sections with canopy point cloud.

Panel (a) — Line L1: canopy point cloud coloured by height, topographic
            surface, canopy top from raster, and seismic velocity.
Panel (b) — Line L2: same as (a).

Output
------
outputs/fig02_seismic_profiles.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.boggs_data import boggs_topo_struct, DOMAIN, REPO_ROOT
from src.plot_utils import (build_forest_colormap, panel_label,
                             VELOCITY_CMAP, V_MIN, V_MAX, MASK_THRESH)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data & extract profiles
# ---------------------------------------------------------------------------
print("Loading data …")
bt = boggs_topo_struct(DOMAIN,read_seismic=True)

print("Extracting L1 point-cloud profile …")
l1_veg_x, l1_veg_elv, l1_veg_h, l1_topo = bt.extract_pc_along_path(
    bt.l1_e, bt.l1_n, bt.l1_ll, N=1000, width=2)
l1_rst_vegH = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.vegH)
l1_rst_elv  = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.elv_2019)

# Remove extreme outliers (e.g. birds / noise > 35 m canopy)
keep = l1_veg_h < 35
l1_veg_x, l1_veg_elv, l1_veg_h = (l1_veg_x[keep], l1_veg_elv[keep], l1_veg_h[keep])

print("Extracting L2 point-cloud profile …")
l2_veg_x, l2_veg_elv, l2_veg_h, l2_topo = bt.extract_pc_along_path(
    bt.l2_e, bt.l2_n, bt.l2_ll, N=1000, width=2)
l2_rst_vegH = bt.extract_raster_values(bt.l2_e, bt.l2_n, bt.EE, bt.NN, bt.vegH)
l2_rst_elv  = bt.extract_raster_values(bt.l2_e, bt.l2_n, bt.EE, bt.NN, bt.elv_2019)

# ---------------------------------------------------------------------------
# Build masked triangulations for seismic velocity
# ---------------------------------------------------------------------------
def make_masked_tri(nodes, cv, ray_cov):
    tri_obj  = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=cv)
    mask_vals = (ray_cov >= MASK_THRESH).astype(float)
    tri_obj.set_mask(1 - mask_vals)
    return tri_obj

L1_tri = make_masked_tri(bt.seis_L1_nodes, bt.seis_L1_cv, bt.seis_L1_rayCov)
L2_tri = make_masked_tri(bt.seis_L2_nodes, bt.seis_L2_cv, bt.seis_L2_rayCov)

forest_cmap = build_forest_colormap()
z_depth     = -5    # depth of white dashed reference line below surface

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=[7, 4])
gs  = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()
ax1.set_xlim([0, 190])
ax1.set_ylim([240, 295])
ax1.set_aspect(1)
ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("Elevation (m)")

ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1, sharey=ax1)
ax2.set_aspect(1)
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Elevation (m)")

# --- L1 ---
ax1.scatter(l1_veg_x, l1_veg_elv, c=l1_veg_h,
            cmap=forest_cmap, s=1, vmin=0, vmax=10)
ax1.plot(bt.l1_ll, l1_topo, c="k", lw=2)
ax1.plot(bt.l1_ll, l1_rst_vegH + l1_rst_elv, c="k", lw=1)
ax1.tripcolor(L1_tri, bt.seis_L1_vel, cmap=VELOCITY_CMAP, vmin=V_MIN, vmax=V_MAX)
ax1.plot(bt.l1_ll, l1_topo + z_depth, c="w", lw=1, ls="--")

# --- L2 ---
ax2.scatter(l2_veg_x, l2_veg_elv, c=l2_veg_h,
            cmap=forest_cmap, s=1, vmin=0, vmax=10)
ax2.plot(bt.l2_ll, l2_topo, c="k", lw=2)
ax2.plot(bt.l2_ll, l2_rst_vegH + l2_rst_elv, c="k", lw=1)
cbar_vel = ax2.tripcolor(L2_tri, bt.seis_L2_vel,
                          cmap=VELOCITY_CMAP, vmin=V_MIN, vmax=V_MAX)
ax2.plot(bt.l2_ll, l2_topo + z_depth, c="w", lw=1, ls="--")

plt.colorbar(cbar_vel, ax=[ax1, ax2], orientation="vertical",
             location="right", label="Velocity (m/s)", shrink=0.8)

panel_label(ax1, "(a)", color="k", fontsize=16, loc="bottom-left")
panel_label(ax2, "(b)", color="k", fontsize=16, loc="bottom-left")

out_path = os.path.join(OUT_DIR, "fig02_seismic_profiles.png")
fig.savefig(out_path, dpi=300)
print(f"Saved → {out_path}")
