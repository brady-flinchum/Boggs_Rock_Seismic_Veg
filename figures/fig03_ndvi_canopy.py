"""
fig03_ndvi_canopy.py
--------------------
Figure 3: NDVI and canopy height along L1, with NDVI–canopy height scatter
          coloured by near-surface seismic velocity.

Panel (a) — NDVI along L1 (mean ± 1 s.d. shading) with soil-sample
            locations coloured by velocity.
Panel (b) — Topographic cross-section of L1 with canopy point cloud
            (coloured by height) and soil-sample depths.
Panel (c) — NDVI vs canopy-height scatter with velocity colour scale,
            overlaid on background KDE density (pre-computed or computed here).

Output
------
outputs/fig03_ndvi_canopy.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.boggs_data import boggs_topo_struct, DOMAIN, REPO_ROOT
from src.plot_utils import (build_forest_colormap, set_times_font,
                             VELOCITY_CMAP, V_MIN, V_MAX)

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
bt = boggs_topo_struct(DOMAIN,read_seismic=True)

print("Extracting L1 profile …")
l1_veg_x, l1_veg_elv, l1_veg_h, l1_topo = bt.extract_pc_along_path(
    bt.l1_e, bt.l1_n, bt.l1_ll, N=1000, width=2)
keep = l1_veg_h < 35
l1_veg_x, l1_veg_elv, l1_veg_h = l1_veg_x[keep], l1_veg_elv[keep], l1_veg_h[keep]

l1_ndvi     = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.NDVI)
l1_ndvi_std = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.NDVI_std)
l1_rst_vegH = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.vegH)
l1_rst_elv  = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.elv_2019)

# ---------------------------------------------------------------------------
# Near-surface velocity at each soil sample location
# ---------------------------------------------------------------------------
z0       = 5        # depth window (m)
win_width = 5       # horizontal half-window (m)

s_vel     = np.zeros(len(bt.s_distance))
s_stdvel  = np.zeros(len(bt.s_distance))
for i, x0 in enumerate(bt.s_distance):
    ind = np.where(
        np.logical_and(bt.seis_L1_cc[:, 0] <= x0 + win_width,
                       bt.seis_L1_cc[:, 0] >  x0 - win_width))[0]
    tmp_cc    = bt.seis_L1_cc[ind, :2]
    tmp_vels  = bt.seis_L1_vel[ind]
    tmp_depths = np.interp(tmp_cc[:, 0], bt.l1_ll, l1_topo) - tmp_cc[:, 1]
    shallow   = tmp_depths <= z0
    s_vel[i]    = np.mean(tmp_vels[shallow])
    s_stdvel[i] = np.std(tmp_vels[shallow])

bt.s_vel = s_vel

# ---------------------------------------------------------------------------
# Background KDE (NDVI vs canopy height over whole study grid)
# ---------------------------------------------------------------------------
print("Computing KDE background …")
X, Y = np.meshgrid(np.arange(-0.15, 0.5, 0.01),
                   np.arange(0, 40, 0.1))
ndvi_flat = bt.NDVI.ravel()
vegH_flat = bt.vegH.ravel()
valid     = ~np.isnan(ndvi_flat)
kernel    = stats.gaussian_kde(np.vstack([ndvi_flat[valid], vegH_flat[valid]]))
Z         = np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)

# ---------------------------------------------------------------------------
# Marker lookup by ecological group
# ---------------------------------------------------------------------------
markers = {0: "o", 1: "D", 2: "P"}   # thin soil, outcrop shrub, oak/hickory
sizes   = {0: 20,  1: 20,  2: 40}
labels  = {0: "Thin Soil Cover (Moss/Lichen)",
           1: "Outcrop Shrub/Forest",
           2: "Oak/Hickory Forest"}

forest_cmap = build_forest_colormap()
set_times_font(14)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=[8, 4])
gs  = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(gs[0, 0:2])
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()
ax1.set_xlim([-5, 195])
ax1.yaxis.set_major_locator(MultipleLocator(0.2))
ax1.xaxis.set_major_locator(MultipleLocator(25))
ax1.set_xlabel("Distance (m)")
ax1.set_ylabel("NDVI")

ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax1)
ax2.yaxis.set_major_locator(MultipleLocator(15))
ax2.xaxis.set_major_locator(MultipleLocator(25))
ax2.set_xlabel("Distance (m)")
ax2.set_ylabel("Elevation (m)")

ax5 = fig.add_subplot(gs[0:2, 2:4])
ax5.yaxis.set_label_position("right")
ax5.yaxis.tick_right()

# --- Panel (a): NDVI along L1 ---
ax1.fill_between(bt.l1_ll,
                 l1_ndvi - l1_ndvi_std,
                 l1_ndvi + l1_ndvi_std,
                 color="gray", alpha=0.5)
ax1.plot(bt.l1_ll, l1_ndvi, c="k", lw=2)

for g, (mk, sz) in enumerate(zip(markers.values(), sizes.values())):
    mask = bt.s_groups == g
    ax1.scatter(bt.s_distance[mask], bt.s_distance[mask] * 0,
                s=sz, c=s_vel[mask], marker=mk, ec="k", zorder=3,
                cmap=VELOCITY_CMAP, vmin=V_MIN, vmax=V_MAX)

# --- Panel (b): cross-section ---
ax2.scatter(l1_veg_x, l1_veg_elv, c=l1_veg_h,
            cmap=forest_cmap, s=1, vmin=0, vmax=10)
ax2.plot(bt.l1_ll, l1_topo, c="k", lw=2)
ax2.plot(bt.l1_ll, l1_rst_vegH + l1_rst_elv, c="k", lw=1)

for g, (mk, sz) in enumerate(zip(markers.values(), sizes.values())):
    mask = bt.s_groups == g
    depth_offset = {0: 0, 1: -5, 2: -10}[g]
    ax2.scatter(bt.s_distance[mask],
                bt.soil_elv[mask] - bt.s_maxDepth[mask] + depth_offset,
                s=sz if g < 2 else 40,
                c=s_vel[mask], marker=mk, ec="k", zorder=3,
                cmap=VELOCITY_CMAP, vmin=V_MIN, vmax=V_MAX)

ax2.set_aspect(1, adjustable="datalim")

# --- Panel (c): NDVI vs canopy height ---
ax5.pcolormesh(X, Y, Z, cmap="cubehelix_r", vmin=0, vmax=0.25, alpha=0.5)
for g, (mk, sz, lbl) in enumerate(zip(markers.values(), sizes.values(), labels.values())):
    mask = bt.s_groups == g
    cbar = ax5.scatter(bt.soil_ndvi[mask], bt.soil_vegH[mask],
                       s=50, marker=mk, c=s_vel[mask],
                       ec="k", cmap=VELOCITY_CMAP, vmin=V_MIN, vmax=V_MAX,
                       label=lbl)

plt.colorbar(cbar, ax=ax5, label="Velocity of Soil (m/s)", shrink=0.8)
ax5.legend(fontsize="x-small", loc="upper left")
ax5.set_xlim([-0.15, 0.45])
ax5.set_ylim([0, 35])
ax5.set_xlabel("NDVI")
ax5.set_ylabel("Canopy Height (m)")

ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, fontsize=16, va="top")
ax2.text(0.01, 0.98, "(b)", transform=ax2.transAxes, fontsize=16, va="top")
ax5.text(0.98, 0.02, "(c)", transform=ax5.transAxes, fontsize=16,
         va="bottom", ha="right")

out_path = os.path.join(OUT_DIR, "fig03_ndvi_canopy.png")
fig.savefig(out_path, dpi=300)
print(f"Saved → {out_path}")
