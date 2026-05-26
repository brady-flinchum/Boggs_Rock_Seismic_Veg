"""
fig01_map.py
------------
Figure 1: Study-area map.

Panel (a) — Planet Labs RGB image with topographic contours and seismic lines.
Panel (b) — Same extent with canopy-height (green) and exposed-rock NDVI
            (turbo) overlays.
Panel (c) — Field photograph of the outcrop.

Output
------
outputs/fig01_map.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Make the repo root importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.boggs_data import boggs_topo_struct, DOMAIN, REPO_ROOT
from src.plot_utils import panel_label

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
bt = boggs_topo_struct(DOMAIN)

# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------
lvls = np.arange(240, 296, 1)

masked_ndvi = np.copy(bt.NDVI)
masked_ndvi[masked_ndvi > 0.1] = np.nan          # expose bare rock pixels only

rgb = np.dstack([bt.planet_R_fullRes,
                 bt.planet_G_fullRes,
                 bt.planet_B_fullRes])

# Field photograph
photo_path = os.path.join(REPO_ROOT, "data", "FieldPhotos",
                          "11-09-2024-DSC_8653-Pano(3).jpg")
img = mpimg.imread(photo_path)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(constrained_layout=True, figsize=[7, 9])
gs  = fig.add_gridspec(5, 3)

ax1 = fig.add_subplot(gs[0:3, 0:2])
ax1.ticklabel_format(style="plain", axis="both")
ax1.set_aspect(1)
ax1.xaxis.set_label_position("top")
ax1.xaxis.tick_top()

ax2 = fig.add_subplot(gs[0:3, 1:3], sharex=ax1, sharey=ax1)
ax2.set_aspect(1)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position("top")
ax2.xaxis.tick_top()

ax3 = fig.add_subplot(gs[3:5, 0:3])

# ---------------------------------------------------------------------------
# Panel (a) — RGB + contours + lines
# ---------------------------------------------------------------------------
ax1.imshow(rgb, extent=[
    np.min(bt.planet_easting_fullRes),  np.max(bt.planet_easting_fullRes),
    np.min(bt.planet_northing_fullRes), np.max(bt.planet_northing_fullRes),
])
ax1.contour(bt.EE, bt.NN, bt.elv_2019, levels=lvls,
            colors="w", linewidths=1, alpha=0.3)
ax1.scatter(345046.73, 3852958.29, marker="*", s=150, color="w", ec="k")
ax1.plot(bt.l1_e, bt.l1_n, c="tab:purple", lw=2)
ax1.plot(bt.l2_e, bt.l2_n, c="tab:orange", lw=2)
ax1.scatter(bt.l1_e[0], bt.l1_n[0], c="tab:purple", ec="k", s=50, zorder=5)
ax1.scatter(bt.l2_e[0], bt.l2_n[0], c="tab:orange", ec="k", s=50, zorder=5)
ax1.set_xlabel("Easting (m)")
ax1.set_ylabel("Northing (m)")
ax1.set_xlim([DOMAIN["min x"], DOMAIN["max x"]])
ax1.set_ylim([DOMAIN["min y"], DOMAIN["max y"]])

# ---------------------------------------------------------------------------
# Panel (b) — vegetation height + bare-rock NDVI
# ---------------------------------------------------------------------------
ax2.imshow(rgb, extent=[np.min(bt.EE), np.max(bt.EE),
                         np.min(bt.NN), np.max(bt.NN)])
cbar_veg  = ax2.pcolormesh(bt.EE, bt.NN, bt.vegH,
                            vmin=0, vmax=30, cmap="Greens")
cbar_ndvi = ax2.pcolormesh(bt.EE, bt.NN, masked_ndvi,
                            vmin=-0.08, vmax=0.02, cmap="turbo")
ax2.contour(bt.EE, bt.NN, bt.elv_2019, levels=lvls,
            colors="k", linewidths=1, alpha=0.3)
ax2.scatter(345046.73, 3852958.29, marker="*", s=150, color="w", ec="k")
ax2.plot(bt.l1_e, bt.l1_n, c="tab:purple", lw=2)
ax2.plot(bt.l2_e, bt.l2_n, c="tab:orange", lw=2)
ax2.scatter(bt.l1_e[0], bt.l1_n[0], c="tab:purple", ec="k", s=50, zorder=5)
ax2.scatter(bt.l2_e[0], bt.l2_n[0], c="tab:orange", ec="k", s=50, zorder=5)
ax2.set_xlabel("Easting (m)")
ax2.set_ylabel("Northing (m)")

plt.colorbar(cbar_ndvi, ax=[ax1, ax2], orientation="horizontal",
             location="top",    shrink=0.5, label="NDVI")
plt.colorbar(cbar_veg,  ax=[ax1, ax2], orientation="horizontal",
             location="bottom", shrink=0.5, label="Canopy Height (m)")

# ---------------------------------------------------------------------------
# Panel (c) — field photograph
# ---------------------------------------------------------------------------
ax3.imshow(img, origin="upper")
ax3.axis("off")
ax3.set_aspect("equal")
ax3.set_anchor("C")

panel_label(ax1, "(a)")
panel_label(ax2, "(b)")
panel_label(ax3, "(c)")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = os.path.join(OUT_DIR, "fig01_map.png")
fig.savefig(out_path, dpi=300)
print(f"Saved → {out_path}")
