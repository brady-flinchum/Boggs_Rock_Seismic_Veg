"""
figS1_ndvi_timeseries.py
------------------------
Supplemental Figure S1: NDVI time series along Line L1 and NDVI maps
for each acquisition date.

Panel set 1 (saved as figS1a_ndvi_L1_profile.png)
  — NDVI along L1 for all dates with mean ± s.d. envelope.

Panel set 2 (saved as figS1b_ndvi_maps.png)
  — 7-panel map grid (one per acquisition date).

Output
------
outputs/figS1a_ndvi_L1_profile.png
outputs/figS1b_ndvi_maps.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.boggs_data import boggs_topo_struct, DOMAIN, REPO_ROOT

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data …")
bt = boggs_topo_struct(DOMAIN)

l1_ndvi     = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.NDVI)
l1_ndvi_std = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, bt.NDVI_std)

# ---------------------------------------------------------------------------
# Panel S1a — NDVI profile along L1 for every date
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=[7, 6], constrained_layout=True)
ax.fill_between(bt.l1_ll,
                l1_ndvi - l1_ndvi_std,
                l1_ndvi + l1_ndvi_std,
                color="gray", alpha=0.5)

for i, ndvi_grid in enumerate(bt.all_NDVI):
    tmp = bt.extract_raster_values(bt.l1_e, bt.l1_n, bt.EE, bt.NN, ndvi_grid)
    lbl = bt.NAIP["Dates"][i] if i < 6 else np.datetime64("2024-08-02")
    ax.plot(bt.l1_ll, tmp, label=str(lbl))

ax.plot(bt.l1_ll, l1_ndvi, c="k", lw=4, label="Mean")
ax.set_xlabel("Distance (m)")
ax.set_ylabel("NDVI")
ax.set_ylim([-0.5, 1.2])
ax.set_xlim([0, 190])
ax.legend(loc="upper left", fontsize="small", ncol=2)

out1 = os.path.join(OUT_DIR, "figS1a_ndvi_L1_profile.png")
fig.savefig(out1, dpi=300)
plt.close(fig)
print(f"Saved → {out1}")

# ---------------------------------------------------------------------------
# Panel S1b — NDVI maps for each date
# ---------------------------------------------------------------------------
n_dates   = len(bt.all_NDVI)
date_labels = [str(d) for d in bt.NAIP["Dates"]] + ["2024-08-02"]

fig, axes = plt.subplots(1, n_dates, figsize=[10.5, 3.5],
                          constrained_layout=True)
vmin, vmax = -0.3, 1.0

for i, (ax, lbl) in enumerate(zip(axes, date_labels)):
    ax.set_title(lbl, fontsize="medium")
    cbar_im = ax.pcolormesh(bt.EE, bt.NN, bt.all_NDVI[i],
                            vmin=vmin, vmax=vmax, cmap="cubehelix_r")
    ax.plot(bt.l1_e, bt.l1_n, c="k", ls="--")
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])

plt.colorbar(cbar_im, ax=axes.tolist(),
             orientation="horizontal", location="bottom",
             shrink=0.5, label="NDVI")

out2 = os.path.join(OUT_DIR, "figS1b_ndvi_maps.png")
fig.savefig(out2, dpi=300)
plt.close(fig)
print(f"Saved → {out2}")
