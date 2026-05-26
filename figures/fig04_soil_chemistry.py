"""
fig04_soil_chemistry.py
-----------------------
Figure 4: Soil geochemistry by ecological group.

Each attribute is saved as a separate PNG.  The four panels used in the
paper (assembled in Illustrator) are:
    fig04a_soil_pH.png         — Soil pH
    fig04b_soil_CEC.png        — CEC (meq/100g)
    fig04c_soil_P.png          — Phosphorus (mg/cm³)
    fig04d_soil_CN.png         — C:N ratio

All other attributes are also saved for completeness / supplementary use.

Layout per panel (shared helper `_plot_attribute`)
---------------------------------------------------
Top    : horizontal box-and-whisker by group
Bottom : NDVI vs canopy-height scatter coloured by attribute, overlaid
         on KDE density of the full study grid.

Output
------
outputs/fig04a_soil_pH.png  … (plus all other attributes)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.boggs_data import boggs_topo_struct, DOMAIN, REPO_ROOT

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data …")
bt = boggs_topo_struct(DOMAIN,read_seismic=True)

# Near-surface velocity (needed for panel layout — re-used from fig03)
l1_veg_x, l1_veg_elv, l1_veg_h, l1_topo = bt.extract_pc_along_path(
    bt.l1_e, bt.l1_n, bt.l1_ll, N=1000, width=2)

z0, win_width = 5, 5
s_vel = np.zeros(len(bt.s_distance))
for i, x0 in enumerate(bt.s_distance):
    ind = np.where(
        (bt.seis_L1_cc[:, 0] <= x0 + win_width) &
        (bt.seis_L1_cc[:, 0] >  x0 - win_width))[0]
    tmp_depths = (np.interp(bt.seis_L1_cc[ind, 0], bt.l1_ll, l1_topo)
                  - bt.seis_L1_cc[ind, 1])
    shallow = tmp_depths <= z0
    s_vel[i] = np.mean(bt.seis_L1_vel[ind][shallow])

bt.s_vel = s_vel

# ---------------------------------------------------------------------------
# Background KDE
# ---------------------------------------------------------------------------
print("Computing KDE …")
X, Y = np.meshgrid(np.arange(-0.15, 0.5, 0.01),
                   np.arange(0, 40, 0.1))
ndvi_flat = bt.NDVI.ravel()
vegH_flat = bt.vegH.ravel()
valid     = ~np.isnan(ndvi_flat)
kernel    = stats.gaussian_kde(np.vstack([ndvi_flat[valid], vegH_flat[valid]]))
Z         = np.reshape(kernel(np.vstack([X.ravel(), Y.ravel()])).T, X.shape)

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
MARKERS  = {0: "o", 1: "D", 2: "P"}
COLORS   = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}
LABELS   = {0: "Thin Soil", 1: "Outcrop Forest", 2: "Oak/Hickory Forest"}
WIDTHS   = 0.9
MED_PROPS  = {"lw": 2, "c": "k"}
MEAN_PROPS = {"marker": "o", "markerfacecolor": "white",
              "markeredgecolor": "black", "markersize": 8}
FLIER_PROPS = {"marker": "x", "markerfacecolor": "white",
               "markeredgecolor": "black", "markersize": 5}


def _plot_attribute(attribute, attribute_label, cmap, vmin, vmax, out_path):
    """Generate and save one soil-chemistry figure."""
    g0 = attribute[bt.s_groups == 0]
    g1 = attribute[bt.s_groups == 1]
    g2 = attribute[bt.s_groups == 2]

    fig = plt.figure(constrained_layout=True, figsize=[7, 4])
    ax_box  = fig.add_subplot(211)
    ax_scat = fig.add_subplot(212)

    # --- box plot (top) ---
    ax_box.xaxis.set_label_position("top")
    ax_box.xaxis.tick_top()
    bp = ax_box.boxplot(
        [g2, g1, g0],                    # order: oak → outcrop → thin soil
        patch_artist=True,
        tick_labels=[LABELS[2], LABELS[1], LABELS[0]],
        notch=False, showfliers=True, vert=False, showmeans=True,
        widths=WIDTHS, medianprops=MED_PROPS,
        meanprops=MEAN_PROPS, flierprops=FLIER_PROPS,
    )
    for patch, g in zip(bp["boxes"], [2, 1, 0]):
        patch.set_facecolor(COLORS[g])
    ax_box.set_xlabel(attribute_label)

    # --- scatter (bottom) ---
    ax_scat.set_xlim([-0.15, 0.45])
    ax_scat.set_ylim([0, 35])
    ax_scat.set_xlabel("NDVI")
    ax_scat.set_ylabel("Canopy Height (m)")
    ax_scat.yaxis.set_major_locator(MultipleLocator(10))

    ax_scat.pcolormesh(X, Y, Z, cmap="cubehelix_r", vmin=0, vmax=0.25, alpha=0.5)
    for g, mk in MARKERS.items():
        mask = bt.s_groups == g
        sc   = ax_scat.scatter(
            bt.soil_ndvi[mask], bt.soil_vegH[mask],
            s=50, c=attribute[mask], ec="k",
            cmap=cmap, vmin=vmin, vmax=vmax, marker=mk,
        )
    plt.colorbar(sc, ax=[ax_box, ax_scat],
                 label=attribute_label, orientation="vertical",
                 location="right", shrink=0.8)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Define all attributes
# Note: panels used in Fig. 4 of the paper are marked with (PAPER FIG 4)
# ---------------------------------------------------------------------------
attributes = [
    # (file_suffix, attribute_array, label, cmap, vmin, vmax)
    # ---- PAPER FIGURE 4 PANELS ----
    ("fig04a_soil_pH",  bt.s_pH,       "Soil pH",                "rainbow",  4.0,  5.7),
    ("fig04b_soil_CEC", bt.s_CEC,      "Soil CEC (meq/100g)",    "rainbow",  0,   20),
    ("fig04c_soil_P",   bt.s_P,        r"P (mg/cm$^3$)",         "rainbow",  0,    0.025),
    ("fig04d_soil_CN",  bt.s_C_to_N,   "C:N",                    "rainbow",  0,   75),
    # ---- ADDITIONAL / SUPPLEMENTARY ----
    ("supp_soil_K",           bt.s_K,             r"K (mg/cm$^3$)",             "rainbow", 0,    0.2),
    ("supp_soil_Ca",          bt.s_Ca,            r"Ca (mg/cm$^3$)",            "rainbow", 0,    1.5),
    ("supp_soil_Mg",          bt.s_Mg,            r"Mg (mg/cm$^3$)",            "rainbow", 0,    0.1),
    ("supp_soil_Na",          bt.s_Na,            r"Na (mg/cm$^3$)",            "rainbow", 0,    0.012),
    ("supp_soil_Zn",          bt.s_Zn,            r"Zn (mg/cm$^3$)",            "rainbow", 0,    0.015),
    ("supp_soil_Mn",          bt.s_Mn,            r"Mn (mg/cm$^3$)",            "rainbow", 0,    0.1),
    ("supp_soil_Cu",          bt.s_Cu,            r"Cu (mg/cm$^3$)",            "rainbow", 0,    0.001),
    ("supp_soil_B",           bt.s_B,             r"B (mg/cm$^3$)",             "rainbow", 0.001, 0.002),
    ("supp_soil_NO3",         bt.s_NO3,           r"NO$_3$ (ppm)",              "rainbow", 0,   15),
    ("supp_soil_OM",          bt.s_OM_frac,       "Organic Matter (fraction)",  "rainbow", 0,    0.6),
    ("supp_soil_acidity",     bt.s_acidity,       "Acidity (meq/100g)",         "rainbow", 0,   12),
    ("supp_soil_total_C",     bt.s_total_C_fract, "Total Carbon (fraction)",    "rainbow", 0,    0.35),
    ("supp_soil_total_N",     bt.s_total_N_fract, "Total Nitrogen (fraction)",  "rainbow", 0,    0.02),
    ("supp_soil_total",       bt.s_total_frac,    "Total (fraction)",           "rainbow", 0,    0.5),
    ("supp_soil_log10vel",    np.log10(bt.s_vel), r"Log$_{10}$ Soil Velocity (m/s)", "rainbow", 2.6, 3.7),
]

print("\nGenerating soil chemistry figures …")
for fname, attr, lbl, cmap, vmin, vmax in attributes:
    _plot_attribute(
        attribute=attr,
        attribute_label=lbl,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        out_path=os.path.join(OUT_DIR, f"{fname}.png"),
    )

print("\nDone.  Paper panels: fig04a–fig04d_*.png")
print("Supplementary panels: supp_soil_*.png")
