# Boggs Rock — Reproducibility Repository

Code and data for:

> **[Paper title]**
> [Authors]
> *[Journal]*, [Year]
> DOI: [doi]

---

## Repository layout

```
Boggs_data_repo/
│
├── README.md               ← you are here
├── environment.yml         ← conda environment (recommended)
├── requirements.txt        ← pip alternative (GDAL must be pre-installed)
│
├── src/
│   ├── boggs_data.py       ← core data-loading class & spatial helpers
│   └── plot_utils.py       ← shared colour maps, label helpers
│
├── figures/
│   ├── fig01_map.py                ← Figure 1  (study-area map)
│   ├── fig02_seismic_profiles.py   ← Figure 2  (seismic cross-sections)
│   ├── fig03_ndvi_canopy.py        ← Figure 3  (NDVI & canopy height)
│   ├── fig04_soil_chemistry.py     ← Figure 4  (soil geochemistry panels)
│   └── figS1_ndvi_timeseries.py    ← Supplemental S1 (NDVI time series)
│
├── data/                   ← all input datasets (see Data section below)
│   ├── Seismic/
│   │   ├── Boggs_L1/
│   │   └── Boggs_L2/
│   ├── SoilSamples/
│   ├── Spatial/
│   │   ├── LiDAR/
│   │   ├── NAIP/
│   │   └── SkyWatch/
│   └── FieldPhotos/
│
└── outputs/                ← generated figures land here (created on run)
```

---

## Quickstart

### 1 — Clone the repository

```bash
git clone https://github.com/[username]/Boggs_data_repo.git
cd Boggs_data_repo
```

### 2 — Create the Python environment

**Recommended (conda):**

```bash
conda env create -f environment.yml
conda activate boggs
```

**Alternative (pip) — requires GDAL to be installed separately:**

```bash
pip install -r requirements.txt
```

GDAL installation varies by platform; the simplest cross-platform route is
via conda-forge as above.

### 3 — Run a figure script

All scripts are run from the **repository root** so that relative paths
resolve correctly:

```bash
python figures/fig01_map.py
python figures/fig02_seismic_profiles.py
python figures/fig03_ndvi_canopy.py
python figures/fig04_soil_chemistry.py
python figures/figS1_ndvi_timeseries.py
```

Outputs are written to `outputs/`.  The directory is created automatically
if it does not exist.

> **Note:** `fig03_ndvi_canopy.py` and `fig04_soil_chemistry.py` both
> compute a KDE of NDVI vs canopy height over the full study grid, which
> takes ~1–2 minutes the first time.

---

## Data

| Dataset | Source | Location in repo |
|---------|--------|-----------------|
| Seismic P-wave models (bootstrap ensemble) | This study (GIMli inversion) | `data/Seismic/Boggs_L{1,2}/STACKED/Gimli_v2/runs_multiThread/` |
| LiDAR point clouds — NOAA | [NOAA Digital Coast](https://coast.noaa.gov/digitalcoast/) | `data/Spatial/LiDAR/NOAA/` |
| LiDAR point clouds — SC 2011 | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | `data/Spatial/LiDAR/SC PickensCo 2011/` |
| LiDAR point clouds — SC 2019 | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | `data/Spatial/LiDAR/SC SavannahPeeDee 1 2019/` |
| NAIP imagery (2011–2023) | [USDA NAIP](https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/) | `data/Spatial/NAIP/Time_Lapse_NAIP/` |
| Planet Labs imagery (2024) | SkyWatch / Planet | `data/Spatial/SkyWatch/SKYWATCH_UTM.tif` |
| Soil geochemistry | This study (Waypoint Analytical) | `data/SoilSamples/soil_sample_results.csv` |
| Field photographs | This study | `data/FieldPhotos/` |

---

## Figure index

| Script | Output file(s) | Paper figure |
|--------|---------------|-------------|
| `fig01_map.py` | `fig01_map.png` | Figure 1 |
| `fig02_seismic_profiles.py` | `fig02_seismic_profiles.png` | Figure 2 |
| `fig03_ndvi_canopy.py` | `fig03_ndvi_canopy.png` | Figure 3 |
| `fig04_soil_chemistry.py` | `fig04a_soil_pH.png`, `fig04b_soil_CEC.png`, `fig04c_soil_P.png`, `fig04d_soil_CN.png` (+ supplementary panels) | Figure 4 (assembled in Illustrator) |
| `figS1_ndvi_timeseries.py` | `figS1a_ndvi_L1_profile.png`, `figS1b_ndvi_maps.png` | Supplemental S1 |

---

## Software versions

Developed and tested with:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| NumPy | 1.26 |
| Pandas | 2.1 |
| SciPy | 1.11 |
| Matplotlib | 3.8 |
| GDAL | 3.7 |

---

## Citation

If you use this code or data please cite:

```
[BibTeX entry]
```

---

## Contact

[Name] — [email]
University of Newcastle, Australia
# Boggs_Rock_Seismic_Veg
