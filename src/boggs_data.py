"""
boggs_data.py
-------------
Core data-loading class (boggs_topo_struct) and associated helper functions
for the Boggs Rock study. All paths are resolved relative to the repository
root so the repo is fully self-contained.

Usage
-----
from src.boggs_data import boggs_topo_struct, DOMAIN

bt = boggs_topo_struct(DOMAIN)
"""

import os
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator, interp1d, griddata
from matplotlib.path import Path
import glob as glb

# ---------------------------------------------------------------------------
# Repository root: two levels up from this file (src/boggs_data.py)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _data(*parts):
    """Return an absolute path inside the repo data/ directory."""
    return os.path.join(REPO_ROOT, "data", *parts)


# ---------------------------------------------------------------------------
# Study domain (shared by all figure scripts)
# ---------------------------------------------------------------------------
DOMAIN = {
    "min x": 345010,
    "max x": 345130,
    "min y": 3852800,
    "max y": 3853050,
    "dx": 0.5,
    "dy": 0.5,
}


# ---------------------------------------------------------------------------
# Stand-alone maths helpers
# ---------------------------------------------------------------------------

def calc_mean_gaussian_curvature(f, h):
    """Return (KM, KG): mean and Gaussian curvature of scalar field f."""
    grad = np.gradient(f, h)
    fx = grad[1]
    fy = grad[0]
    fxx = np.gradient(fx, h)[1]
    fxy = np.gradient(fx, h)[0]
    fyy = np.gradient(fy, h)[0]
    KG = (fxx * fyy - fxy ** 2) / (1 + fx ** 2 + fy ** 2) ** 2
    KM = -((1 + fy ** 2) * fxx - 2 * fxy * fx * fy + (1 + fx ** 2) * fyy) / (
        2 * (fx ** 2 + fy ** 2 + 1) ** (3 / 2)
    )
    return KM, KG


def get_principal_curvatures(KG, KM):
    """Return (kmax, kmin) principal curvatures from Gaussian and mean."""
    kmax = KM + (KM ** 2 - KG) ** 0.5
    kmin = KM - (KM ** 2 - KG) ** 0.5
    return kmax, kmin


def calc_slope(D, pixel_size):
    """Return (slope, slope_deg) for elevation grid D."""
    grad = np.gradient(D, pixel_size)
    dx, dy = grad[1], grad[0]
    slope = (dx ** 2 + dy ** 2) ** 0.5
    slope_deg = np.arctan(slope) * 180 / np.pi
    return slope, slope_deg


# ---------------------------------------------------------------------------
# Main data class
# ---------------------------------------------------------------------------

class boggs_topo_struct:
    """Load and expose all datasets for the Boggs Rock study."""

    # Seismic glob patterns (relative to repo root)
    _SEIS_L1_GLOB = _data(
        "Seismic", "Boggs_L1", "STACKED", "Gimli_v2",
        "runs_multiThread", "*_gimliResults.vtk",
    )
    _SEIS_L2_GLOB = _data(
        "Seismic", "Boggs_L2", "STACKED", "Gimli_v2",
        "runs_multiThread", "*_gimliResults.vtk",
    )

    def __init__(self, domain,read_seismic=False):
        self.set_study_domain(domain)
        self._set_line_locs()
        self._set_naip_files()
        self._set_topo_files()
        self._build_point_clouds()
        self._get_planet_labs_ndvi()
        self.build_avg_ndvi(include_planet=True)
        self._read_soil_data()
        if read_seismic:
            self._read_boggs_seismic()

    # ------------------------------------------------------------------
    # Domain / grid
    # ------------------------------------------------------------------

    def set_study_domain(self, domain):
        self.study_area = domain
        self.study_east = np.arange(
            domain["min x"], domain["max x"], domain["dx"]
        )
        self.study_north = np.arange(
            domain["min y"], domain["max y"], domain["dy"]
        )
        self.EE, self.NN = np.meshgrid(self.study_east, self.study_north)

    # ------------------------------------------------------------------
    # Seismic line locations
    # ------------------------------------------------------------------

    def _set_line_locs(self):
        L1 = np.loadtxt(
            _data("Seismic", "Boggs_L1", "STACKED", "Gimli_v2", "Boggs_L1_liDAR.txt")
        )
        self.l1_e  = L1[:, 4]
        self.l1_n  = L1[:, 5]
        self.l1_ll = L1[:, 0]

        L2 = np.loadtxt(
            _data("Seismic", "Boggs_L2", "STACKED", "Gimli_v2", "Boggs_L2_LiDAR.txt")
        )
        self.l2_e  = L2[:, 4]
        self.l2_n  = L2[:, 5]
        self.l2_ll = L2[:, 0]

    # ------------------------------------------------------------------
    # LiDAR point clouds
    # ------------------------------------------------------------------

    def _build_point_clouds(self, frac2keep=1):
        pc1 = _data("Spatial", "LiDAR", "NOAA", "NOAA_PC.csv")
        pc2 = _data("Spatial", "LiDAR", "SC PickensCo 2011", "Veg", "SC2011.csv")
        pc3 = _data("Spatial", "LiDAR", "SC SavannahPeeDee 1 2019", "Veg", "SC2019.csv")

        df1, df2, df3 = pd.read_csv(pc1), pd.read_csv(pc2), pd.read_csv(pc3)
        x = np.concatenate([df1["X"], df2["X"], df3["X"]])
        y = np.concatenate([df1["Y"], df2["Y"], df3["Y"]])
        z = np.concatenate([df1["Z"], df2["Z"], df3["Z"]])

        idx = np.random.randint(0, len(z), int(np.floor(len(z) * frac2keep)))
        self.pc_x = x[idx]
        self.pc_y = y[idx]
        self.pc_z = z[idx]

    # ------------------------------------------------------------------
    # NAIP imagery
    # ------------------------------------------------------------------

    def _set_naip_files(self):
        base = _data("Spatial", "NAIP", "Time_Lapse_NAIP")
        files = [
            os.path.join(base, "m_3408211_sw_17_1_20110424",   "m_3408211_sw_17_1_20110424.tif"),
            os.path.join(base, "m_3408211_sw_17_1_20150421",   "m_3408211_sw_17_1_20150421.tif"),
            os.path.join(base, "m_3408211_sw_17_1_20170731",   "m_3408211_sw_17_1_20170731.tif"),
            os.path.join(base, "m_3408211_sw_17_060_20190906", "m_3408211_sw_17_060_20190906.tif"),
            os.path.join(base, "m_3408211_sw_17_060_20211017", "m_3408211_sw_17_060_20211017.tif"),
            os.path.join(base, "m_3408211_sw_17_060_20230411", "m_3408211_sw_17_060_20230411.tif"),
        ]
        dates = np.array([
            np.datetime64("2011-04-24"),
            np.datetime64("2015-04-21"),
            np.datetime64("2017-07-31"),
            np.datetime64("2019-09-06"),
            np.datetime64("2021-10-17"),
            np.datetime64("2023-04-11"),
        ])
        months = (dates.astype("datetime64[M]").view("int64") % 12) + 1
        month_names = pd.Index(dates).month_name()
        self.NAIP = {
            "files": files,
            "Dates": dates,
            "Month Numbers": months,
            "Month Names": month_names,
        }

    def read_naip_file(self, file_name):
        gtif = gdal.Open(file_name)
        bands = [gtif.GetRasterBand(i) for i in (1, 2, 3, 4)]
        R, G, B, NIR = [self._scale_to_unit(b.ReadAsArray(), 0, 255) for b in bands]

        x0, dx, _, y0, _, dy = gtif.GetGeoTransform()
        nrows, ncols = R.shape
        easting  = np.linspace(x0, x0 + dx * ncols, num=ncols)
        northing = np.linspace(y0, y0 + dy * nrows, num=nrows)

        def _interp(arr):
            u = RegularGridInterpolator(
                (northing, easting), arr,
                bounds_error=False, fill_value=0, method="linear"
            )
            pts = np.column_stack([self.NN.ravel(), self.EE.ravel()])
            return u(pts).reshape(self.EE.shape)

        R, G, B, NIR = _interp(R), _interp(G), _interp(B), _interp(NIR)
        NDVI = (NIR - R) / (NIR + R)
        return R, G, B, NIR, NDVI

    def build_avg_ndvi(self, include_planet=True):
        all_ndvi = [self.read_naip_file(f)[4] for f in self.NAIP["files"]]
        if include_planet:
            all_ndvi.append(self.planet_NDVI)
        self.all_NDVI   = np.array(all_ndvi)
        self.NDVI       = np.nanmean(all_ndvi, axis=0)
        self.NDVI_std   = np.nanstd(all_ndvi,  axis=0)

    # ------------------------------------------------------------------
    # Planet Labs (SkyWatch) imagery
    # ------------------------------------------------------------------

    def _get_planet_labs_ndvi(self):
        fname = _data("Spatial", "SkyWatch", "SKYWATCH_UTM.tif")
        gtif  = gdal.Open(fname)

        r_min, r_max = 200, 1500

        def _read(band_idx):
            return self._scale_to_unit(
                gtif.GetRasterBand(band_idx).ReadAsArray(), r_min, r_max
            )

        R, G, B, NIR = _read(4), _read(3), _read(2), _read(6)

        x0, dx, _, y0, _, dy = gtif.GetGeoTransform()
        nrows, ncols = R.shape
        easting  = np.linspace(x0, x0 + dx * ncols, num=ncols)
        northing = np.linspace(y0, y0 + dy * nrows, num=nrows)

        # Store full-res copies
        self.planet_R_fullRes       = R
        self.planet_G_fullRes       = G
        self.planet_B_fullRes       = B
        self.planet_NIR_fullRes     = NIR
        self.planet_NDVI_fullRes    = (NIR - R) / (NIR + R)
        self.planet_easting_fullRes  = easting
        self.planet_northing_fullRes = northing

        def _interp(arr):
            u = RegularGridInterpolator(
                (northing, easting), arr,
                bounds_error=False, fill_value=0, method="linear"
            )
            pts = np.column_stack([self.NN.ravel(), self.EE.ravel()])
            return u(pts).reshape(self.EE.shape)

        R, G, B, NIR = _interp(R), _interp(G), _interp(B), _interp(NIR)
        self.planet_R    = R
        self.planet_G    = G
        self.planet_B    = B
        self.planet_NIR  = NIR
        self.planet_NDVI = (NIR - R) / (NIR + R)

    # ------------------------------------------------------------------
    # Topography / LiDAR DEM
    # ------------------------------------------------------------------

    def _set_topo_files(self):
        base11 = _data("Spatial", "LiDAR", "SC PickensCo 2011")
        base19 = _data("Spatial", "LiDAR", "SC SavannahPeeDee 1 2019")

        e11, n11, elv_max11 = self.get_data_from_geotiff(
            os.path.join(base11, "Veg",    "output.localgridding", "output.max.tif"))
        _,   _,   elv_min11 = self.get_data_from_geotiff(
            os.path.join(base11, "Veg",    "output.localgridding", "output.min.tif"))
        _,   _,   elv11     = self.get_data_from_geotiff(
            os.path.join(base11, "Ground", "output.tin.tif"))
        EE11, NN11 = np.meshgrid(e11, n11)
        veg11 = elv_max11 - elv_min11

        e19, n19, elv_max19 = self.get_data_from_geotiff(
            os.path.join(base19, "Veg",    "output.localgridding", "output.max.tif"))
        _,   _,   elv_min19 = self.get_data_from_geotiff(
            os.path.join(base19, "Veg",    "output.localgridding", "output.min.tif"))
        _,   _,   elv19     = self.get_data_from_geotiff(
            os.path.join(base19, "Ground", "output.tin.tif"))
        EE19, NN19 = np.meshgrid(e19, n19)
        veg19 = elv_max19 - elv_min19

        all_veg = np.copy(veg11)
        all_veg[veg19 > veg11] = veg19[veg19 > veg11]
        all_veg[all_veg > 40]  = 0

        def _interp_to_grid(arr, e, n):
            u = RegularGridInterpolator(
                (n, e), arr, bounds_error=False, fill_value=0, method="linear"
            )
            pts = np.column_stack([self.NN.ravel(), self.EE.ravel()])
            return u(pts).reshape(self.EE.shape)

        self.TOPO = {
            "Veg Height":  _interp_to_grid(all_veg, e19, n19),
            "Elev 2011":   _interp_to_grid(elv11,   e19, n19),
            "Elev 2019":   _interp_to_grid(elv19,   e19, n19),
            "Veg Height full": all_veg,
            "Eastings full":   EE19,
            "Northings full":  NN19,
        }

        self.elv_2019 = self.TOPO["Elev 2019"]
        self.elv_2011 = self.TOPO["Elev 2011"]
        self.vegH     = self.TOPO["Veg Height"]

    # ------------------------------------------------------------------
    # Soil geochemistry
    # ------------------------------------------------------------------

    def _read_soil_data(self):
        df = pd.read_csv(_data("SoilSamples", "soil_sample_results.csv"))

        # Unit conversion: lbs/acre -> mg/cm^3 (assuming 6-inch sample depth)
        conv = 0.0007472
        self.s_pH           = df["Soil pH"].to_numpy().ravel()
        self.s_buf_pH       = df["Buffer pH"].to_numpy().ravel()
        self.s_P            = df["P (lbs/A)"].to_numpy().ravel()  * conv
        self.s_K            = df["K (lbs/A)"].to_numpy().ravel()  * conv
        self.s_Ca           = df["Ca (lbs/A)"].to_numpy().ravel() * conv
        self.s_Mg           = df["Mg (lbs/A)"].to_numpy().ravel() * conv
        self.s_Zn           = df["Zn (lbs/A)"].to_numpy().ravel() * conv
        self.s_Mn           = df["Mn (lbs/A)"].to_numpy().ravel() * conv
        self.s_Cu           = df["Cu (lbs/A)"].to_numpy().ravel() * conv
        self.s_B            = df["B (lbs/A)"].to_numpy().ravel()  * conv
        self.s_Na           = df["Na (lbs/A)"].to_numpy().ravel() * conv
        self.s_NO3          = df["NO3-N (ppm)"].to_numpy().ravel()
        self.s_OM_frac      = df["OM (%)"].to_numpy().ravel()  / 100.0
        self.s_CEC          = df["CEC (meg/100g)"].to_numpy().ravel()
        self.s_acidity      = df["Acidity (meg/100g)"].to_numpy().ravel()
        self.s_total_C_fract = df["C (%)"].to_numpy().ravel() / 100.0
        self.s_total_N_fract = df["N (%)"].to_numpy().ravel() / 100.0
        self.s_C_to_N       = self.s_total_C_fract / self.s_total_N_fract
        self.s_total_frac   = df["Total  (%)"].to_numpy().ravel() / 100.0
        self.s_number       = df["Sample"].to_numpy().ravel()
        self.s_distance     = df["Distance Along L1 (m)"].to_numpy().ravel()
        self.s_maxDepth     = df["Max Depth (m)"].to_numpy().ravel().copy()
        self.s_maxDepth[self.s_maxDepth == -999] = 0.25

        # Manual position adjustment (see field notes)
        self.s_distance[-4] = 168

        # Map sample distances onto line coordinates
        self.s_easting  = np.zeros(len(self.s_distance))
        self.s_northing = np.zeros(len(self.s_distance))
        for i, d in enumerate(self.s_distance):
            idx = np.argmin((d - self.l1_ll) ** 2)
            self.s_easting[i]  = self.l1_e[idx]
            self.s_northing[i] = self.l1_n[idx]

        self.soil_ndvi     = self.extract_raster_values(
            self.s_easting, self.s_northing, self.EE, self.NN, self.NDVI)
        self.soil_ndvi_std = self.extract_raster_values(
            self.s_easting, self.s_northing, self.EE, self.NN, self.NDVI_std)
        self.soil_vegH     = self.extract_raster_values(
            self.s_easting, self.s_northing, self.EE, self.NN, self.vegH)
        self.soil_elv      = self.extract_raster_values(
            self.s_easting, self.s_northing, self.EE, self.NN, self.elv_2019)

        # Assign ecological groups (0 = thin soil/lichen, 1 = outcrop shrub, 2 = oak/hickory)
        self.s_groups = np.zeros(len(self.s_distance))
        self.s_groups[8:15]  = 1
        self.s_groups[17:28] = 1
        self.s_groups[31:]   = 2
        self.s_groups[15:17] = 2

    # ------------------------------------------------------------------
    # Seismic
    # ------------------------------------------------------------------

    def _read_boggs_seismic(self):
        for attr_prefix, glob_pat in [("L1", self._SEIS_L1_GLOB),
                                       ("L2", self._SEIS_L2_GLOB)]:
            cc, cv, nodes, vel, std, rc, models = self._get_avg_vel_profile(glob_pat)
            setattr(self, f"seis_{attr_prefix}_cc",     cc)
            setattr(self, f"seis_{attr_prefix}_cv",     cv)
            setattr(self, f"seis_{attr_prefix}_nodes",  nodes)
            setattr(self, f"seis_{attr_prefix}_vel",    vel)
            setattr(self, f"seis_{attr_prefix}_std",    std)
            setattr(self, f"seis_{attr_prefix}_rayCov", rc)
            setattr(self, f"seis_{attr_prefix}_models", models)

    def _get_avg_vel_profile(self, glob_pattern):
        files = glb.glob(glob_pattern)
        if not files:
            raise FileNotFoundError(f"No VTK files matched: {glob_pattern}")

        cc, cv, nodes, vel = self._read_vtk_file(files[0], "Velocity")
        _,  _,  _,     rc  = self._read_vtk_file(files[0], "Standard_Ray_Coverage")

        vel_models = np.zeros((len(vel), len(files)))
        rc_all     = np.zeros_like(vel_models)
        vel_models[:, 0] = vel
        rc_all[:, 0]     = rc

        for i, f in enumerate(files[1:], start=1):
            _, _, _, v = self._read_vtk_file(f, "Velocity")
            _, _, _, r = self._read_vtk_file(f, "Standard_Ray_Coverage")
            vel_models[:, i] = v
            rc_all[:, i]     = r
            #print(f,np.max(v),np.min(v))
            print(f"  Loaded seismic model {i}/{len(files)-1}")

        mean_vel = np.mean(vel_models, axis=1)
        std_vel  = np.std(vel_models,  axis=1)
        mean_rc  = np.sum(rc_all, axis=1) / rc_all.shape[1]
        return cc, cv, nodes, mean_vel, std_vel, mean_rc, [vel_models, rc_all]

    # def _read_vtk_file(self, vtk_file, property_name):
    #     """Parse a simple ASCII VTK triangular mesh file."""
    #     with open(vtk_file, "r") as fh:
    #         content = fh.read().split("\n")

    #     nhdr   = 4
    #     n_nodes = int(content[nhdr].split()[1])
    #     node_pos = np.array(
    #         [content[nhdr + 1 + i].split()[:2] for i in range(n_nodes)],
    #         dtype=float,
    #     )

    #     n_cells = int(content[nhdr + n_nodes + 1].split()[1])
    #     start   = nhdr + n_nodes + 2
    #     cell_verts   = np.zeros((n_cells, 3), dtype=int)
    #     cell_centers = np.zeros((n_cells, 2))
    #     for i in range(n_cells):
    #         row = content[start + i].split()
    #         v   = int(row[1]), int(row[2]), int(row[3])
    #         cell_verts[i]   = v
    #         cell_centers[i] = node_pos[list(v), :].mean(axis=0)

    #     prop_vals = None
    #     for lnum, line in enumerate(content):
    #         if (line.startswith("SCALARS") and
    #                 property_name in line):
    #             vals = content[lnum + 2].split()
    #             prop_vals = np.array(vals, dtype=float)
    #             break
    #     if prop_vals is None:
    #         raise ValueError(f"Property '{property_name}' not found in {vtk_file}")

    #     return cell_centers, cell_verts, node_pos, prop_vals
    def _read_vtk_file(self,vtkFile,propertyName):
        """
        the vtk file and the 1D data file have to have the same number. 
        """
        nhdrLines = 4
        f= open(vtkFile,'r')
        content = f.read().split('\n')
        
        #Extract the nodes and calculate midpoints
        nNodes = content[nhdrLines].split(' ')
        nNodes = int(nNodes[1])
        
        nodePositions = np.zeros((nNodes,2),dtype=float)
        index = 0
        for i in range(nhdrLines+1,nhdrLines+nNodes+1):
            posArr = content[i].split('\t')
            posArr = np.array(posArr)    
            nodePositions[index,0] = posArr[0]
            nodePositions[index,1] = posArr[1]
            index = index + 1
        
        nCells = int(content[nhdrLines+nNodes+1].split(' ')[1])
        startInd = nhdrLines+nNodes+2
        endInd = startInd + nCells
        cellCenters = np.zeros((nCells,2),dtype=float)
        cellVerts = np.zeros((nCells,3),dtype=float)
        index = 0
        for i in range(startInd,endInd):
            cell1 = int(content[i].split('\t')[1])
            cell2 = int(content[i].split('\t')[2])
            cell3 = int(content[i].split('\t')[3])
            cellVerts[index,0] = cell1
            cellVerts[index,1] = cell2
            cellVerts[index,2] = cell3
            #print([cell1,cell2,cell3])
            xc = (nodePositions[cell1,0] + nodePositions[cell2,0] + nodePositions[cell3,0])/3
            yc = (nodePositions[cell1,1] + nodePositions[cell2,1] + nodePositions[cell3,1])/3
            cellCenters[index,0] = xc
            cellCenters[index,1] = yc
            index = index + 1
         
        lineNum = 0
        for line in content:
            if line[0:7] == "SCALARS" and line[8:9+len(propertyName)-1]==propertyName:
                #print(lineNum)
                propVals = content[lineNum+2].split(' ')
                propVals = propVals[0:len(propVals)-1]
                propVals = np.asarray(propVals).astype('float')
            lineNum = lineNum + 1
        
        return cellCenters, cellVerts, nodePositions, propVals
    

    # ------------------------------------------------------------------
    # Spatial utility methods
    # ------------------------------------------------------------------

    def get_data_from_geotiff(self, file_name):
        gtif  = gdal.Open(file_name)
        elev  = gtif.GetRasterBand(1).ReadAsArray()
        nrows, ncols = elev.shape
        x0, dx, _, y0, _, dy = gtif.GetGeoTransform()
        easting  = np.linspace(x0, x0 + dx * ncols, num=ncols)
        northing = np.linspace(y0, y0 + dy * nrows, num=nrows)
        return easting, northing, elev

    def extract_raster_values(self, x, y, EE, NN, ZZ):
        u = RegularGridInterpolator(
            (NN[:, 0], EE[0, :]), ZZ,
            bounds_error=False, fill_value=0, method="linear",
        )
        return u(np.column_stack([y, x]))

    def _scale_to_unit(self, x, vmin, vmax):
        x = np.asarray(x, dtype=np.float32)
        return np.clip((x - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Profile / point-cloud extraction along seismic lines
    # ------------------------------------------------------------------

    def calc_unit_vector(self, SoL, EoL):
        dx, dy = EoL[0] - SoL[0], EoL[1] - SoL[1]
        mag = np.sqrt(dx**2 + dy**2)
        return dx / mag, dy / mag, mag

    def calculate_heading(self, easting, northing):
        heading = np.zeros((len(easting), 2))
        heading[0, 0], heading[0, 1], _ = self.calc_unit_vector(
            [easting[0], northing[0]], [easting[1], northing[1]])
        for i in range(1, len(easting)):
            heading[i, 0], heading[i, 1], _ = self.calc_unit_vector(
                [easting[i-1], northing[i-1]], [easting[i], northing[i]])
        return heading

    def calculate_smoothed_linear(self, x, y, npts):
        x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
        t_orig = np.linspace(0.0, 1.0, x.size)
        t_new  = np.linspace(0.0, 1.0, npts)
        xx = interp1d(t_orig, x, kind="linear", fill_value="extrapolate")(t_new)
        yy = interp1d(t_orig, y, kind="linear", fill_value="extrapolate")(t_new)
        heading = self.calculate_heading(xx, yy)
        ll = np.insert(np.cumsum(np.sqrt(np.diff(xx)**2 + np.diff(yy)**2)), 0, 0)
        return xx, yy, heading[:, 0], heading[:, 1], ll

    def build_polygon(self, x, y, h_ux, h_uy, width):
        line_x = np.zeros(len(x))
        line_y = np.zeros(len(y))
        for k in range(len(x)):
            line_x[k] = -h_uy[k] * width + x[k]
            line_y[k] =  h_ux[k] * width + y[k]
        for k in range(len(x) - 1, -1, -1):
            line_x = np.append(line_x,  h_uy[k] * width + x[k])
            line_y = np.append(line_y, -h_ux[k] * width + y[k])
        line_x = np.append(line_x, line_x[0])
        line_y = np.append(line_y, line_y[0])
        path = Path(np.column_stack([line_x, line_y]))
        return line_x, line_y, path

    def get_points_in_poly(self, x, y, z, poly_path):
        mask = poly_path.contains_points(np.column_stack([x, y]))
        return x[mask], y[mask], z[mask]

    def get_profile_point_cloud(self, x_in, y_in, xx, yy, ll):
        closest = np.zeros(len(x_in))
        for i in range(len(x_in)):
            idx = np.argmin((x_in[i] - xx)**2 + (y_in[i] - yy)**2)
            closest[i] = ll[idx]
        return closest

    def extract_pc_along_path(self, x, y, l, N=1000, width=3):
        xx, yy, h_ux, h_uy, ll = self.calculate_smoothed_linear(x, y, N)
        _, _, poly_path = self.build_polygon(xx, yy, h_ux, h_uy, width)
        x_in, y_in, z_in = self.get_points_in_poly(
            self.pc_x, self.pc_y, self.pc_z, poly_path)
        closest_dist = self.get_profile_point_cloud(x_in, y_in, xx, yy, ll)
        surf_elv  = self.extract_raster_values(x, y, self.EE, self.NN, self.elv_2019)
        surf_elev = np.interp(closest_dist, l, surf_elv)
        veg_h     = z_in - surf_elev
        return closest_dist, z_in, veg_h, surf_elv
