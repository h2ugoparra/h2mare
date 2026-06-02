"""
Script to process bathymetry data from NOAA ETOPO 15s resolution mosaic data.
Mosaics are organized in 15x15 degree tiles.

Objectives:
    1) Create a merged layer at native 15 arc-second resolution for the North Atlantic domain.
    2) Create a mean and std bathymetry file at 0.25 degree resolution. Mean and std are
       computed from all 15s pixels within each 0.25 degree pixel.
"""

import warnings

import xarray as xr

from h2mare.config import get_settings
from h2mare.types import BBox
from h2mare.utils import GridBuilder, create_filename_label, resolve_store_path

warnings.filterwarnings("ignore")

DX, DY = 0.25, 0.25

var_key = "bathy"
var_cfg = get_settings().app_config.variables[var_key]
var_dir = resolve_store_path(var_cfg)

geo_extent = var_cfg.bbox
if geo_extent is None:
    raise ValueError("bathy config entry is missing required 'bbox' field")
xmin, ymin, xmax, ymax = geo_extent


# ----------------------
# ---- Merged layer ----
# ----------------------

surf15_dir = var_dir / "15s_resolution/surface"
files = list(surf15_dir.glob("ETOPO_2022_v1_15s_*_surface.nc"))

ds = (
    xr.open_mfdataset(files, combine="by_coords")
    .drop_vars(["crs"])
    .sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
)

file_name = (
    f"etopo_15s_{create_filename_label(BBox.from_tuple(geo_extent), 'year')}_surface.nc"
)
ds.to_netcdf(var_dir / file_name)
ds.close()


# --------------------------------
# ---- Mean and Std at 0.25° ----
# --------------------------------

da = xr.open_dataset(var_dir / file_name)["z"]

base_grid = GridBuilder(
    BBox.from_tuple((xmin, ymin, xmax, ymax)), DX, DY
).generate_grid()

coarsen_factor = int(round(DX / (da.lon.values[1] - da.lon.values[0])))

da_coarse = da.coarsen(
    lat=coarsen_factor,
    lon=coarsen_factor,
    boundary="pad",
    coord_func="mean",
)

da_mean = da_coarse.mean()  # type: ignore[union-attr]
da_std = da_coarse.std()  # type: ignore[union-attr]

ds_new = xr.Dataset({"bathy": da_mean, "bathy_std": da_std})

out_file_name = f"etopo_0.25deg_{create_filename_label(BBox.from_tuple(geo_extent), 'year')}_mean-std_surface.nc"
ds_new.to_netcdf(var_dir / out_file_name)
