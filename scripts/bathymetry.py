"""
Script to process bathymetry data from NOAA ETOPO 15s resolution mosaic data.
Mosaics are organized in 15x15 degree tiles.

Two independent build stages (run both by default, or import one):
    1) build_merged_layer  — merge the 15s tiles into a native-resolution Zarr
       store, tiled spatially so geometry extraction reads only overlapping tiles.
    2) build_coarse_layer  — coarsen that merged layer to a 0.25° mean/std netCDF
       (mean and std over all 15s pixels within each 0.25° cell).

Stage 2 reads stage 1's output from disk, so either stage can be re-run on its
own (e.g. re-tile without recomputing the 0.25° layer, or vice versa).
"""

import warnings
from pathlib import Path
from typing import NamedTuple

import xarray as xr

from h2mare.config import get_settings
from h2mare.types import BBox
from h2mare.utils import create_filename_label, resolve_store_path

warnings.filterwarnings("ignore")

DX = 0.25  # target coarse-grid cell size (degrees)

# Square spatial tile (cells) for the native-resolution Zarr store. At 15 arc-sec
# (~240 cells/°) a 512-cell tile spans ~2.1°, so a geometry reads only the few
# overlapping tiles instead of a multi-degree block. See Extractor._extract_bathy.
TILE = 512


class BathyConfig(NamedTuple):
    var_dir: Path
    bbox: BBox


def _load_config() -> BathyConfig:
    """Resolve the bathy store directory and domain bbox from app config."""
    var_cfg = get_settings().app_config.variables["bathy"]
    if var_cfg.bbox is None:
        raise ValueError("bathy config entry is missing required 'bbox' field")
    return BathyConfig(
        var_dir=resolve_store_path(var_cfg),
        bbox=BBox.from_tuple(var_cfg.bbox),
    )


def build_merged_layer(cfg: BathyConfig) -> Path:
    """Merge the 15s tiles into a spatially-tiled Zarr store and return its path."""
    surf15_dir = cfg.var_dir / "15s_resolution/surface"
    files = list(surf15_dir.glob("ETOPO_2022_v1_15s_*_surface.nc"))

    ds = (
        xr.open_mfdataset(files, combine="by_coords")
        .drop_vars(["crs"])
        .sel(
            lon=slice(cfg.bbox.xmin, cfg.bbox.xmax),
            lat=slice(cfg.bbox.ymin, cfg.bbox.ymax),
        )
    )

    store_path = (
        cfg.var_dir
        / f"etopo_15s_{create_filename_label(cfg.bbox, 'year')}_surface.zarr"
    )
    # Tiling makes geometry extraction cheap: a small geometry reads only the
    # overlapping tiles instead of decompressing a multi-degree chunk.
    ds = ds.chunk({"lat": TILE, "lon": TILE})
    # Drop the inherited netCDF chunk encoding; it conflicts with the new dask
    # chunks on write ("would overlap multiple Dask chunks").
    for var in ds.variables:
        ds[var].encoding.pop("chunks", None)
    ds.to_zarr(store_path, mode="w")
    ds.close()
    return store_path


def build_coarse_layer(cfg: BathyConfig, merged: Path) -> Path:
    """Coarsen the merged 15s layer to a 0.25° mean/std netCDF and return its path."""
    da = xr.open_zarr(merged)["z"]

    coarsen_factor = int(round(DX / (da.lon.values[1] - da.lon.values[0])))
    da_coarse = da.coarsen(
        lat=coarsen_factor,
        lon=coarsen_factor,
        boundary="pad",
        coord_func="mean",
    )

    ds_new = xr.Dataset(
        {
            "bathy": da_coarse.mean(),  # type: ignore[union-attr]
            "bathy_std": da_coarse.std(),  # type: ignore[union-attr]
        }
    )

    out_path = (
        cfg.var_dir
        / f"etopo_0.25deg_{create_filename_label(cfg.bbox, 'year')}_mean-std_surface.nc"
    )
    ds_new.to_netcdf(out_path)
    return out_path


if __name__ == "__main__":
    cfg = _load_config()
    merged = build_merged_layer(cfg)
    build_coarse_layer(cfg, merged)
