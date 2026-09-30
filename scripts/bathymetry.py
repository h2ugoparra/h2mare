"""
Build the static bathymetry layers declared under ``layers`` in the bathy
config entry, from NOAA ETOPO 2022 v1 surface elevation.

Layers (file names come from config.yaml, never from here):
    15s, 60s — native-resolution layers over the bathy bbox, written as
        spatially-tiled Zarr so extraction reads only the tiles it touches.
        15s merges the 15x15 degree mosaic tiles; 60s subsets the global file.
        Each holds ``bathy`` plus the ``derived_vars`` of the bathy entry
        (``bathy_std``: 3x3 rolling std), and keeps the source file's global
        attributes and ``z`` attributes.
    0.25deg — the 15s layer coarsened to a 0.25 deg mean/std netCDF (mean and
        std over all 15s cells inside each 0.25 deg cell). Compile reads it.
        Reads the 15s layer from disk, so build 15s first.

Usage:
    uv run python scripts/bathymetry.py                       # every layer
    uv run python scripts/bathymetry.py --layers 60s          # some layers
"""

import argparse
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.models import KeyVarConfigEntry
from h2mare.processing.derived import apply_derived_vars
from h2mare.storage.xarray_helpers import apply_cf_attrs
from h2mare.types import BBox
from h2mare.utils import resolve_store_path, static_layer_path

# Narrow to RuntimeWarning (matches processing/compiler.py): silences the noisy
# all-NaN / empty-slice reduction warnings without hiding Deprecation/Future ones.
warnings.filterwarnings("ignore", category=RuntimeWarning)

DX = 0.25  # target coarse-grid cell size (degrees)

# Square spatial tile (cells) for the native-resolution Zarr stores. At 15
# arc-sec (~240 cells/deg) a 512-cell tile spans ~2.1 deg, at 60 arc-sec ~8.5
# deg, so a geometry reads only the few overlapping tiles.
TILE = 512

# Raw source files per native layer, relative to the bathy store directory.
NATIVE_SOURCES = {
    "15s": ("15s_resolution/surface", "ETOPO_2022_v1_15s_*_surface.nc"),
    "60s": ("60s_resolution", "ETOPO_2022_v1_60s_*_surface.nc"),
}
COARSE_LAYER = "0.25deg"
BUILD_ORDER = (*NATIVE_SOURCES, COARSE_LAYER)


class BathyConfig(NamedTuple):
    var_cfg: KeyVarConfigEntry
    var_dir: Path
    bbox: BBox

    def layer_path(self, layer: str) -> Path:
        return static_layer_path(self.var_cfg, layer, self.var_dir)


def _load_config() -> BathyConfig:
    """Resolve the bathy store directory and domain bbox from app config."""
    var_cfg = get_settings().app_config.variables["bathy"]
    if var_cfg.bbox is None:
        raise ValueError("bathy config entry is missing required 'bbox' field")
    return BathyConfig(
        var_cfg=var_cfg,
        var_dir=resolve_store_path(var_cfg),
        bbox=BBox.from_tuple(var_cfg.bbox),
    )


def merge_source_attrs(attr_dicts: list[dict]) -> dict:
    """
    One set of global attributes for a layer built from several source files.

    A key every file agrees on is kept as is. Where files differ (the 15s tiles
    each carry their own ``GDAL_TIFFTAG_DATETIME``, and a few coastal tiles say
    ``Bathymetry`` or ``Topography`` where the rest say
    ``Topography-Bathymetry``), the most common value wins, ties going to the
    greatest — the latest timestamp.
    """
    merged: dict = {}
    for key in dict.fromkeys(k for attrs in attr_dicts for k in attrs):
        counts = Counter(str(a[key]) for a in attr_dicts if key in a)
        winner = max(counts, key=lambda v: (counts[v], v))
        merged[key] = next(a[key] for a in attr_dicts if str(a.get(key)) == winner)
    return merged


def _source_files(cfg: BathyConfig, layer: str) -> list[Path]:
    subdir, pattern = NATIVE_SOURCES[layer]
    src_dir = cfg.var_dir / subdir
    files = sorted(src_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {layer} ETOPO files ({pattern}) under {src_dir}")
    return files


def build_native_layer(cfg: BathyConfig, layer: str) -> Path:
    """Write the *layer* ("15s" or "60s") Zarr with bathy + derived_vars and return its path."""
    files = _source_files(cfg, layer)
    logger.info(f"Building {layer} layer from {len(files)} file(s)")

    raw_attrs = merge_source_attrs([xr.open_dataset(f).attrs for f in files])

    # sortby before slicing: slice(min, max) only selects correctly on ascending
    # coords. ETOPO files are ascending today, but a descending batch would make
    # the slice silently return an empty array — sort defensively.
    src = xr.open_mfdataset(files, combine="by_coords")
    z_attrs = dict(src["z"].attrs)
    ds = (
        src.drop_vars(["crs"], errors="ignore")
        .sortby(["lat", "lon"])
        .sel(
            lon=slice(cfg.bbox.xmin, cfg.bbox.xmax),
            lat=slice(cfg.bbox.ymin, cfg.bbox.ymax),
        )
    )
    if ds.lat.size == 0 or ds.lon.size == 0:
        raise ValueError(
            f"bbox {cfg.bbox} selects no pixels from the {layer} files "
            f"(lat={ds.lat.size}, lon={ds.lon.size}) — check bbox vs file coverage"
        )
    logger.info(
        f"Selected grid: lat={ds.lat.size}, lon={ds.lon.size} | bbox={cfg.bbox}"
    )

    # Tiled before the rolling std so dask computes it tile by tile (with a
    # one-cell halo) rather than over one multi-degree block, and again after:
    # the rolling shifts its chunk edges by a cell, which Zarr refuses.
    tiles = {"lat": TILE, "lon": TILE}
    ds = ds.rename({"z": "bathy"}).chunk(tiles)
    # Stripped first: the rolling std would inherit them, and positive/height
    # describe an elevation, not its spread.
    ds["bathy"].attrs = {}
    ds = apply_derived_vars(ds, cfg.var_cfg.derived_vars, "bathy").chunk(tiles)

    # Source attrs first, config's CF attrs over them. grid_mapping named the
    # `crs` variable dropped above.
    z_attrs.pop("grid_mapping", None)
    ds["bathy"].attrs = z_attrs
    ds = apply_cf_attrs(ds, native_var_key="bathy")
    for var in ds.data_vars:
        ds[var].attrs["product_id"] = f"ETOPO 2022 v1 {layer}"

    # Last, since the rename/rolling above can drop dataset attrs.
    derived = ", ".join(
        f"{name} = {spec.op.value} window {spec.window}"
        for name, spec in (cfg.var_cfg.derived_vars or {}).items()
    )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    note = (
        f"{stamp}: h2mare scripts/bathymetry.py: {len(files)} ETOPO 2022 v1 {layer} "
        f"surface file(s) subset to bbox {cfg.bbox}, z renamed bathy"
        + (f"; {derived}" if derived else "")
    )
    ds.attrs = {
        **raw_attrs,
        "history": "\n".join(filter(None, [raw_attrs.get("history"), note])),
    }

    # Drop the inherited netCDF chunk encoding; it conflicts with the new dask
    # chunks on write ("would overlap multiple Dask chunks").
    for var in ds.variables:
        ds[var].encoding.pop("chunks", None)

    store_path = cfg.layer_path(layer)
    logger.info(f"Writing tiled Zarr (tile={TILE}) -> {store_path}")
    t0 = time.perf_counter()
    ds.to_zarr(store_path, mode="w")
    src.close()
    logger.success(f"{layer} layer written in {time.perf_counter() - t0:.1f}s")
    return store_path


def build_coarse_layer(cfg: BathyConfig) -> Path:
    """Coarsen the 15s layer to a 0.25 deg mean/std netCDF and return its path."""
    merged = cfg.layer_path("15s")
    da = xr.open_zarr(merged)["bathy"]

    # Guard the factor: DX must be a near-integer multiple of the native cell
    # size, otherwise round() would silently pick a wrong window size for a
    # surprise input resolution.
    native_res = float(da.lon.values[1] - da.lon.values[0])
    ratio = DX / native_res
    coarsen_factor = round(ratio)
    if abs(ratio - coarsen_factor) > 1e-3:
        raise ValueError(
            f"native resolution {native_res:.6f}° does not divide DX={DX}° to an "
            f"integer factor (ratio={ratio:.4f}); check the merged grid"
        )
    logger.info(f"Coarsening {merged.name} by factor {coarsen_factor} -> {DX}°")

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

    out_path = cfg.layer_path(COARSE_LAYER)
    ds_new.to_netcdf(out_path)
    logger.success(f"Coarse mean/std written -> {out_path}")
    return out_path


def main(layers: list[str]) -> None:
    cfg = _load_config()
    for layer in BUILD_ORDER:
        if layer not in layers:
            continue
        if layer == COARSE_LAYER:
            build_coarse_layer(cfg)
        else:
            build_native_layer(cfg, layer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the static bathy layers.")
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=BUILD_ORDER,
        default=list(BUILD_ORDER),
        help="Layers to build, in dependency order (default: all).",
    )
    main(parser.parse_args().layers)
