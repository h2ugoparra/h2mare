"""
Registry mapping var_key → compile processor for h2ds compilation.

Add a new entry to COMPILE_PROCESSORS when a variable needs custom handling
during the Zarr compilation step. Variables not registered here use
``compile_default``, which opens the catalog and interpolates to the base grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateRange
from h2mare.utils.spatial import clip_land_data

if TYPE_CHECKING:
    from h2mare.processing.compiler import Compiler

# Each processor receives the Compiler instance (for grid/bbox/config access),
# the variable's ZarrCatalog (None for system variables like bathy and moon),
# and the compilation DateRange.
CompileProcessor = Callable[
    ["Compiler", Optional[ZarrCatalog], DateRange],
    Optional[xr.Dataset],
]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _open_or_warn(
    catalog: ZarrCatalog,
    var_key: str,
    date_range: DateRange,
    bbox,
    **kwargs,
) -> xr.Dataset | None:
    """Open a dataset from the catalog, returning None and logging on missing data."""
    from loguru import logger

    try:
        return catalog.open_dataset(
            start_date=date_range.start,
            end_date=date_range.end,
            bbox=bbox,
            **kwargs,
        )
    except FileNotFoundError:
        logger.warning(
            f"No data for {var_key} during "
            f"{date_range.start.date()}–{date_range.end.date()} — skipping."
        )
        return None


# ---------------------------------------------------------------------------
# Registered processors (one per special-cased variable)
# ---------------------------------------------------------------------------


def _compile_bathy(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    var_cfg = compiler.app_config.variables["bathy"]
    if var_cfg.data_file is None:
        raise ValueError("bathy config entry is missing required 'data_file' field")
    data_path = compiler.remote_store_root / var_cfg.local_folder / var_cfg.data_file
    ds = xr.open_dataset(data_path).sel(
        lon=slice(compiler.bbox.xmin, compiler.bbox.xmax),
        lat=slice(compiler.bbox.ymin, compiler.bbox.ymax),
    )
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


def _compile_moon(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset:
    # Lazy import breaks the compiler.py ↔ compiler_registry.py cycle.
    from h2mare.processing.compiler import calculate_moon_phase

    dates = pd.date_range(date_range.start, date_range.end, freq="D")
    lat = float(compiler.base_grid.lat.mean().values)
    lon = float(compiler.base_grid.lon.mean().values)
    moon_phase = calculate_moon_phase(lat, lon, dates)
    da = xr.DataArray(
        np.broadcast_to(
            np.array(moon_phase)[:, None, None],
            (len(dates), len(compiler.base_grid.lat), len(compiler.base_grid.lon)),
        ),
        name="moon_phase",
        dims=["time", "lat", "lon"],
        coords={
            "time": dates,
            "lat": compiler.base_grid.lat,
            "lon": compiler.base_grid.lon,
        },
    )
    return clip_land_data(da.to_dataset())


def _compile_o2(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    _depths = [0, 100, 500, 1000]
    ds = _open_or_warn(catalog, "o2", date_range, compiler.bbox)
    if ds is None:
        return None
    ds_depths = ds.sel(depth=_depths, method="nearest")
    ds_interp = ds_depths.interp_like(
        compiler.base_grid, method="linear", assume_sorted=True
    )
    return xr.Dataset(
        {
            f"o2_{target}": ds_interp.o2.isel(depth=i).drop_vars("depth")
            for i, target in enumerate(_depths)
        }
    )


def _compile_thetao(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    _depths = [100, 200, 500, 1000]
    ds = _open_or_warn(
        catalog, "thetao", date_range, compiler.bbox, chunks={"depth": 1}
    )
    if ds is None:
        return None
    ds_depths = ds.sel(depth=_depths, method="nearest")
    ds_interp = ds_depths.interp_like(
        compiler.base_grid, method="linear", assume_sorted=True
    )
    return xr.Dataset(
        {
            f"thetao_{target}": ds_interp.thetao.isel(depth=i).drop_vars("depth")
            for i, target in enumerate(_depths)
        }
    )


def _compile_atm_accum_avg(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    ds = _open_or_warn(catalog, "atm-accum-avg", date_range, compiler.var_config.bbox)
    if ds is None:
        return None
    ds = ds.drop_vars(["dayofyear", "month", "quantile"])
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


def _compile_sst(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    from h2mare.processing.compiler import postprocess_sst_fdist

    ds = _open_or_warn(catalog, "sst", date_range, compiler.var_config.bbox)
    if ds is None:
        return None
    ds = postprocess_sst_fdist(ds)
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


# ---------------------------------------------------------------------------
# Default processor (open from catalog + interp to base grid)
# ---------------------------------------------------------------------------


def compile_default(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """Fallback processor: open from catalog and interpolate to the base grid."""
    from loguru import logger

    try:
        ds = catalog.open_dataset(
            start_date=date_range.start,
            end_date=date_range.end,
            bbox=compiler.var_config.bbox,
        )
    except FileNotFoundError:
        logger.warning(
            f"No data during "
            f"{date_range.start.date()}–{date_range.end.date()} — skipping."
        )
        return None
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPILE_PROCESSORS: dict[str, CompileProcessor] = {
    "bathy": _compile_bathy,
    "moon": _compile_moon,
    "o2": _compile_o2,
    "thetao": _compile_thetao,
    "atm-accum-avg": _compile_atm_accum_avg,
    "sst": _compile_sst,
}
