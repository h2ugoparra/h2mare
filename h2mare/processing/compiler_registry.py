"""
Registry mapping var_key → compile processor for h2ds compilation.

Add a new entry to COMPILE_PROCESSORS when a variable needs custom handling
during the Zarr compilation step. Variables not registered here use
``compile_default``, which opens the catalog and regrids to the base grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.models import depth_levels_for, step_freq
from h2mare.storage.xarray_helpers import select_depth_levels
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateRange
from h2mare.utils.datetime_utils import end_of_day
from h2mare.utils.paths import store_root_for
from h2mare.utils.spatial import clip_land_data, regrid_to

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
    catalog: ZarrCatalog | None,
    var_key: str,
    date_range: DateRange,
    bbox,
    **kwargs,
) -> xr.Dataset | None:
    """Open a dataset from the catalog, returning None and logging on missing data."""
    from loguru import logger

    # CompileProcessor passes None for system variables (bathy, moon), but those
    # have their own processors and never reach here — a None catalog means a
    # non-system var_key was wired to one, which is a bug rather than missing data.
    if catalog is None:
        raise ValueError(
            f"_open_or_warn requires a catalog, got None for '{var_key}'. "
            f"System variables must not be dispatched to a catalog-reading processor."
        )

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
    # bathy is a plain NetCDF file rather than a catalogued store, so it cannot
    # go through _catalog_for and has to resolve its own root the same way.
    bathy_root = store_root_for(var_cfg, compiler.remote_store_root)
    data_path = bathy_root / var_cfg.local_folder / var_cfg.data_file
    ds = xr.open_dataset(data_path).sel(
        lon=slice(compiler.bbox.xmin, compiler.bbox.xmax),
        lat=slice(compiler.bbox.ymin, compiler.bbox.ymax),
    )
    return regrid_to(ds, compiler.base_grid)


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


def _compile_depth_var(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    Processor for stores with a depth axis: one column per configured level.

    Dispatched by config (any var_key with depth levels), not by name. Each
    listed variable becomes ``<variable>_<level>``; 2-D variables in the same
    store pass through.
    """
    assert catalog is not None
    var_key = catalog.var_key
    var_config = compiler.app_config.variables[var_key]
    levels = depth_levels_for(var_key, var_config)
    if not levels:
        # Not an assert: assertions are stripped under `python -O`, and without
        # this the depth axis would surface as an error somewhere unrelated. A
        # config error should name the config key.
        raise ValueError(
            f"'{var_key}' is compiled as a 3-D variable but declares no "
            f"depth_levels (or compile_depth_slices) in config.yaml. Add the "
            f"depth levels it should publish (e.g. {{thetao: [0, 100, 500]}})."
        )
    if step_freq(var_config) == "h":
        raise ValueError(
            f"'{var_key}' is hourly and has depth levels; compiling an hourly "
            f"3-D store is not supported."
        )

    ds = _open_or_warn(catalog, var_key, date_range, compiler.bbox, chunks={"depth": 1})
    if ds is None:
        return None
    return regrid_to(select_depth_levels(ds, levels, var_key), compiler.base_grid)


#: Decoded source bytes to aim for per slab of the hourly reduction.
#:
#: Reducing a whole year in one graph needs ~10.7 GB of hourly source before the
#: curl's stencil temporaries, which does not fit a 16 GB machine — it thrashed
#: at 0.3 cores and emitted no chunks. Slabbing caps the source in flight at
#: roughly this figure regardless of how long a range was asked for.
_SLAB_SOURCE_BUDGET_BYTES = 1024**3


# Slicing an hourly axis with a midnight-stamped end would drop that day's other
# 23 steps. ``ZarrCatalog`` resolves whole days on the way in, but a ``.sel``
# against an already-open hourly dataset has to do it itself.
_end_of_day = end_of_day


def _slab_days(ds: xr.Dataset) -> int:
    """Days per slab that keep the decoded source read near the budget.

    Derived from the data rather than from machine memory, so the choice is
    deterministic and adapts on its own as variables or resolution change.
    """
    cells = ds.sizes["lat"] * ds.sizes["lon"]
    per_step = sum(cells * ds[v].dtype.itemsize for v in ds.data_vars)

    # Distinct calendar days, not the first-to-last timedelta: a whole month of
    # hourly steps spans 30 days 23 h, which would report 24.8 steps per day.
    times = pd.DatetimeIndex(ds.time.values)
    span_days = max(len(times.normalize().unique()), 1)
    steps_per_day = max(ds.sizes["time"] / span_days, 1.0)

    per_day = per_step * steps_per_day
    return max(int(_SLAB_SOURCE_BUDGET_BYTES // per_day), 1)


def _slabs(date_range: DateRange, days: int) -> list[DateRange]:
    """Split a range into consecutive slabs of at most ``days`` days."""
    out: list[DateRange] = []
    start = pd.Timestamp(date_range.start)
    end = pd.Timestamp(date_range.end)

    # A slab can never usefully outrun the range. Without this a small enough
    # store — a narrow bbox, few variables — divides the budget into a day count
    # that overflows pd.Timedelta rather than simply yielding one slab.
    days = max(min(days, int((end - start).days) + 1), 1)

    while start <= end:
        stop = min(start + pd.Timedelta(days=days - 1), end)
        out.append(DateRange(start, stop))
        start = stop + pd.Timedelta(days=1)
    return out


def _covered_range(ds: xr.Dataset, date_range: DateRange) -> DateRange | None:
    """
    The part of *date_range* the opened store actually has days for.

    ``None`` when the two do not overlap at all, which is a variable whose
    store ends before the window opens rather than an error.
    """
    times = pd.DatetimeIndex(ds.time.values)
    if len(times) == 0:
        return None
    start = max(pd.Timestamp(date_range.start), times.min().normalize())
    end = min(pd.Timestamp(date_range.end), times.max().normalize())
    return DateRange(start, end) if start <= end else None


def _reduce_hourly_in_slabs(
    catalog: ZarrCatalog | None,
    var_key: str,
    bbox,
    date_range: DateRange,
    reduce_slab: Callable[[xr.Dataset, DateRange], xr.Dataset],
    warmup_days: int = 0,
) -> xr.Dataset | None:
    """
    Read an hourly store once, then reduce it to daily slab by slab.

    Peak memory follows ``_SLAB_SOURCE_BUDGET_BYTES`` rather than the length of
    the requested range. That keeps it to one write per output period: shrinking
    the *compile* window instead would rewrite the whole yearly h2ds file once
    per window.

    ``warmup_days`` widens the read backwards for reductions with rolling state;
    each slab widens by the same amount so it starts warm, and ``reduce_slab``
    is responsible for trimming the warm-up back off.
    """
    from loguru import logger

    warm = DateRange(
        date_range.start - pd.Timedelta(days=warmup_days),
        date_range.end,
    )
    ds = _open_or_warn(catalog, var_key, warm, bbox)
    if ds is None:
        return None

    # Slabs must tile what the store holds, not what was asked for. Each CDS
    # variable lags its provider by a different amount — accum and waves run a
    # month ahead of instante and radiation — so a compile window reaching
    # today outruns whichever store is furthest behind. A slab past the last
    # stamp selects nothing, and a reducer with no warm-up then resamples an
    # empty axis and raises, taking the whole compile with it. atm-accum-avg
    # only ever survived that by accident: its 20-day ekman warm-up widens the
    # read backwards, so the slice stayed non-empty where atm-instante's did not.
    covered = _covered_range(ds, date_range)
    if covered is None:
        logger.warning(
            f"[{var_key}] store holds nothing within {date_range} — skipping. "
            "Ordinary provider lag at the tail; the variable catches up on the "
            "next compile."
        )
        return None
    days = _slab_days(ds)
    slabs = _slabs(covered, days)

    # One line per variable, and it only reaches INFO when there is something to
    # notice: a store falling short of the window means that variable lags and
    # its tail lands in h2ds as NaN. A reduction that covers what was asked for
    # is routine, and at 29 chunks a run says so 29 times per variable.
    summary = (
        f"[{var_key}] hourly → daily over {covered} in "
        f"{len(slabs)} slab(s) of up to {days} day(s)"
    )
    if covered != date_range:
        logger.info(f"{summary} — short of the requested {date_range}")
    else:
        logger.debug(summary)

    # An interior hole can still leave a slab with nothing in it, which clipping
    # the ends cannot see. Skipping costs a coordinate read and keeps the same
    # empty-resample crash from coming back by another route.
    parts = [
        reduce_slab(ds, slab)
        for slab in slabs
        if ds.sel(time=slice(slab.start, _end_of_day(slab.end))).sizes.get("time", 0)
    ]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    # join="exact": every slab comes off the same store, so lat/lon must match
    # exactly. A silent outer join would NaN-fill a slab that somehow differed
    # instead of surfacing it.
    return xr.concat(parts, dim="time", join="exact")


def _daily_mean_for_slab(ds_hourly: xr.Dataset, slab: DateRange) -> xr.Dataset:
    """
    Reduce one slab of any hourly store to daily means, materialised.

    The generic counterpart to the per-variable reducers below: no warm-up, no
    derived fields, no unit conversion — every variable in the slab is averaged
    over its own day. Used by ``compile_default`` for hourly variables with no
    processor of their own, and by any registered processor whose reduction is
    the same plain mean.
    """
    from h2mare.processing.core.cds import resample_daily_mean

    sub = ds_hourly.sel(time=slice(slab.start, _end_of_day(slab.end)))
    return resample_daily_mean(sub).compute()


def _daily_features_for_slab(ds_hourly: xr.Dataset, slab: DateRange) -> xr.Dataset:
    """
    Reduce one slab of hourly source to its daily features, materialised.

    Computed eagerly rather than returned lazy: holding the whole range as one
    graph is exactly what does not fit, and the daily result is ~1/24 the size,
    so paying it slab by slab is what bounds peak memory.
    """
    from h2mare.processing.core.cds import (
        _EKMAN_WARMUP_DAYS,
        add_engineered_ekman,
        compute_curl_and_ekman,
        daily_total_rain,
    )

    warm_start = pd.Timestamp(slab.start) - pd.Timedelta(days=_EKMAN_WARMUP_DAYS)
    sub = ds_hourly.sel(time=slice(warm_start, _end_of_day(slab.end)))

    # Both of these already take hourly in and give daily out — they are the
    # same functions the daily convert path calls, just later.
    ds_ekman = compute_curl_and_ekman(sub)
    features = add_engineered_ekman(
        ds_ekman["ekman_pumping"], var_key="atm-accum-avg", seed_from_store=False
    )
    merged = xr.merge(
        [ds_ekman, features, daily_total_rain(sub)], compat="override", join="outer"
    )
    return merged.sel(time=slice(slab.start, slab.end)).compute()


def _atm_accum_from_hourly(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    Derive the daily ekman/rain features from an hourly atm-accum-avg store.

    Each slab is widened by ``_EKMAN_WARMUP_DAYS`` so its rolling features start
    warm, which replaces ``get_previous_dates_da``'s trip to the store: there is
    no stored daily ``ekman_pumping`` to seed from once the store is hourly, and
    reading real hours is strictly better than reading back a previous day's
    output. Warm-up days are trimmed off each slab before it is kept.
    """
    from h2mare.processing.core.cds import _EKMAN_WARMUP_DAYS

    return _reduce_hourly_in_slabs(
        catalog,
        "atm-accum-avg",
        compiler.var_config.bbox,
        date_range,
        # Indirect so monkeypatching the module attribute still takes effect.
        lambda ds, slab: _daily_features_for_slab(ds, slab),
        warmup_days=_EKMAN_WARMUP_DAYS,
    )


def _compile_atm_accum_avg(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    h2ds stays daily whatever cadence the atm-accum-avg store is kept at.

    A daily store already holds the derived features, so this only regrids. An
    hourly store holds raw stress and precipitation, so the whole ekman chain is
    computed here instead — yielding the same h2ds columns either way.
    """
    # .get, not [...]: step_freq already treats a missing entry as daily, which
    # keeps stand-in configs on the path every existing store actually uses.
    var_config = compiler.app_config.variables.get("atm-accum-avg")
    if step_freq(var_config) == "h":
        ds = _atm_accum_from_hourly(compiler, catalog, date_range)
    else:
        ds = _open_or_warn(
            catalog, "atm-accum-avg", date_range, compiler.var_config.bbox
        )
    if ds is None:
        return None
    # Coords ride along from the climatology alignment (.sel on dayofyear/month)
    # whether the features came off disk or were just computed.
    ds = ds.drop_vars(["dayofyear", "month", "quantile"], errors="ignore")
    return regrid_to(ds, compiler.base_grid)


def _daily_atm_instante_for_slab(ds_hourly: xr.Dataset, slab: DateRange) -> xr.Dataset:
    """
    Reduce one slab of hourly source to its daily wind/cloud/pressure fields.

    No warm-up: every one of these is a within-day aggregate, so a slab needs
    nothing from the days before it. Computed eagerly for the same reason as
    ``_daily_features_for_slab`` — the daily result is ~1/24 the size, so paying
    it slab by slab is what bounds peak memory.
    """
    from h2mare.processing.core.cds import (
        daily_cloud_cover,
        daily_sea_level_pressure,
        daily_wind,
    )

    sub = ds_hourly.sel(time=slice(slab.start, _end_of_day(slab.end)))
    # The same functions the daily convert path calls, just later.
    merged = xr.merge(
        [daily_wind(sub), daily_cloud_cover(sub), daily_sea_level_pressure(sub)],
        compat="override",
        join="outer",
    )
    return merged.compute()


def _compile_atm_instante(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    h2ds stays daily whatever cadence the atm-instante store is kept at.

    A daily store already holds the aggregates, so this only regrids. An hourly
    store holds raw wind, cloud and pressure, so the daily reductions run here
    instead — yielding the same h2ds columns either way.
    """
    # .get, not [...]: step_freq already treats a missing entry as daily, which
    # keeps stand-in configs on the path every existing store actually uses.
    var_config = compiler.app_config.variables.get("atm-instante")
    if step_freq(var_config) == "h":
        ds = _reduce_hourly_in_slabs(
            catalog,
            "atm-instante",
            compiler.var_config.bbox,
            date_range,
            # Indirect so monkeypatching the module attribute still takes effect.
            lambda hourly, slab: _daily_atm_instante_for_slab(hourly, slab),
        )
    else:
        ds = _open_or_warn(
            catalog, "atm-instante", date_range, compiler.var_config.bbox
        )
    if ds is None:
        return None
    return regrid_to(ds, compiler.base_grid)


#: radiation has no entry here on purpose. Both cadences settle their units at
#: convert time (J/m²→W/m²), so all compile owes is a daily mean for an hourly
#: store and nothing for a daily one — which is precisely ``compile_default``.
#: Its processor was a byte-identical copy of it and was removed rather than
#: kept in sync by hand.


def _compile_waves(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    h2ds stays daily whatever cadence the waves store is kept at.

    With ``time_step: hourly`` the convert step leaves the resample undone, so it
    happens here — through the same ``daily_waves`` the daily path has always
    used, so both cadences yield identical h2ds columns.
    """
    from h2mare.processing.core.cds import (
        daily_waves,
        direction_to_uv,
        uv_to_direction,
    )

    ds = _open_or_warn(catalog, "waves", date_range, compiler.var_config.bbox)
    if ds is None:
        return None
    if step_freq(compiler.app_config.variables["waves"]) == "h":
        ds = daily_waves(ds)

    # mdts is a direction in degrees, so it cannot be regridded by a weighted
    # average: interpolating between 350° and 10° sweeps the long way round and
    # lands on 180°. Interpolate the unit-vector components instead, then
    # recombine. swh is a magnitude and interpolates directly.
    components = direction_to_uv(ds["mdts"])
    to_interp = ds[["swh"]].assign(u_ts=components["u_ts"], v_ts=components["v_ts"])
    out = regrid_to(to_interp, compiler.base_grid)
    out["mdts"] = uv_to_direction(out["u_ts"], out["v_ts"])
    out["mdts"].attrs.update(ds["mdts"].attrs)
    return out.drop_vars(["u_ts", "v_ts"])


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
    return regrid_to(ds, compiler.base_grid)


# ---------------------------------------------------------------------------
# Default processor (open from catalog + interp to base grid)
# ---------------------------------------------------------------------------


def compile_default(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    Fallback processor: open from catalog, reduce an hourly store to daily, and
    regrid to the base grid.

    h2ds is daily whatever cadence its sources are kept at, and the compiler
    merges the per-variable results with an outer join. An hourly store handed
    over unreduced would therefore not fail — it would union its stamps into the
    axis, turning every day of h2ds into 24 rows with the daily variables null
    in 23 of them. Reducing here is what keeps ``time_step: hourly`` a property
    of the source store alone, and what lets an hourly variable be added by
    config without also writing a processor for it.

    The reduction is a daily mean, which is what an instantaneous field wants —
    temperature, currents, sea level, the hourly products CMEMS publishes. An
    accumulated variable (precipitation, radiation totals) needs its own
    ``COMPILE_PROCESSORS`` entry instead: a mean of hourly accumulations is a
    plausible-looking number rather than an error, so the reduction applied is
    logged rather than left to be inferred.
    """
    if catalog is None:
        return None

    var_key = catalog.var_key
    bbox = compiler.var_config.bbox

    # .get, not [...]: step_freq already treats a missing entry as daily, which
    # keeps stand-in configs on the path every existing store actually uses.
    if step_freq(compiler.app_config.variables.get(var_key)) == "h":
        from loguru import logger

        # Debug, not info: the line below already reports the reduction, and
        # which one it was matters when reading back a surprising number, not
        # once per chunk during a run.
        logger.debug(f"[{var_key}] unregistered hourly store — plain daily mean")
        ds = _reduce_hourly_in_slabs(
            catalog,
            var_key,
            bbox,
            date_range,
            # Indirect so monkeypatching the module attribute still takes effect.
            lambda hourly, slab: _daily_mean_for_slab(hourly, slab),
        )
    else:
        ds = _open_or_warn(catalog, var_key, date_range, bbox)

    if ds is None:
        return None
    if "depth" in ds.dims:
        # interp_like would carry the axis into h2ds, which is 2-D.
        raise ValueError(
            f"'{var_key}' has a depth axis but declares no depth_levels, so it "
            f"cannot be compiled. Add depth_levels (e.g. {{variable: [0, 100]}}) "
            f"to its config entry."
        )
    return regrid_to(ds, compiler.base_grid)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPILE_PROCESSORS: dict[str, CompileProcessor] = {
    "bathy": _compile_bathy,
    "moon": _compile_moon,
    "atm-accum-avg": _compile_atm_accum_avg,
    "atm-instante": _compile_atm_instante,
    "sst": _compile_sst,
    "waves": _compile_waves,
}
