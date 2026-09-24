"""
Recompute a variable's ``boa_fronts`` layers from its own Zarr store.

Front distances are produced at convert time, from the raw NetCDF files — which
are deleted after each conversion for every variable that does not set
``archive_raw``. So the CLI route to a changed threshold (``h2mare convert``)
means re-downloading the whole record, ~29 years per variable for sst and chl.
The distances do not need the raw files, though: they are computed from the
field itself, and that field is in the store. This reads it back, re-detects,
and writes the layers over the ones already there.

Only the front layers are written. The store's merge semantics keep every
variable the incoming dataset does not carry (``storage.py::_append_data``), so
``sst`` and ``analysis_error`` survive a ``sst_fdist`` rewrite untouched — and
a store that has no layer yet gets one, through the same call.

Written for step 3 of ``plans/eddy-distance-metric.md``: every stored front
distance predates the 2026-09 fix to ``haversine_min_distance_kdtree``, which
measured on a lat/lon plane and so overstated any east-west separation by
``1/cos(lat)``. The fronts themselves are unaffected — detection never leaves
index space — so what a recompute changes is the distance to them.

A dry run detects a few days per file and compares them against what is stored,
which is the cheap way to see what a threshold change would do (or what the
metric fix does) before committing to hours of compute:

    uv run python scripts/recompute_fronts.py sst                    # sample + compare
    uv run python scripts/recompute_fronts.py sst --years 2021 2022
    uv run python scripts/recompute_fronts.py sst --years 2021 --apply
    uv run python scripts/recompute_fronts.py --all --apply          # every var_key with fronts

Cost, per store file: detection is one BOA pass and one KD-tree query per day
across ``n_workers`` processes, and the file is rewritten wholesale through a
tmp-and-swap. Budget roughly twice the file's size free beside the store, and
about as much again under ``INTERIM_DIR`` for the staging the detection writes.
An interrupted ``--apply`` leaves the store recoverable: the swap is atomic and
``recover_zarr_store`` restores a stranded backup on the next run.

Recomputing a native store is not the end of it — h2ds and the Parquet store
still hold the old values. Follow a run with, over the same window:

    uv run h2mare compile -v <var_key> --start-date ... --end-date ...
    uv run h2mare parquet --start-date ... --end-date ...
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare import get_settings
from h2mare.models import BOAFrontSpec
from h2mare.processing.core.fronts import FrontProcessor, clear_staging
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import apply_cf_attrs
from h2mare.storage.zarr_catalog import ZarrCatalog

#: Days sampled per file in a dry run. Enough to tell "identical" from
#: "different" without paying for a year of detection.
DEFAULT_SAMPLE = 3


def fronts_for(var_key: str) -> dict[str, BOAFrontSpec]:
    """The var_key's declared front layers, or {} if it declares none."""
    var_config = get_settings().app_config.variables.get(var_key)
    if var_config is None:
        raise KeyError(f"'{var_key}' is not a configured var_key")
    return dict(var_config.boa_fronts or {})


def store_files(var_key: str, years: list[int] | None) -> list[Path]:
    """
    Store files for *var_key*, earliest first, optionally limited to *years*.

    Grouped by path, because the catalog holds a row per contiguous span
    rather than per file: sst's 2026 file is two rows where the rep segment
    ends and the nrt one begins, and detection would otherwise run over it
    twice.
    """
    df = ZarrCatalog(var_key).df
    if df.empty:
        return []

    spans = (
        df.groupby("path")
        .agg(start=("start_date", "min"), end=("end_date", "max"))
        .sort_values("start")
    )
    if years is not None:
        wanted = set(years)
        spans = spans[
            [
                bool(
                    wanted & set(range(pd.Timestamp(s).year, pd.Timestamp(e).year + 1))
                )
                for s, e in zip(spans["start"], spans["end"])
            ]
        ]
    return [Path(p) for p in spans.index]


def sample_times(ds: xr.Dataset, count: int) -> pd.DatetimeIndex:
    """*count* days spread across the file's axis, ends included."""
    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) <= count:
        return times
    idx = np.linspace(0, len(times) - 1, count).round().astype(int)
    return times[np.unique(idx)]


def detect(
    var_key: str, name: str, spec: BOAFrontSpec, ds: xr.Dataset, workers: int | None
) -> xr.DataArray:
    """One layer for whatever days *ds* carries."""
    if workers is not None:
        spec = BOAFrontSpec(
            source=spec.source, threshold=spec.threshold, n_workers=workers
        )
    return FrontProcessor(var_key, name, spec).from_dataset(ds)


def compare(computed: xr.DataArray, stored: xr.DataArray) -> str:
    """
    How far one freshly detected day sits from the stored one.

    Front pixels are reported alongside the distances because they are the
    thing detection actually decides: a cell at distance 0 is a front pixel,
    and one extra of those in a quiet stretch of ocean moves the distance of
    every cell around it. Distance differences alone read as far more
    disagreement than the detection shows.
    """
    new = computed.values.astype("float64")
    old = stored.values.astype("float64")

    new_nan, old_nan = np.isnan(new), np.isnan(old)
    mask_note = ""
    if not np.array_equal(new_nan, old_nan):
        mask_note = (
            f", NaN cells {int(old_nan.sum())} stored vs {int(new_nan.sum())} computed"
        )

    both = ~(new_nan | old_nan)
    if not both.any():
        return f"no cell finite in both{mask_note}"

    new_front, old_front = (new == 0) & both, (old == 0) & both
    shared = int((new_front & old_front).sum())
    detail = (
        f"fronts {int(old_front.sum())} stored / {int(new_front.sum())} computed, "
        f"{shared} shared"
    )

    delta = np.abs(new[both] - old[both])
    moved = int((delta > 1e-3).sum())
    return (
        f"{detail}; distances max {delta.max():.4g} km, mean {delta.mean():.4g} km, "
        f"{100 * moved / both.sum():.1f}% of cells moved >1 m{mask_note}"
    )


def survey_file(
    var_key: str,
    path: Path,
    fronts: dict[str, BOAFrontSpec],
    *,
    sample: int,
    workers: int | None,
) -> None:
    """Detect a few days and report how they compare with the store."""
    with xr.open_zarr(path, consolidated=False) as ds:
        times = sample_times(ds, sample)
        dates = ", ".join(t.strftime("%Y-%m-%d") for t in times)
        print(f"  {path.name}: {ds.sizes.get('time', 0)} day(s), sampling {dates}")

        for name, spec in fronts.items():
            if spec.source not in ds.data_vars:
                print(f"    {name}: source '{spec.source}' not in this file — skipped")
                continue
            subset = ds[[spec.source]].sel(time=times)
            computed = detect(var_key, name, spec, subset, workers)
            if name not in ds.data_vars:
                print(f"    {name}: not in the store yet — would be added")
                continue
            # Per day rather than pooled: a store's layers are written over
            # many runs, and a disagreement usually belongs to some of them.
            for day in times:
                line = compare(
                    computed.sel(time=day), ds[name].sel(time=day, method="nearest")
                )
                print(f"    {name} {day:%Y-%m-%d}: {line}")


def recompute_file(
    var_key: str,
    path: Path,
    fronts: dict[str, BOAFrontSpec],
    *,
    workers: int | None,
) -> None:
    """Re-detect every declared layer for one store file and write them back."""
    t0 = time.perf_counter()
    ds = xr.open_zarr(path, consolidated=False)
    layers: dict[str, xr.DataArray] = {}
    try:
        n_days = ds.sizes.get("time", 0)
        print(f"  {path.name}: {n_days} day(s)")
        for name, spec in fronts.items():
            if spec.source not in ds.data_vars:
                print(f"    {name}: source '{spec.source}' not in this file — skipped")
                continue
            layers[name] = detect(var_key, name, spec, ds, workers)
    finally:
        # The layers are backed by their staging stores, not by this file, and
        # the write below renames the file out from under any open handle.
        ds.close()

    if not layers:
        return

    ds_out = apply_cf_attrs(xr.Dataset(layers), native_var_key=var_key)
    write_append_zarr(var_key, ds_out, path)
    clear_staging(var_key)

    with xr.open_zarr(path, consolidated=False) as written:
        for name in layers:
            finite = int(np.isfinite(written[name].isel(time=0).values).sum())
            print(f"    {name}: written, {finite} finite cell(s) on day 1")
        print(
            f"    kept: {sorted(v for v in map(str, written.data_vars) if v not in layers)}"
        )
    print(f"    done in {time.perf_counter() - t0:.1f}s")


def run(
    var_key: str,
    *,
    years: list[int] | None,
    apply: bool,
    sample: int,
    workers: int | None,
) -> int:
    """Survey or recompute one var_key. Returns the number of files handled."""
    fronts = fronts_for(var_key)
    if not fronts:
        print(f"[{var_key}] declares no boa_fronts — nothing to recompute")
        return 0

    paths = store_files(var_key, years)
    if not paths:
        print(f"[{var_key}] no store files" + (f" for {years}" if years else ""))
        return 0

    declared = ", ".join(
        f"{n} (source {s.source}, threshold {s.threshold})" for n, s in fronts.items()
    )
    print(f"[{var_key}] {len(paths)} file(s), layers: {declared}")

    # Anything left by a killed run is a full copy of a period's layer, and the
    # detection below would not touch it.
    clear_staging(var_key)

    for path in paths:
        if apply:
            recompute_file(var_key, path, fronts, workers=workers)
        else:
            survey_file(var_key, path, fronts, sample=sample, workers=workers)
    return len(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("var_keys", nargs="*", help="variable(s) to recompute")
    parser.add_argument(
        "--all", action="store_true", help="every var_key that declares boa_fronts"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, help="limit to these years (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"days per file to compare in a dry run (default {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--workers", type=int, help="override the layer's configured n_workers"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="detect every day and rewrite the store (default is a dry run)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="drop the per-month detection logs (progress only)",
    )
    args = parser.parse_args(argv)

    if args.all:
        var_keys = [
            key
            for key, entry in get_settings().app_config.variables.items()
            if entry.boa_fronts
        ]
    elif args.var_keys:
        var_keys = args.var_keys
    else:
        parser.error("give at least one var_key, or --all")

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    handled = 0
    for var_key in var_keys:
        try:
            handled += run(
                var_key,
                years=args.years,
                apply=args.apply,
                sample=args.sample,
                workers=args.workers,
            )
        except Exception as e:  # noqa: BLE001 - one bad store must not stop the rest
            print(f"[{var_key}] failed: {e}")
            clear_staging(var_key)

    if not args.apply:
        print("\nDry run — re-run with --apply to rewrite the store(s).")
    else:
        print(f"\nRecomputed {handled} file(s).")
        print("h2ds and Parquet still hold the old values — compile and parquet next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
