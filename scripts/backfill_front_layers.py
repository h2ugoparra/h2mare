"""
Backfill a variable's ``front_layers`` over its whole Zarr store.

The layers are produced at convert time (``plans/front-layers.md`` §6), so a
store converted before they were declared holds none — and the raw files are
long gone. Like ``recompute_fronts.py``, this needs no raw file: it reads the
field back from the store, detects, and writes only the layers, which the
store's merge semantics add beside everything already there.

Files are done in date order, and each is written before the next is read, so
a year's 30-day frequency is seeded with the previous year's stored masks —
the chain a continuous convert would have built. A run limited by ``--years``
seeds its first file from whatever masks the store holds before it (none: a
cold start, the frequency NaN until half a window is assessed), and afterwards
brings up to date the frequency of the stored days that follow its last file.

The layers must be declared in the active config — the deployed one, not the
repo's — together with their ``compiled_vars`` and ``variable_attrs``. Pause
the scheduled pipeline during an ``--apply``: both write the same files.

    uv run python scripts/backfill_front_layers.py sst                  # dry run
    uv run python scripts/backfill_front_layers.py sst --years 2024 --apply
    uv run python scripts/backfill_front_layers.py --all --apply

A dry run lists the files and detects one day per file, reporting its front
share, unassessed share, median gradient and a time estimate — a cheap check
that the fixed thresholds behave alike across the record before hours of
compute.

Cost: about 8-12 minutes per variable-year on 12 workers, so 4-6 hours per
variable for 1998-2026. Each file is rewritten through the store's tmp-and-swap,
so budget twice a file's size free beside the store, and about as much again
under ``INTERIM_DIR`` for the staging. Then, over the whole record:

    uv run h2mare compile -v <var_key> --start-date 1998-01-01 --end-date <last day>
    uv run h2mare parquet --start-date 1998-01-01 --end-date <last day>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import msgspec
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare import get_settings
from h2mare.models import FrontLayerSpec
from h2mare.processing.core.front_layers import (
    MaskReader,
    apply_front_layers,
    detect_fronts,
    read_retrying,
    recompute_following_frequency,
    store_mask_reader,
)
from h2mare.processing.core.fronts import clear_staging
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import apply_cf_attrs, chunk_dataset
from h2mare.storage.zarr_catalog import ZarrCatalog

#: Writes a dataset of layers into one store file.
Writer = Callable[[xr.Dataset, Path], None]


def layers_for(var_key: str) -> dict[str, FrontLayerSpec]:
    """The var_key's declared front layers, or {} if it declares none."""
    var_config = get_settings().app_config.variables.get(var_key)
    if var_config is None:
        raise KeyError(f"'{var_key}' is not a configured var_key")
    return dict(var_config.front_layers or {})


def with_workers(
    specs: dict[str, FrontLayerSpec], workers: int | None
) -> dict[str, FrontLayerSpec]:
    if workers is None:
        return specs
    return {n: msgspec.structs.replace(s, n_workers=workers) for n, s in specs.items()}


def store_spans(catalog: ZarrCatalog) -> pd.DataFrame:
    """
    One row per store file, earliest first: ``start``/``end``, indexed by path.

    Grouped by path because the catalog holds a row per contiguous span: sst's
    2026 file is two rows where the rep segment ends and the nrt one begins.
    """
    df = catalog.df
    if df.empty:
        return pd.DataFrame(columns=["start", "end"])
    return (
        df.groupby("path")
        .agg(start=("start_date", "min"), end=("end_date", "max"))
        .sort_values("start")
    )


def select(spans: pd.DataFrame, years: list[int] | None) -> list[Path]:
    """The files touching any of *years* (all when None), earliest first."""
    if years is None:
        return [Path(p) for p in spans.index]
    wanted = set(years)
    return [
        Path(p)
        for p, s, e in zip(spans.index, spans["start"], spans["end"])
        if wanted & set(range(pd.Timestamp(s).year, pd.Timestamp(e).year + 1))
    ]


def _needed(specs: dict[str, FrontLayerSpec]) -> list[str]:
    names = {s.source for s in specs.values()}
    names |= {s.confidence.var for s in specs.values() if s.confidence}
    return sorted(names)


def backfill_file(
    var_key: str,
    path: Path,
    specs: dict[str, FrontLayerSpec],
    read_masks: MaskReader,
    write: Writer,
) -> None:
    """Detect every declared layer over one store file and write them into it."""
    t0 = time.perf_counter()
    ds = xr.open_zarr(path, consolidated=False)
    try:
        missing = [v for v in _needed(specs) if v not in ds.data_vars]
        if missing:
            raise ValueError(f"{path.name} does not hold {missing}")
        print(f"  {path.name}: {ds.sizes.get('time', 0)} day(s)")
        out = apply_front_layers(ds[_needed(specs)], specs, var_key, read_masks)
        names = [v for n, s in specs.items() for v in s.output_names(n)]
        layers = out[names]
    finally:
        # The layers are backed by their staging, not by this file, and the
        # write renames the file out from under any open handle.
        ds.close()
    write(layers, path)
    print(f"    written in {time.perf_counter() - t0:.0f}s")


def refresh_following(
    specs: dict[str, FrontLayerSpec],
    read_masks: MaskReader,
    after: pd.Timestamp,
    spans: pd.DataFrame,
    write: Writer,
) -> int:
    """
    Bring the frequency of the stored days after *after* up to date.

    They were computed from the masks this run replaced. Returns how many days
    were rewritten: none when nothing stored follows, or what follows has no
    layers yet.
    """
    ds = recompute_following_frequency(specs, read_masks, after)
    if ds is None:
        return 0
    days = pd.DatetimeIndex(ds["time"].values)
    for path, start, end in zip(spans.index, spans["start"], spans["end"]):
        inside = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
        if len(inside):
            print(f"  {Path(path).name}: frequency of {len(inside)} following day(s)")
            write(ds.sel(time=inside), Path(path))
    return len(days)


def backfill(
    var_key: str,
    specs: dict[str, FrontLayerSpec],
    paths: list[Path],
    spans: pd.DataFrame,
    read_masks: MaskReader,
    write: Writer,
) -> None:
    """Backfill *paths* in order, then refresh the days that follow the last."""
    for path in paths:
        backfill_file(var_key, path, specs, read_masks, write)
        clear_staging(var_key)
    if paths:
        ends = dict(zip(map(Path, spans.index), spans["end"]))
        after = pd.Timestamp(ends[paths[-1]])
        refresh_following(specs, read_masks, after, spans, write)


def survey_file(path: Path, specs: dict[str, FrontLayerSpec], workers: int) -> None:
    """Detect the middle day of one file and report what the layers would hold."""
    with xr.open_zarr(path, consolidated=False) as ds:
        n = ds.sizes.get("time", 0)
        present = [v for n_, s in specs.items() for v in s.output_names(n_)]
        present = [v for v in present if v in ds.data_vars]
        note = f", would overwrite {present}" if present else ""
        if n == 0:
            print(f"  {path.name}: empty")
            return
        day = ds.isel(time=n // 2)
        lat = np.asarray(ds["lat"].values, dtype="float64")
        lon = np.asarray(ds["lon"].values, dtype="float64")
        print(f"  {path.name}: {n} day(s){note}")
        for name, spec in specs.items():
            values = read_retrying(lambda: day[spec.source].values, path.name)
            conf = (
                read_retrying(lambda: day[spec.confidence.var].values, path.name)
                if spec.confidence
                else None
            )
            t0 = time.perf_counter()
            mask, grad = detect_fronts(
                values,
                lat,
                lon,
                transform=spec.transform,
                sigma_km=spec.sigma_km,
                low_per_km=spec.low_per_km,
                high_per_km=spec.high_per_km,
                confidence=conf,
                confidence_max=spec.confidence.max if spec.confidence else None,
            )
            per_day = time.perf_counter() - t0
            sea = np.isfinite(values)
            assessed = np.isfinite(mask)
            print(
                f"    {name} {pd.Timestamp(day['time'].values):%Y-%m-%d}: "
                f"fronts {100 * np.nansum(mask) / max(assessed.sum(), 1):.2f}% of "
                f"assessed, not assessed {100 * (sea & ~assessed).sum() / max(sea.sum(), 1):.1f}% "
                f"of sea, grad p50 {np.nanmedian(grad):.4g}; "
                f"~{per_day * n / workers / 60:.0f} min for the file"
            )


def run(
    var_key: str, *, years: list[int] | None, apply: bool, workers: int | None
) -> int:
    """Survey or backfill one var_key. Returns the number of files handled."""
    specs = with_workers(layers_for(var_key), workers)
    if not specs:
        print(
            f"[{var_key}] declares no front_layers — declare them in the active "
            f"config.yaml first (plans/front-layers.md §6.6)"
        )
        return 0

    catalog = ZarrCatalog(var_key)
    spans = store_spans(catalog)
    paths = select(spans, years)
    if not paths:
        print(f"[{var_key}] no store files" + (f" for {years}" if years else ""))
        return 0

    declared = ", ".join(
        f"{n} (source {s.source}, sigma {s.sigma_km} km, "
        f"{s.low_per_km}/{s.high_per_km} per km)"
        for n, s in specs.items()
    )
    print(f"[{var_key}] {len(paths)} file(s), layers: {declared}")

    # Anything left by a killed run is a full copy of a period's layers.
    clear_staging(var_key)

    if not apply:
        pool = workers or min(s.n_workers or 10 for s in specs.values())
        for path in paths:
            survey_file(path, specs, pool)
        return len(paths)

    def write(ds: xr.Dataset, path: Path) -> None:
        ds = apply_cf_attrs(ds, native_var_key=var_key)
        write_append_zarr(var_key, chunk_dataset(ds), path)
        # The next file's seed reads these masks back through the catalog.
        catalog.refresh(force=True)

    backfill(var_key, specs, paths, spans, store_mask_reader(catalog, var_key), write)
    return len(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("var_keys", nargs="*", help="variable(s) to backfill")
    parser.add_argument(
        "--all", action="store_true", help="every var_key that declares front_layers"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, help="limit to these years (default: all)"
    )
    parser.add_argument(
        "--workers", type=int, help="override the layers' configured n_workers"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="detect every day and write the layers (default is a dry run)",
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
            if entry.front_layers
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
    failed: list[str] = []
    for var_key in var_keys:
        try:
            handled += run(
                var_key, years=args.years, apply=args.apply, workers=args.workers
            )
        except Exception as e:  # noqa: BLE001 - one bad store must not stop the rest
            print(f"[{var_key}] failed: {e}")
            clear_staging(var_key)
            failed.append(var_key)

    if not args.apply:
        print("\nDry run — re-run with --apply to write the layers.")
    else:
        print(f"\nBackfilled {handled} file(s).")
        print("h2ds and Parquet do not hold them yet — compile and parquet next.")
    if failed:
        print(f"\n{len(failed)} var_key(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
