"""
Rewrite a variable's Zarr files in the layout ``chunk_dataset`` would choose now.

A store keeps the chunking it was created with: appends inherit it from disk,
and nothing else rewrites it. So a store written before a layout rule changed
keeps the old one indefinitely, and every read pays for it. ``o2`` is the case
this was written for — its arrays are ``[2, 46, 256, 256]``, all 46 depth
levels in one chunk, from before non-time non-spatial dims were chunked to 1.
Compile wants four levels and has to decompress all 46 to get them, which is
what dask reports as

    UserWarning: The specified chunks separate the stored chunks along
    dimension "depth" starting at index 1.

This is a pure relayout: every value, attribute and coordinate is written back
as it was read, only the chunk grid changes. It is also the only way to apply a
layout change to data already on disk short of re-converting from raw files,
which for a variable that does not set ``archive_raw`` means re-downloading.

    uv run python scripts/rechunk_store.py o2                   # what would change
    uv run python scripts/rechunk_store.py o2 --years 1998 --apply
    uv run python scripts/rechunk_store.py --all                # survey every var_key

Each file is rewritten beside itself and swapped in — ``foo.zarr.tmp`` becomes
``foo.zarr`` once it has been read back and checked against the original, with
the original held as ``foo.zarr.bak`` until then. Those are the same suffixes
the write path uses, so ``recover_zarr_store`` reconciles whatever an
interrupted run leaves behind. Budget twice a file's size free beside the store.

Note that ``chunk_dataset`` also casts float64 to float32, as it does on every
write path. A store holding float64 is therefore reported as needing a rewrite
that would change its dtype, and is skipped unless ``--allow-downcast`` says
otherwise.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from h2mare import get_settings
from h2mare.storage.recovery import recover_zarr_store
from h2mare.storage.xarray_helpers import (
    chunk_dataset,
    drop_conflicting_missing_value,
)
from h2mare.storage.zarr_catalog import ZarrCatalog


def store_files(var_key: str, years: list[int] | None) -> list[Path]:
    """
    Store files for *var_key*, earliest first, optionally limited to *years*.

    Grouped by path: the catalog holds a row per contiguous span, and a file
    split between a rep and an nrt segment has two.
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


def stored_chunks(ds: xr.Dataset) -> dict[str, tuple[int, ...]]:
    """What each data variable is chunked as on disk."""
    return {
        str(name): tuple(da.encoding.get("chunks") or ())
        for name, da in ds.data_vars.items()
    }


def planned_chunks(ds: xr.Dataset) -> tuple[dict[str, tuple[int, ...]], xr.Dataset]:
    """What ``chunk_dataset`` would write it as, and the dataset it planned."""
    planned = chunk_dataset(ds)
    return {
        str(name): tuple(c[0] for c in da.chunks or ())
        for name, da in planned.data_vars.items()
    }, planned


def downcast_vars(ds: xr.Dataset, planned: xr.Dataset) -> list[str]:
    """Variables the relayout would also change the dtype of."""
    return sorted(
        str(name)
        for name in ds.data_vars
        if ds[name].dtype != planned[name].dtype  # type: ignore[index]
    )


def verify(original: Path, rewritten: Path) -> None:
    """
    Check the rewrite against the original before anything is swapped.

    Shape, dtype and the time axis, then one time step of the largest variable
    read from both — enough to catch a write that silently produced a different
    grid or lost a variable, which is the failure worth guarding a swap against.
    """
    with (
        xr.open_zarr(original, consolidated=False) as old,
        xr.open_zarr(rewritten, consolidated=False) as new,
    ):
        if set(old.data_vars) != set(new.data_vars):
            raise ValueError(
                f"{rewritten.name}: variables differ — "
                f"{sorted(map(str, set(old.data_vars) ^ set(new.data_vars)))}"
            )
        if dict(old.sizes) != dict(new.sizes):
            raise ValueError(
                f"{rewritten.name}: sizes differ — {dict(old.sizes)} vs {dict(new.sizes)}"
            )
        for coord in old.coords:
            if not np.array_equal(old[coord].values, new[coord].values):
                raise ValueError(f"{rewritten.name}: coordinate '{coord}' differs")

        name = max(old.data_vars, key=lambda n: old[n].size)
        sel = {"time": 0} if "time" in old[name].dims else {}
        a, b = old[name].isel(sel).values, new[name].isel(sel).values
        if a.dtype == b.dtype and not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"{rewritten.name}: '{name}' values differ")


def rechunk_file(path: Path, *, allow_downcast: bool) -> bool:
    """Rewrite one file in the current layout. Returns whether it was rewritten."""
    t0 = time.perf_counter()
    ds = xr.open_zarr(path, consolidated=False)
    try:
        before = stored_chunks(ds)
        after, planned = planned_chunks(ds)
        if before == after:
            print(f"  {path.name}: already in the current layout")
            return False

        casts = downcast_vars(ds, planned)
        if casts and not allow_downcast:
            print(
                f"  SKIP  {path.name}: rewriting would also cast {casts} to "
                f"float32; pass --allow-downcast to accept that"
            )
            return False

        # A store that carries both _FillValue and a contradicting
        # missing_value cannot be written back from itself — chl does.
        drop_conflicting_missing_value(planned)

        # Inherited chunk encodings fight the chunking planned above: to_zarr
        # refuses a write whose dask chunks straddle the encoding's.
        for name in planned.variables:
            planned[name].encoding.pop("chunks", None)
            planned[name].encoding.pop("preferred_chunks", None)

        tmp = path.with_name(path.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        planned.to_zarr(tmp)
    finally:
        # Before the swap: Windows will not rename a directory anything still
        # holds a handle into.
        ds.close()

    verify(path, tmp)

    bak = path.with_name(path.name + ".bak")
    shutil.rmtree(bak, ignore_errors=True)
    path.rename(bak)
    tmp.rename(path)
    shutil.rmtree(bak, ignore_errors=True)

    changed = {n: (before[n], after[n]) for n in after if before.get(n) != after[n]}
    for name, (was, now) in changed.items():
        print(f"  {path.name}: {name} {was} -> {now}")
    print(f"    rewritten in {time.perf_counter() - t0:.1f}s")
    return True


def survey_file(path: Path) -> bool:
    """Report what a rewrite would change. Returns whether anything would."""
    with xr.open_zarr(path, consolidated=False) as ds:
        before = stored_chunks(ds)
        after, planned = planned_chunks(ds)
        casts = downcast_vars(ds, planned)

    changed = {n: (before[n], after[n]) for n in after if before.get(n) != after[n]}
    if not changed:
        print(f"  {path.name}: already in the current layout")
        return False
    for name, (was, now) in changed.items():
        print(f"  {path.name}: {name} {was} -> {now}")
    if casts:
        print(f"    note: would also cast {casts} to float32")
    return True


def run(var_key: str, *, years: list[int] | None, apply: bool, allow_downcast: bool):
    """Survey or rewrite one var_key's files. Returns how many need/needed it."""
    catalog = ZarrCatalog(var_key)
    recover_zarr_store(catalog.store_root)

    paths = store_files(var_key, years)
    if not paths:
        print(f"[{var_key}] no store files" + (f" for {years}" if years else ""))
        return 0

    print(f"[{var_key}] {len(paths)} file(s) under {catalog.store_root}")
    touched = 0
    for path in paths:
        if apply:
            touched += rechunk_file(path, allow_downcast=allow_downcast)
        else:
            touched += survey_file(path)

    if apply and touched:
        catalog.refresh(force=True)
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("var_keys", nargs="*", help="variable(s) to rechunk")
    parser.add_argument("--all", action="store_true", help="every configured var_key")
    parser.add_argument(
        "--years", nargs="+", type=int, help="limit to these years (default: all)"
    )
    parser.add_argument(
        "--allow-downcast",
        action="store_true",
        help="rewrite even when it would cast float64 variables to float32",
    )
    parser.add_argument(
        "--apply", action="store_true", help="rewrite (default is a dry run)"
    )
    args = parser.parse_args(argv)

    if args.all:
        var_keys = list(get_settings().app_config.variables)
    elif args.var_keys:
        var_keys = args.var_keys
    else:
        parser.error("give at least one var_key, or --all")

    total = 0
    for var_key in var_keys:
        try:
            total += run(
                var_key,
                years=args.years,
                apply=args.apply,
                allow_downcast=args.allow_downcast,
            )
        except Exception as e:  # noqa: BLE001 - one bad store must not stop the rest
            print(f"[{var_key}] failed: {e}")

    if not args.apply:
        print(f"\nDry run — {total} file(s) would be rewritten. Re-run with --apply.")
    else:
        print(f"\nRewrote {total} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
