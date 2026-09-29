"""
Snap float-drifted coordinate axes in a variable's Zarr store onto one canonical axis.

The same grid written on different occasions can disagree in the last
floating-point bits. Each file stays perfectly monotonic on its own, so nothing
looks wrong until ``open_mfdataset(combine="by_coords")`` compares the arrays
exactly, stops treating the axis as shared, and reports

    Resulting object does not have monotonic global indexes along dimension lon

or, for an axis that is not the one being concatenated along,

    cannot align objects with join='exact' ... along these coordinates: 'depth'

which names neither the cause nor the file. ``ZarrReader`` now snaps such axes
on read so a store in this state is still usable, but the store itself stays
inconsistent until it is repaired — which is what this does.

This rewrites **coordinate arrays only**. Every data array is untouched, so it
is a relabel rather than a regrid: nothing is interpolated and no value moves
between cells. Axes are only touched when they already agree with the canonical
one to within ``--tol``, so a genuinely different grid is reported and skipped
rather than silently overwritten.

The canonical axis is the earliest file's, matching what
``EDDIESProcessor._grid_from_store`` picks when it re-establishes a store grid.

Usage:
    uv run python scripts/repair_axis_drift.py eddies              # dry run
    uv run python scripts/repair_axis_drift.py eddies --apply
    uv run python scripts/repair_axis_drift.py --all               # survey every var_key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from h2mare import get_settings
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.storage.zarr_reader import AXIS_SNAP_TOL

#: Coordinates worth repairing, and the agreement each needs before it is
#: treated as the same axis. Imported rather than restated so the repair can
#: never disagree with what the reader snaps — a copy of the tolerance here
#: drifted out of step once already, leaving the o2 store's depth axis
#: unreadable *and* unrepairable. Time is excluded deliberately, at the source:
#: a differing time axis is different data, never a relabel.
COORDS = tuple(AXIS_SNAP_TOL)


def canonical_paths(var_key: str) -> list[Path]:
    """Store files for *var_key*, earliest first — so index 0 is canonical."""
    catalog = ZarrCatalog(var_key)
    df = catalog.df
    if df.empty:
        return []
    return [Path(p) for p in df.sort_values("start_date")["path"]]


def read_axes(path: Path) -> dict[str, np.ndarray]:
    with xr.open_zarr(path, consolidated=False) as ds:
        return {c: ds.coords[c].values for c in COORDS if c in ds.coords}


def classify(
    current: np.ndarray, reference: np.ndarray, tol: float
) -> tuple[str, float]:
    """One of 'match' | 'drift' | 'different', plus the max absolute delta."""
    if current.shape != reference.shape:
        return "different", float("nan")
    if np.array_equal(current, reference):
        return "match", 0.0
    delta = float(np.abs(current - reference).max())
    return ("drift" if delta <= tol else "different"), delta


def write_axis(path: Path, name: str, values: np.ndarray) -> None:
    """
    Overwrite one coordinate array in place.

    Written through zarr directly rather than by rewriting the dataset: the
    coordinate is its own small array in the store, so this touches kilobytes
    and leaves every data chunk exactly as it was.
    """
    root = zarr.open_group(str(path), mode="r+")
    arr = root[name]
    if arr.shape != values.shape:  # guarded again at the point of writing
        raise ValueError(f"{path.name}: {name} shape {arr.shape} != {values.shape}")
    arr[:] = values.astype(arr.dtype)


def repair(var_key: str, *, tol: float | None, apply: bool) -> int:
    """
    Report (and optionally fix) drift for one var_key. Returns files repaired.

    *tol* of ``None`` uses each coordinate's own limit — they are not in the
    same units, so one number cannot serve degrees and metres at once.
    """
    paths = canonical_paths(var_key)
    if len(paths) < 2:
        print(f"[{var_key}] {len(paths)} file(s) — nothing to compare")
        return 0

    reference = read_axes(paths[0])
    if not reference:
        print(f"[{var_key}] no {'/'.join(COORDS)} coords — skipped")
        return 0

    print(f"[{var_key}] {len(paths)} files, canonical axis from {paths[0].name}")

    repaired = 0
    for path in paths[1:]:
        axes = read_axes(path)
        drifted: dict[str, float] = {}
        blocked: dict[str, float] = {}

        for name, ref in reference.items():
            if name not in axes:
                continue
            limit = AXIS_SNAP_TOL[name] if tol is None else tol
            verdict, delta = classify(axes[name], ref, limit)
            if verdict == "drift":
                drifted[name] = delta
            elif verdict == "different":
                blocked[name] = delta

        if blocked:
            detail = ", ".join(f"{n} (max delta {d:.3e})" for n, d in blocked.items())
            print(f"  SKIP  {path.name}: {detail} — a different grid, not drift")
            continue

        if not drifted:
            continue

        detail = ", ".join(f"{n} by {d:.3e}" for n, d in drifted.items())
        if not apply:
            print(f"  DRIFT {path.name}: {detail}  (dry run)")
            continue

        for name in drifted:
            write_axis(path, name, reference[name])
        print(f"  FIXED {path.name}: {detail}")
        repaired += 1

    if not repaired and apply:
        print(f"[{var_key}] nothing to repair")
    return repaired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("var_keys", nargs="*", help="variable(s) to check")
    parser.add_argument(
        "--all", action="store_true", help="check every configured var_key"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=None,
        help=f"one limit for every coordinate; default is each one's own "
        f"({', '.join(f'{k}={v:g}' for k, v in AXIS_SNAP_TOL.items())})",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the repair (default is a dry run)"
    )
    args = parser.parse_args(argv)

    if args.all:
        var_keys = list(get_settings().app_config.variables)
    elif args.var_keys:
        var_keys = args.var_keys
    else:
        parser.error("give at least one var_key, or --all")

    total = 0
    failed: list[str] = []
    for var_key in var_keys:
        try:
            total += repair(var_key, tol=args.tol, apply=args.apply)
        except Exception as e:  # noqa: BLE001 - one bad store must not stop the survey
            print(f"[{var_key}] could not be checked: {e}")
            failed.append(var_key)

    if not args.apply:
        print("\nDry run — re-run with --apply to write.")
    else:
        print(f"\nRepaired {total} file(s).")
    # Non-zero when any var_key failed, as in recompute_fronts.py.
    if failed:
        print(f"\n{len(failed)} var_key(s) failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
