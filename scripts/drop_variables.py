"""
Delete variables from a var_key's native Zarr store.

For layers that are wrong and no longer produced — the BOA front distances
(``sst_fdist``, ``chl_fdist``), retired by ``plans/front-layers.md`` because
every stored value predates the 2026-09 distance-metric fix. Only the named
arrays go; every other variable, the coordinates and the root attributes stay,
and the consolidated metadata is rewritten to match.

A name the active config still produces for the var_key is refused: the next
convert would write it back, and a native extraction would raise over every
day it is missing from. Take it out of the config first.

    uv run python scripts/drop_variables.py sst sst_fdist             # dry run
    uv run python scripts/drop_variables.py sst sst_fdist --apply

Irreversible: the arrays are deleted, not moved. h2ds and the Parquet store
are separate copies and are not touched — the store merges keep a variable an
incoming dataset does not carry, so a recompile or a Parquet rebuild does not
remove it from them either.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import xarray as xr
import zarr

from h2mare import get_settings
from h2mare.storage.zarr_catalog import ZarrCatalog

#: Never deletable: they index every other array.
COORDS = {"time", "lat", "lon", "depth"}


def produced(var_key: str) -> set[str]:
    """Every name the active config writes into *var_key*'s store."""
    entry = get_settings().app_config.variables.get(var_key)
    if entry is None:
        raise KeyError(f"'{var_key}' is not a configured var_key")
    names = set(entry.compiled_vars or [])
    names |= set((entry.source_renames or {}).values())
    names |= set(entry.derived_vars or {})
    names |= set(entry.boa_fronts or {})
    for name, spec in (entry.front_layers or {}).items():
        names |= set(spec.output_names(name))
    return names


def refused(names: list[str], produced_names: set[str]) -> list[str]:
    """Why each refused name may not be dropped; empty when all may."""
    reasons = []
    for n in names:
        if n in COORDS:
            reasons.append(f"'{n}' is a coordinate")
        elif n in produced_names:
            reasons.append(f"'{n}' is still produced by the config")
    return reasons


def _size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def drop_file(path: Path, names: list[str], *, apply: bool) -> int:
    """Drop *names* from one store file; return the bytes they held."""
    group = zarr.open_group(path, mode="r+" if apply else "r")
    present = [n for n in names if n in group.array_keys()]
    if not present:
        print(f"  {path.name}: none present")
        return 0
    freed = sum(_size(path / n) for n in present)
    print(f"  {path.name}: {present}, {freed / 1e6:.0f} MB")
    if not apply:
        return freed

    consolidated = group.metadata.consolidated_metadata is not None
    for n in present:
        del group[n]
        # The store deletes the array's metadata; a chunk left behind by an
        # interrupted write is not always listed, so clear the directory too.
        shutil.rmtree(path / n, ignore_errors=True)
    if consolidated:
        zarr.consolidate_metadata(path)

    with xr.open_zarr(path, consolidated=consolidated) as ds:
        left = set(present) & set(map(str, ds.variables))
        if left:
            raise RuntimeError(f"{path.name}: {sorted(left)} still listed")
        print(f"    kept: {sorted(map(str, ds.data_vars))}")
    return freed


def run(var_key: str, names: list[str], *, apply: bool) -> int:
    """Drop *names* from every file of *var_key*. Returns the files touched."""
    reasons = refused(names, produced(var_key))
    if reasons:
        raise ValueError(f"not dropping: {'; '.join(reasons)}")

    catalog = ZarrCatalog(var_key)
    paths = (
        sorted({Path(p) for p in catalog.df["path"]}) if not catalog.df.empty else []
    )
    print(f"[{var_key}] {len(paths)} file(s), dropping {names}")
    freed, touched = 0, 0
    for path in paths:
        n = drop_file(path, names, apply=apply)
        freed += n
        touched += n > 0
    if apply and touched:
        # The index lists each file's variables.
        catalog.refresh(force=True)
    verb = "freed" if apply else "would free"
    print(f"[{var_key}] {touched} file(s), {verb} {freed / 1e9:.2f} GB")
    return touched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("var_key", help="the store to drop from")
    parser.add_argument("names", nargs="+", help="variable(s) to delete")
    parser.add_argument(
        "--apply", action="store_true", help="delete (default is a dry run)"
    )
    args = parser.parse_args(argv)

    try:
        run(args.var_key, args.names, apply=args.apply)
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"[{args.var_key}] failed: {e}")
        return 1
    if not args.apply:
        print("\nDry run — re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
