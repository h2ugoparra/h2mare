"""
Re-chunk existing Zarr stores to spatial tiles + float32 (Fix #1).

Existing per-variable stores were written with full-size spatial chunks (or, for
older stores, one global slice per timestep in float64), so a small-bbox / point
extraction must read and decompress the whole grid for every timestep. This is
the ~15-minute case profiled in scripts/profile_extract_chunking.py.

This tool rewrites each period .zarr file through ``chunk_dataset`` (spatial
tiling + float32), so subsequent extractions touch only the overlapping tiles.
It does NOT change pipeline behaviour — it only reshapes data already on disk.

Safety:
  * --dry-run (default) only reports each file's dtype / chunks / size and the
    layout it WOULD get. Nothing is written.
  * --apply rewrites via an atomic tmp -> .bak swap (same pattern as
    storage.write_append_zarr); the original is restored on any failure.
  * Files already matching the target spatial chunking are skipped.
  * Each rewrite needs transient free space ~= the size of that one file.

Run:
    uv run python scripts/rechunk_stores.py --var-keys seapodym            # dry run
    uv run python scripts/rechunk_stores.py --var-keys seapodym --apply
    uv run python scripts/rechunk_stores.py --apply                        # all stores
"""

from __future__ import annotations

import argparse
import gc
import shutil
import time
from pathlib import Path

import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.storage.xarray_helpers import chunk_dataset
from h2mare.utils.paths import resolve_store_path

SPATIAL_DIMS = {"lat", "lon", "latitude", "longitude", "x", "y"}

# Canonical format is float32 + spatial tiles. We keep the source compressor but
# drop the chunk-defining keys (else to_zarr reuses chunks=(1, full, full) and
# raises "would overlap multiple Dask chunks") and the int-packing keys
# (dtype/scale/offset) plus the integer _FillValue, so the rechunked float32
# arrays are written raw with NaN fill rather than re-packed to int16.
_DROP_ENCODING = {
    "chunks",
    "preferred_chunks",
    "shards",
    "dtype",
    "scale_factor",
    "add_offset",
    "_FillValue",
}


def _du_mb(path: Path) -> float:
    """Approximate on-disk size of a zarr directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1e6


def _spatial_chunks_ok(ds: xr.Dataset, spatial_chunk: int) -> bool:
    """True if every spatial dim is already tiled at <= spatial_chunk.

    Checks each variable's own chunks rather than the dataset-level
    ``ds.chunksizes`` aggregate, which raises "inconsistent chunks along
    dimension ..." when variables disagree (e.g. chl has time-chunks of 1 while
    chl_fdist has 2). Per-variable chunks never trigger that, and chunk_dataset's
    ``.chunk()`` unifies them on rewrite anyway.
    """
    for var in ds.data_vars:
        da = ds[var]
        if da.chunks is None:
            continue
        for dim, sizes in zip(da.dims, da.chunks):
            if dim.lower() in SPATIAL_DIMS and sizes and sizes[0] > spatial_chunk:
                return False
    return True


def _describe(ds: xr.Dataset) -> str:
    """One-line summary of the heaviest var: on-disk dtype + chunk shape."""
    main = max(ds.data_vars, key=lambda v: ds[v].nbytes, default=None)
    if main is None:
        return "no data vars"
    disk_dtype = ds[main].encoding.get("dtype", ds[main].dtype)
    shape = tuple(c[0] for c in ds[main].chunks) if ds[main].chunks else "unchunked"
    return f"{main} {disk_dtype} chunks={shape}"


def _rewrite(path: Path, spatial_chunk: int, target_mb: int) -> None:
    """Atomically re-tile a single zarr file into float32 + spatial tiles."""
    ds = xr.open_zarr(path, consolidated=False)
    ds_out = chunk_dataset(ds, target_mb=target_mb, spatial_chunk=spatial_chunk)

    # Keep the source compressor, drop chunk + int-packing encoding so the
    # rechunked float32 data is written raw (see _DROP_ENCODING).
    for var in ds_out.variables:
        ds_out[var].encoding = {
            k: v for k, v in ds[var].encoding.items() if k not in _DROP_ENCODING
        }

    tmp = path.with_name(path.name + ".tmp")
    bak = path.with_name(path.name + ".bak")
    shutil.rmtree(tmp, ignore_errors=True)

    t0 = time.perf_counter()
    ds_out.to_zarr(tmp)
    # Release handles before the swap (Windows locks open zarr directories).
    ds_out.close()
    ds.close()
    del ds_out, ds
    gc.collect()

    shutil.move(str(path), str(bak))
    try:
        shutil.move(str(tmp), str(path))
        shutil.rmtree(str(bak), ignore_errors=True)
    except Exception:
        shutil.rmtree(str(path), ignore_errors=True)
        shutil.move(str(bak), str(path))
        raise
    logger.success(f"    rewrote in {time.perf_counter() - t0:.1f}s")


def process_var_key(
    var_key: str,
    spatial_chunk: int,
    target_mb: int,
    apply: bool,
    match: str | None = None,
    limit: int | None = None,
) -> tuple[int, int]:
    """Report (and optionally rewrite) .zarr files for one var_key.

    ``match`` keeps only files whose name contains the substring. ``limit`` caps
    the number of rewrites in this call (apply mode only). Returns
    (n_rewritten, n_skipped).
    """
    var_cfg = get_settings().app_config.variables[var_key]
    store_root = resolve_store_path(var_cfg, warn_if_missing=False)
    files = sorted(store_root.glob("*.zarr"))
    if match:
        files = [f for f in files if match in f.name]

    if not files:
        logger.info(f"[{var_key}] no matching .zarr files in {store_root} — skipping")
        return 0, 0

    logger.info(f"[{var_key}] {len(files)} file(s) in {store_root}")
    rewritten = skipped = 0
    for path in files:
        if apply and limit is not None and rewritten >= limit:
            logger.info(f"  reached --limit {limit}, stopping")
            break
        ds = xr.open_zarr(path, consolidated=False)
        before = _describe(ds)
        size_mb = _du_mb(path)
        # Canonical = spatially tiled AND float32. A tiled-but-int16 file (e.g.
        # one re-chunked under the earlier packing-preserving pass) still needs
        # converting, so dtype is part of the skip test, not just chunk size.
        main = max(ds.data_vars, key=lambda v: ds[v].nbytes, default=None)
        is_f32 = main is not None and str(
            ds[main].encoding.get("dtype", ds[main].dtype)
        ) == "float32"
        already = _spatial_chunks_ok(ds, spatial_chunk) and is_f32
        try:
            target = _describe(chunk_dataset(ds, target_mb=target_mb, spatial_chunk=spatial_chunk))
        except ValueError as e:
            target = f"<cannot tile: {e}>"
        ds.close()

        if already:
            logger.info(
                f"  {path.name}  ({size_mb:.0f} MB)  {before}  -> already canonical"
            )
            skipped += 1
            continue

        logger.info(f"  {path.name}  ({size_mb:.0f} MB)")
        logger.info(f"      now:    {before}")
        logger.info(f"      target: {target}")

        if apply:
            # Per-file isolation: a failed rewrite leaves that file's original
            # intact (error precedes the atomic swap) and must not abort the rest
            # of the batch.
            try:
                _rewrite(path, spatial_chunk, target_mb)
                rewritten += 1
            except Exception as e:
                logger.error(f"    REWRITE FAILED for {path.name}: {e}")

    return rewritten, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--var-keys",
        nargs="*",
        default=None,
        help="Var keys to re-chunk. Default: all that have a zarr store on disk.",
    )
    parser.add_argument("--spatial-chunk", type=int, default=256)
    parser.add_argument("--target-mb", type=int, default=32)
    parser.add_argument(
        "--match",
        default=None,
        help="Only process files whose name contains this substring (e.g. a year).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of files to rewrite total (apply mode). Default: no cap.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files. Without this flag the script only reports.",
    )
    args = parser.parse_args()

    var_keys = args.var_keys or get_settings().get_available_var_keys()
    mode = "APPLY" if args.apply else "DRY-RUN"
    logger.info(
        f"[{mode}] spatial_chunk={args.spatial_chunk} target_mb={args.target_mb} "
        f"| {len(var_keys)} var_key(s)"
        + (f" | match={args.match!r}" if args.match else "")
        + (f" | limit={args.limit}" if args.limit is not None else "")
    )

    total_rewritten = total_skipped = 0
    remaining = args.limit
    for var_key in var_keys:
        if args.apply and remaining is not None and remaining <= 0:
            break
        try:
            r, s = process_var_key(
                var_key,
                args.spatial_chunk,
                args.target_mb,
                args.apply,
                match=args.match,
                limit=remaining,
            )
            total_rewritten += r
            total_skipped += s
            if remaining is not None:
                remaining -= r
        except Exception as e:
            logger.error(f"[{var_key}] failed: {e}")

    logger.info("=" * 60)
    if args.apply:
        logger.success(
            f"Done. Rewrote {total_rewritten} file(s), skipped {total_skipped} "
            f"already-tiled."
        )
    else:
        logger.info(
            f"Dry run complete. {total_skipped} already tiled; rerun with --apply "
            f"to rewrite the rest."
        )


if __name__ == "__main__":
    main()
