"""
Profile the effect of spatial chunking (Fix #1) on Extractor reads.

Hypothesis: the compiled store keeps lat/lon at full global size in every chunk
(see storage/xarray_helpers.chunk_dataset), so subsetting to a small bbox still
reads & decompresses the whole global slice per timestep. Spatially tiling the
store should let a small bbox touch only a couple of tiles.

Two profiles run on the SAME data, each under two layouts:

    BEFORE: the store as it exists today (global spatial chunks)
    AFTER : a temporary copy rechunked into spatial tiles

  1. CONTIGUOUS  — read the full (bbox x time) cube. Isolates the chunk-layout
     I/O effect (Fix #1) with no sparse/vindex overhead.
  2. SPARSE      — mirrors the real CSV point path: random (date, lon, lat)
     samples -> cat.open_dataset(dates=..., bbox=...) -> Extractor.extract_from_csv.
     Surfaces the sparse-open cost (map_dates_to_paths loop) and the dask
     pointwise-vindex cost (Fixes #2/#3), and how both shift under tiling.

It does NOT modify the real store. The AFTER copy is written to a temp zarr and
deleted at the end. Timings scale ~linearly with N_DAYS / N_POINTS, so a few
hundred is enough to see the ratio; bump them toward the real run for absolutes.

Run:
    uv run python scripts/profile_extract_chunking.py
"""

from __future__ import annotations

import shutil
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.config import get_settings
from h2mare.processing.extractor import Extractor
from h2mare.storage.xarray_helpers import xr_float64_to_float32
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox

# ---- knobs ---------------------------------------------------------------
VAR_KEY = "seapodym"
BBOX = BBox.from_tuple((-45.0, 20.0, 0.0, 50.0))  # xmin, ymin, xmax, ymax
N_DAYS = 365                  # length of the window to read
N_POINTS = 1500               # synthetic point samples for the sparse path
SPATIAL_CHUNK = {"lat": 240, "lon": 240}  # proposed tile size for Fix #1
INDEX_COL = "row_id"
SEED = 0
# --------------------------------------------------------------------------


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    print(f"  {label:<34} {time.perf_counter() - t0:8.2f} s")


def describe(ds, label: str) -> None:
    nbytes = sum(v.nbytes for v in ds.data_vars.values())
    main = max(ds.data_vars, key=lambda v: ds[v].nbytes)
    chunks = ds[main].chunks
    chunk_shape = tuple(c[0] for c in chunks) if chunks else "in-memory"
    print(f"  {label}: dims={dict(ds.sizes)} | {nbytes / 1e6:.1f} MB "
          f"| {main} chunk={chunk_shape}")


def make_points(times: np.ndarray) -> pd.DataFrame:
    """Synthesize a CSV-like frame of random (date, lon, lat) samples in the bbox."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(
        {
            "time": rng.choice(times, size=N_POINTS),
            "lon": rng.uniform(BBOX.xmin, BBOX.xmax, size=N_POINTS),
            "lat": rng.uniform(BBOX.ymin, BBOX.ymax, size=N_POINTS),
        }
    )
    df.index.name = INDEX_COL
    return df


def main() -> None:
    cat = ZarrCatalog(VAR_KEY)
    cov = cat.get_time_coverage()
    if cov is None:
        raise SystemExit(f"No coverage for {VAR_KEY}")

    start = pd.Timestamp(cov.start)
    end = start + pd.Timedelta(days=N_DAYS - 1)
    print(f"{VAR_KEY}: window {start.date()} -> {end.date()} ({N_DAYS} days) | bbox {BBOX}")

    # =================== CONTIGUOUS (Fix #1 isolation) ===================
    print("\n[CONTIGUOUS]  BEFORE (store as-is):")
    ds_now = cat.open_dataset(start_date=start, end_date=end, bbox=BBOX)
    times = ds_now.time.values  # cheap coord read; reused by the sparse path
    describe(ds_now, "bbox cube (lazy)")
    with timed("compute bbox cube"):
        ds_now.compute()

    # ---------- build a spatially-tiled copy of the SAME window ----------
    tmp = get_settings().INTERIM_DIR / f"_profile_{VAR_KEY}_tiled.zarr"
    if tmp.exists():
        shutil.rmtree(tmp)

    # open the FULL global window (no bbox) so the rewrite reflects a real
    # spatially-tiled store, then read the bbox back out of it
    ds_full = cat.open_dataset(start_date=start, end_date=end)
    ds_full = xr_float64_to_float32(ds_full).chunk(
        {"time": ds_now.sizes["time"], **SPATIAL_CHUNK}
    )
    print(f"\nWriting tiled copy -> {tmp.name} (setup, not timed)...")
    with timed("write tiled copy"):
        ds_full.to_zarr(tmp, mode="w", consolidated=True)

    print("\n[CONTIGUOUS]  AFTER (spatially tiled):")
    ds_tiled = xr.open_zarr(tmp, consolidated=True).sel(
        lat=slice(BBOX.ymin, BBOX.ymax), lon=slice(BBOX.xmin, BBOX.xmax)
    )
    describe(ds_tiled, "bbox cube (lazy)")
    with timed("compute bbox cube"):
        ds_tiled.compute()

    # =================== SPARSE (real CSV point path) ===================
    points = make_points(times)
    sparse_dates = sorted(pd.DatetimeIndex(points["time"]).drop_duplicates())
    print(f"\n[SPARSE]  {N_POINTS} points over {len(sparse_dates)} unique dates")

    print("[SPARSE]  BEFORE (store as-is):")
    with timed("open(dates=...) + sortby"):
        ds_sp = cat.open_dataset(dates=sparse_dates, bbox=BBOX).sortby("time")
    with timed("extract_from_csv (vindex+compute)"):
        Extractor.extract_from_csv(points, ds_sp, INDEX_COL)

    print("[SPARSE]  AFTER (spatially tiled):")
    # map_dates_to_paths cost is layout-independent (same catalog), so the open
    # figure above carries over; here we isolate the vindex+compute under tiling.
    ds_sp_tiled = xr.open_zarr(tmp, consolidated=True).sel(
        lat=slice(BBOX.ymin, BBOX.ymax), lon=slice(BBOX.xmin, BBOX.xmax)
    ).sel(time=pd.DatetimeIndex(sparse_dates), method="nearest")
    with timed("extract_from_csv (vindex+compute)"):
        Extractor.extract_from_csv(points, ds_sp_tiled, INDEX_COL)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nDone.")
    print("  CONTIGUOUS before/after  -> Fix #1 (spatial chunking) payoff")
    print("  SPARSE 'open' time       -> map_dates_to_paths overhead (Fix #3)")
    print("  SPARSE 'extract' before/after -> vindex cost vs tiling (Fixes #1+#2)")


if __name__ == "__main__":
    main()
