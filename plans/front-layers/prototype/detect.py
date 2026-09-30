"""
Prototype front detection for SDM layers (read-only against the stores).

Scale-aware Canny-style detector on L4 fields:
  gap fill -> Gaussian (sigma in km) -> per-km gradient (cos-lat metric)
  -> non-maximum suppression -> hysteresis -> optional confidence mask.
Plus the current BOA (h2mare.processing.core.fronts.boa) as the baseline.
"""

from __future__ import annotations

import pathlib

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, label

R_KM = 111.2  # km per degree of latitude

# (store var, transform, sigma_km, mask var)
SPEC = {
    "sst": dict(transform=None, sigma_km=5.0, err="analysis_error"),
    "chl": dict(transform="log10", sigma_km=7.0, err=None),
}


def store_file(cfg, vk: str, year: int) -> pathlib.Path:
    from h2mare.utils.paths import store_root_for

    v = cfg.variables[vk]
    return sorted(pathlib.Path(store_root_for(v) / v.local_folder).glob(f"*{year}.zarr"))[0]


def prepare(values: np.ndarray, transform: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Transformed field with gaps filled from the nearest valid cell, and the gap mask."""
    f = values.astype("float64")
    if transform == "log10":
        f = np.log10(np.where(f > 0, f, np.nan))
    gaps = np.isnan(f)
    if gaps.all():
        return f, gaps
    if gaps.any():
        idx = distance_transform_edt(gaps, return_distances=False, return_indices=True)
        f = f[tuple(idx)]
    return f, gaps


def metric_gradient(f: np.ndarray, lat: np.ndarray, lon: np.ndarray, sigma_km: float):
    """
    Gaussian-smoothed gradient per km. Returns magnitude (units/km) and the
    index-space components (for the NMS direction).
    """
    dy = abs(lat[1] - lat[0]) * R_KM
    dx_row = abs(lon[1] - lon[0]) * R_KM * np.cos(np.deg2rad(lat))
    dx = dx_row[:, None]
    # sigma in km on both axes: separable, the east-west width set per row
    # because a longitude cell narrows with latitude.
    s = gaussian_filter1d(f, sigma_km / dy, axis=0, mode="nearest")
    s = np.stack([gaussian_filter1d(row, sigma_km / w, mode="nearest") for row, w in zip(s, dx_row)])
    gy_c, gx_c = np.gradient(s)  # per cell, index space
    mag = np.hypot(gx_c / dx, gy_c / dy)
    return mag, gx_c, gy_c


def nms(mag: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Keep pixels that are the maximum across the front (4 direction bins)."""
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    p = np.pad(mag, 1, mode="edge")
    c = p[1:-1, 1:-1]
    e, w = p[1:-1, 2:], p[1:-1, :-2]
    n, s = p[:-2, 1:-1], p[2:, 1:-1]
    ne, sw = p[:-2, 2:], p[2:, :-2]
    nw, se = p[:-2, :-2], p[2:, 2:]
    keep = np.zeros_like(mag, dtype=bool)
    b0 = (ang < 22.5) | (ang >= 157.5)
    b45 = (ang >= 22.5) & (ang < 67.5)
    b90 = (ang >= 67.5) & (ang < 112.5)
    b135 = (ang >= 112.5) & (ang < 157.5)
    keep |= b0 & (c >= e) & (c >= w)
    keep |= b90 & (c >= n) & (c >= s)
    # row index grows southward when lat is ascending? lat ascending -> row 0 is south.
    # gradient direction is in index space, so neighbours are taken in index space too.
    keep |= b45 & (c >= se) & (c >= nw)
    keep |= b135 & (c >= ne) & (c >= sw)
    return keep


def hysteresis(mag, thin, low, high) -> np.ndarray:
    weak = thin & (mag >= low)
    lab, n = label(weak, structure=np.ones((3, 3), bool))
    if n == 0:
        return weak
    strong_ids = np.unique(lab[thin & (mag >= high)])
    strong_ids = strong_ids[strong_ids > 0]
    return np.isin(lab, strong_ids)


def detect_day(values, lat, lon, spec, low, high, err=None, err_max=None):
    """New detector: returns (front mask, gradient magnitude, valid mask)."""
    f, gaps = prepare(values, spec["transform"])
    if gaps.all():
        z = np.zeros(values.shape, bool)
        return z, np.full(values.shape, np.nan, "float32"), ~gaps
    mag, gx, gy = metric_gradient(f, lat, lon, spec["sigma_km"])
    allowed = ~gaps
    if err is not None and err_max is not None:
        allowed &= ~(err > err_max)
    fronts = hysteresis(mag, nms(mag, gx, gy) & allowed, low, high)
    mag = np.where(gaps, np.nan, mag).astype("float32")
    return fronts, mag, ~gaps


def boa_day(values, lat, lon, threshold):
    """Baseline: the pipeline's BOA (post-#239) as a pixel mask."""
    from h2mare.processing.core.fronts import boa

    pts = boa(lon, lat, values.astype("float64"), threshold)
    m = np.zeros(values.shape, bool)
    if len(pts):
        m[np.searchsorted(lat, pts[:, 0]), np.searchsorted(lon, pts[:, 1])] = True
    return m


def read_values(da, tries: int = 6):
    """
    Load a DataArray's values, retrying OSError with backoff.

    The store drive intermittently fails reads under heavy concurrent access
    (OSError: [Errno 22] Invalid argument); the same read succeeds moments later.
    """
    import time

    for i in range(tries):
        try:
            return da.values
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
