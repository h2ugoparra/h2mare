import pathlib, numpy as np, xarray as xr
from scipy.ndimage import sobel, distance_transform_edt
from h2mare.config import get_settings
from h2mare.utils.paths import store_root_for
from h2mare.processing.core.fronts import filt5, filt3
cfg = get_settings().load_app_config()
DAYS = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
R = 111.2  # km per degree

def km_gradient(g, lat, lon):
    """BOA-preprocessed field, Sobel per axis scaled by the true cell size -> units/km."""
    inv = np.isnan(g)
    idx = distance_transform_edt(inv, return_distances=False, return_indices=True)
    f = g[tuple(idx)]
    f = filt3(f, filt5(f))
    dy = abs(lat[1]-lat[0]) * R
    dx = abs(lon[1]-lon[0]) * R * np.cos(np.deg2rad(lat))[:, None]
    gx = sobel(f, axis=1) / (8 * dx); gy = sobel(f, axis=0) / (8 * dy)
    gi = np.hypot(sobel(f, axis=1), sobel(f, axis=0)) / (8 * dy)   # what BOA does today: index space
    return np.hypot(gx, gy), gi, ~inv

for vk, transform, unit in (("sst", None, "degC/km"), ("chl", None, "mg/m3/km"), ("chl", np.log10, "log10(mg/m3)/km")):
    v = cfg.variables[vk]
    f = sorted(pathlib.Path(store_root_for(v) / v.local_folder).glob("*2024.zarr"))[0]
    ds = xr.open_zarr(f, consolidated=False)[vk]
    pct = []; ratio = []
    for d in DAYS:
        day = ds.sel(time=d).load(); g = day.values.astype("float64")
        if transform is not None:
            g = transform(np.where(g > 0, g, np.nan))
        gk, gi, ok = km_gradient(g, day.lat.values, day.lon.values)
        pct.append(np.percentile(gk[ok], [50, 75, 90, 95, 99]))
        lat2d = np.broadcast_to(day.lat.values[:, None], g.shape)
        hi = ok & (lat2d > 55); lo = ok & (lat2d < 20)
        ratio.append((np.median(gi[hi] / gk[hi]), np.median(gi[lo] / gk[lo])))
    p = np.mean(pct, axis=0); r = np.mean(ratio, axis=0)
    print(f"{vk}{' (log10)' if transform else ''}: |grad| percentiles in {unit}: p50 {p[0]:.4f}  p75 {p[1]:.4f}  p90 {p[2]:.4f}  p95 {p[3]:.4f}  p99 {p[4]:.4f}"
          f"   | index-space/true ratio: >55N {r[0]:.2f}, <20N {r[1]:.2f}")
