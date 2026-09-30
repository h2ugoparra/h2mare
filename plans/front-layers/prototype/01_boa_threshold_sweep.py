import pathlib, numpy as np, xarray as xr
from h2mare.config import get_settings
from h2mare.utils.paths import store_root_for
from h2mare.processing.core.fronts import boa, create_base_grid
from h2mare.utils.spatial import haversine_min_distance_kdtree
cfg = get_settings().load_app_config()
DAYS = ["2024-01-15", "2024-04-15", "2024-07-15", "2024-10-15"]
SPEC = {"sst": ([0.2, 0.4, 0.8, 1.2, 1.6, 2.4], 0.05 * 111.2),     # threshold list, km per cell (lat)
        "chl": ([0.015, 0.03, 0.06, 0.12, 0.24, 0.5], 111.2 / 24)}
for vk, (thrs, km) in SPEC.items():
    v = cfg.variables[vk]
    f = sorted(pathlib.Path(store_root_for(v) / v.local_folder).glob("*2024.zarr"))[0]
    ds = xr.open_zarr(f, consolidated=False)[vk]
    rows = {t: [] for t in thrs}
    for d in DAYS:
        day = ds.sel(time=d).load(); g = day.values; lat = day.lat.values; lon = day.lon.values
        pts, sea = create_base_grid(lat, lon)
        valid = ~np.isnan(g); cells = sea & valid
        pts_valid = np.column_stack(np.nonzero(cells))   # indices
        q = np.column_stack((lat[pts_valid[:, 0]], lon[pts_valid[:, 1]]))
        for t in thrs:
            fr = boa(lon, lat, g, t)
            cov = 100 * len(fr) / cells.sum()
            dist = haversine_min_distance_kdtree(q, fr) if len(fr) else np.full(len(q), np.nan)
            rows[t].append((cov, np.nanmedian(dist), np.nanpercentile(dist, 90)))
    unit = "degC/km" if vk == "sst" else "mg/m3/km"
    print(f"\n{vk}  (threshold -> per-km gradient = thr / (8 x {km:.2f} km); mean over {len(DAYS)} days)")
    print(f"{'thr':>7} {unit:>10} {'front %':>8} {'fdist p50':>10} {'fdist p90':>10}   per-day front %")
    for t in thrs:
        a = np.array(rows[t])
        print(f"{t:>7g} {t/(8*km):>10.4f} {a[:,0].mean():>8.1f} {a[:,1].mean():>8.1f}km {a[:,2].mean():>8.1f}km   " + " ".join(f"{x:4.0f}" for x in a[:,0]))
