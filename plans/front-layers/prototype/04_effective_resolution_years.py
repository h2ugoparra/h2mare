import pathlib, numpy as np, xarray as xr
from h2mare.config import get_settings
from h2mare.utils.paths import store_root_for
cfg = get_settings().load_app_config()
BOXES = {"subtropical gyre": (20, 35, -60, -35), "Gulf Stream / NAC": (35, 50, -55, -30),
         "NE Atlantic": (38, 55, -30, -15), "tropical Atlantic": (4, 18, -45, -25)}
R = 111.2
def lam_half(cube, dx):
    lines = cube.transpose(0, 2, 1).reshape(-1, cube.shape[1]); lines = lines[~np.isnan(lines).any(1)]
    n = lines.shape[1]; t = np.arange(n); A = np.vstack([t, np.ones(n)]).T
    lines = lines - (A @ np.linalg.lstsq(A, lines.T, rcond=None)[0]).T
    P = (np.abs(np.fft.rfft(lines * np.hanning(n), axis=1)) ** 2).mean(0)[1:]; k = np.fft.rfftfreq(n, dx)[1:]
    lam = 1 / k; m = (lam >= 150) & (lam <= 600); s, c = np.polyfit(np.log10(k[m]), np.log10(P[m]), 1)
    r = P / 10 ** (c + s * np.log10(k)); b = np.where((lam < 150) & (r < 0.5))[0]
    small = (lam >= 12) & (lam <= 25)
    return (lam[b[0]] if b.size else np.nan), np.median(r[small])
for vk, tr in (("sst", None), ("chl", np.log10)):
    v = cfg.variables[vk]; root = pathlib.Path(store_root_for(v) / v.local_folder)
    for yr in (2004, 2014, 2024):
        f = sorted(root.glob(f"*{yr}.zarr"))[0]; ds = xr.open_zarr(f, consolidated=False)[vk]
        step = abs(float(ds.lat[1] - ds.lat[0]))
        row = []
        for name, (a, b, c, d) in BOXES.items():
            cube = ds.sel(time=[f"{yr}-{m:02d}-15" for m in range(1, 13)], lat=slice(a, b), lon=slice(c, d)).load().values.astype("float64")
            if tr is not None: cube = tr(np.where(cube > 0, cube, np.nan))
            lh, floor = lam_half(cube, step * R)
            row.append(f"{lh:4.0f}km (x{floor:5.2f})")
        print(f"{vk} {yr}: " + " | ".join(row))
print("columns: " + " | ".join(BOXES) + "   [lambda_half (power/fit at 12-25 km)]")
