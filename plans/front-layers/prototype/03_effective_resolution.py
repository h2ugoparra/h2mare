import pathlib, numpy as np, xarray as xr
from scipy.ndimage import distance_transform_edt, gaussian_filter
from h2mare.config import get_settings
from h2mare.utils.paths import store_root_for
cfg = get_settings().load_app_config()
DAYS = [f"2024-{m:02d}-15" for m in range(1, 13)]
BOXES = {  # name: (lat0, lat1, lon0, lon1) open ocean, all within the store bbox
    "subtropical gyre":   (20, 35, -60, -35),
    "Gulf Stream / NAC":  (35, 50, -55, -30),
    "NE Atlantic":        (38, 55, -30, -15),
    "tropical Atlantic":  (4, 18, -45, -25),
}
R = 111.2
OUT = {}

def spectra(block, dx_km, axis):
    """Mean power spectrum of detrended, Hann-windowed lines along *axis*; complete lines only."""
    lines = np.moveaxis(block, axis, -1).reshape(-1, block.shape[axis])
    lines = lines[~np.isnan(lines).any(axis=1)]
    n = lines.shape[1]
    t = np.arange(n)
    A = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, lines.T, rcond=None)
    lines = lines - (A @ coef).T
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(lines * w, axis=1)) ** 2
    k = np.fft.rfftfreq(n, d=dx_km)          # cycles per km
    return k[1:], P[:, 1:].mean(axis=0), len(lines)

def rolloff(k, P, fit=(150, 600)):
    lam = 1 / k
    m = (lam >= fit[0]) & (lam <= fit[1])
    slope, icpt = np.polyfit(np.log10(k[m]), np.log10(P[m]), 1)
    ratio = P / 10 ** (icpt + slope * np.log10(k))
    small = lam < fit[0]
    below = np.where(small & (ratio < 0.5))[0]
    return slope, (lam[below[0]] if below.size else np.nan), ratio

for vk, transform, label in (("sst", None, "sst (degC)"), ("chl", np.log10, "log10 chl")):
    v = cfg.variables[vk]
    f = sorted(pathlib.Path(store_root_for(v) / v.local_folder).glob("*2024.zarr"))[0]
    ds = xr.open_zarr(f, consolidated=False)[vk]
    step = abs(float(ds.lat[1] - ds.lat[0]))
    print(f"\n== {label}: grid {step:.4f} deg = {step*R:.2f} km meridional ==")
    print(f"{'box':20} {'slope(150-600km)':>17} {'meridional lambda_eff':>22} {'zonal lambda_eff':>17} {'lines':>6}")
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        cube = ds.sel(time=DAYS, lat=slice(la0, la1), lon=slice(lo0, lo1)).load().values.astype("float64")
        if transform is not None:
            cube = transform(np.where(cube > 0, cube, np.nan))
        # meridional: along lat axis (axis=1 of time,lat,lon), constant spacing
        km, Pm, nm = spectra(cube.transpose(0, 2, 1).reshape(-1, cube.shape[1])[:, :, None].squeeze(-1)[None], step * R, 2) if False else (None, None, None)
        lines_m = cube.transpose(0, 2, 1).reshape(-1, cube.shape[1])
        km, Pm, nm = spectra(lines_m, step * R, 1)
        lines_z = cube.reshape(-1, cube.shape[2])
        dxz = step * R * np.cos(np.deg2rad((la0 + la1) / 2))
        kz, Pz, nz = spectra(lines_z, dxz, 1)
        sm, lm, rm = rolloff(km, Pm)
        sz, lz, rz = rolloff(kz, Pz)
        OUT[(vk, name)] = (km, Pm, kz, Pz)
        print(f"{name:20} {sm:>17.2f} {lm:>19.0f} km {lz:>14.0f} km {nm:>6}")
    # method 2: gradient variance retained vs smoothing width, whole NE Atlantic + gyre boxes pooled
    print(f"   gradient variance retained after Gaussian smoothing (sigma in cells / km):")
    sig = [1, 2, 3, 4, 6, 8]
    rets = []
    for name, (la0, la1, lo0, lo1) in BOXES.items():
        cube = ds.sel(time=DAYS, lat=slice(la0, la1), lon=slice(lo0, lo1)).load().values.astype("float64")
        if transform is not None:
            cube = transform(np.where(cube > 0, cube, np.nan))
        r = []
        for day in cube:
            nan = np.isnan(day)
            if nan.all():
                continue
            idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
            filled = day[tuple(idx)]
            # score only cells farther from land/gaps than the widest kernel reaches
            far = distance_transform_edt(~nan) > 3 * max(sig)
            if far.sum() < 1000:
                continue
            g0 = np.hypot(*np.gradient(filled))[far]; base = (g0 ** 2).mean()
            r.append([(np.hypot(*np.gradient(gaussian_filter(filled, s)))[far] ** 2).mean() / base for s in sig])
        rets.append(np.mean(r, axis=0))
    rets = np.mean(rets, axis=0)
    print("   " + "  ".join(f"s={s}({s*step*R:.0f}km): {100*x:4.1f}%" for s, x in zip(sig, rets)))

np.save("spectra.npy", {str(k): v for k, v in OUT.items()}, allow_pickle=True)
