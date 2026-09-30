"""
One-at-a-time sensitivity of the SDM layers to the detector parameters.

Four 30-day windows of 2024 (ending Jan 31, Apr 30, Jul 31, Oct 31); each variant is
detected on all 120 days and its layers on the four window-end dates are compared
with the baseline's (Spearman over sea cells, pooled across dates).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from detect import SPEC, detect_day, read_values, store_file  # noqa: E402

BLOCK = {"sst": 5, "chl": 6}
ENDS = ["2024-01-31", "2024-04-30", "2024-07-31", "2024-10-31"]
W = 30
_W: dict = {}


def _init(vk, lat, lon, spec, low, high, err_max):
    _W.update(vk=vk, lat=lat, lon=lon, spec=spec, low=low, high=high, err_max=err_max, b=BLOCK[vk])


def _blocks(a, b, fn):
    ny, nx = a.shape
    return fn(a.reshape(ny // b, b, nx // b, b), axis=(1, 3))


def _day(args):
    values, err = args
    b = _W["b"]
    fronts, mag, valid = detect_day(values, _W["lat"], _W["lon"], _W["spec"], _W["low"], _W["high"],
                                    err, _W["err_max"])
    assessed = valid & ~(err > _W["err_max"]) if err is not None and _W["err_max"] else valid
    sea = _blocks(valid, b, np.any)
    grad = _blocks(np.where(valid, mag, 0.0), b, np.sum) / np.maximum(_blocks(valid, b, np.sum), 1)
    return dict(fronts=np.packbits(fronts), assessed=np.packbits(assessed), shape=values.shape,
                grad=np.where(sea, grad, np.nan).astype("float32"), sea=sea,
                front_any=_blocks(fronts, b, np.any), assessed_cell=_blocks(assessed, b, np.mean) >= 0.5,
                front_frac=_blocks(fronts, b, np.mean).astype("float32"))


def _unpack(p, shape):
    return np.unpackbits(p)[: np.prod(shape)].reshape(shape).astype(bool)


def _dist(cells_xyz, lat, lon, mask):
    from h2mare.utils.spatial import nearest_on_sphere

    iy, ix = np.nonzero(mask)
    if len(iy) == 0:
        return np.full(len(cells_xyz), np.nan)
    return nearest_on_sphere(cells_xyz, lat[iy], lon[ix])[0]


def run_variant(vk, data, lat, lon, spec, low, high, err_max, persist=(0.3,)):
    """Detect on every window day; return layers on the window-end dates."""
    from h2mare.utils.spatial import to_unit_sphere

    b = BLOCK[vk]
    clat, clon = lat.reshape(-1, b).mean(1), lon.reshape(-1, b).mean(1)
    la, lo = np.meshgrid(clat, clon, indexing="ij")
    xyz = to_unit_sphere(la.ravel(), lo.ravel())
    out = {k: [] for k in ["grad", "front_frac", "freq30", "fdist", *[f"fdist_persist{p}" for p in persist]]}
    px = []
    with mp.get_context("spawn").Pool(6, initializer=_init, initargs=(vk, lat, lon, spec, low, high, err_max)) as pool:
        for vals, errs in data:
            res = pool.map(_day, list(zip(vals, errs)), chunksize=2)
            last = res[-1]
            shape2 = last["sea"].shape
            fa = np.array([r["front_any"] for r in res], "float32")
            seen = np.array([r["assessed_cell"] for r in res], "float32")
            n = seen.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                freq = np.where(n >= W / 2, (fa * seen).sum(0) / n, np.nan)
            win = np.zeros(last["shape"], "int16")
            seen_px = np.zeros(last["shape"], "int16")
            for r in res:
                f = _unpack(r["fronts"], r["shape"])
                px.append(f.mean())
                win += binary_dilation(f, np.ones((3, 3), bool))
                seen_px += _unpack(r["assessed"], r["shape"])
            sea = last["sea"]
            out["grad"].append(last["grad"])
            out["front_frac"].append(np.where(sea, last["front_frac"], np.nan))
            out["freq30"].append(np.where(sea, freq, np.nan))
            fronts_last = _unpack(last["fronts"], last["shape"])
            out["fdist"].append(np.where(sea, _dist(xyz, lat, lon, fronts_last).reshape(shape2), np.nan))
            with np.errstate(invalid="ignore", divide="ignore"):
                pf = win / seen_px
            for p in persist:
                pers = (pf >= p) & (seen_px >= W / 2)
                out[f"fdist_persist{p}"].append(np.where(sea, _dist(xyz, lat, lon, pers).reshape(shape2), np.nan))
    return {k: np.stack(v) for k, v in out.items()}, float(np.mean(px))


def compare(base, var):
    res = {}
    for k, v in var.items():
        bk = base.get(k)
        if bk is None:
            continue
        a, c = bk.ravel(), v.ravel()
        ok = np.isfinite(a) & np.isfinite(c)
        res[k] = round(float(spearmanr(a[ok], c[ok])[0]), 3) if ok.sum() > 100 else np.nan
    return res


def main(vk):
    from h2mare.config import get_settings

    cfg = get_settings().load_app_config()
    cal = json.loads((HERE / "calibration_multi.json").read_text())[vk]
    spec0 = SPEC[vk]

    def thr(fac, q):
        return float(np.mean([cal[y]["grad"][str(fac)][str(q)] for y in ("2004", "2014", "2024")]))

    err = {q: cal["2024"]["err"][str(q)] for q in (90, 95, 99)} if spec0["err"] else None
    ds = xr.open_zarr(store_file(cfg, vk, 2024), consolidated=False)
    lat, lon = ds.lat.values.astype("float64"), ds.lon.values.astype("float64")
    data = []
    for end in ENDS:
        sl = ds.sel(time=slice(pd.Timestamp(end) - pd.Timedelta(days=W - 1), end))
        vals = read_values(sl[vk])
        errs = read_values(sl[spec0["err"]]) if spec0["err"] else [None] * len(vals)
        data.append((vals, errs))

    base_low, base_high = thr(1.0, 75), thr(1.0, 90)
    base_err = err[95] if err else None
    variants = {
        "baseline": dict(spec=spec0, low=base_low, high=base_high, err_max=base_err),
        "sigma x0.5": dict(spec={**spec0, "sigma_km": spec0["sigma_km"] * 0.5}, low=thr(0.5, 75), high=thr(0.5, 90), err_max=base_err),
        "sigma x1.5": dict(spec={**spec0, "sigma_km": spec0["sigma_km"] * 1.5}, low=thr(1.5, 75), high=thr(1.5, 90), err_max=base_err),
        "high p85": dict(spec=spec0, low=base_low, high=thr(1.0, 85), err_max=base_err),
        "high p95": dict(spec=spec0, low=base_low, high=thr(1.0, 95), err_max=base_err),
        "ratio 3:1": dict(spec=spec0, low=base_high / 3, high=base_high, err_max=base_err),
    }
    if err:
        variants["err p90"] = dict(spec=spec0, low=base_low, high=base_high, err_max=err[90])
        variants["err p99"] = dict(spec=spec0, low=base_low, high=base_high, err_max=err[99])

    base, base_px = run_variant(vk, data, lat, lon, **variants["baseline"], persist=(0.2, 0.3, 0.5))
    rows = [dict(variant="baseline", front_px=base_px, fdist_persist_med=float(np.nanmedian(base["fdist_persist0.3"])))]
    for p in (0.2, 0.5):
        c = compare({"fdist_persist0.3": base["fdist_persist0.3"]}, {"fdist_persist0.3": base[f"fdist_persist{p}"]})
        rows.append(dict(variant=f"persist {p}", front_px=base_px, fdist_persist_med=float(np.nanmedian(base[f"fdist_persist{p}"])),
                         fdist_persist=c["fdist_persist0.3"]))
        print(rows[-1], flush=True)
    for name, kw in variants.items():
        if name == "baseline":
            continue
        v, px = run_variant(vk, data, lat, lon, **kw)
        c = compare(base, v)
        rows.append(dict(variant=name, front_px=px, fdist_persist_med=float(np.nanmedian(v["fdist_persist0.3"])),
                         **{("fdist_persist" if k == "fdist_persist0.3" else k): x for k, x in c.items()}))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "out" / f"sensitivity_{vk}.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
