"""
Stage 1: detect fronts for every day of 2024 (with a 30-day warm-up from
December 2023), new method and current BOA, and build candidate SDM layers on
the 0.25 deg h2ds grid. Read-only against the stores; writes to ./out.

Usage: python run_year.py sst|chl
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from detect import SPEC, R_KM, boa_day, detect_day, store_file  # noqa: E402

OUT = HERE / "out"
YEAR = 2024
WARMUP = 30
BOA_THRESHOLD = {"sst": 0.4, "chl": 0.06}
BLOCK = {"sst": 5, "chl": 6}  # native cells per 0.25 deg cell
PERSIST_MIN = 0.3  # a pixel is a persistent front if present (+-1 px) on >= 30% of 30 days
STABLE_PX = 2  # stability: day-d front pixel has a day-(d+1) front within 2 px

_W: dict = {}


def _init(vk, lat, lon, low, high, err_max):
    from h2mare.utils.spatial import to_unit_sphere

    b = BLOCK[vk]
    clat = lat.reshape(-1, b).mean(1)
    clon = lon.reshape(-1, b).mean(1)
    _W.update(vk=vk, lat=lat, lon=lon, low=low, high=high, err_max=err_max, b=b,
              clat=clat, clon=clon)
    lat2, lon2 = np.meshgrid(clat, clon, indexing="ij")
    _W["cells_xyz"] = to_unit_sphere(lat2.ravel(), lon2.ravel())


def _blocks(a, b, fn):
    ny, nx = a.shape
    return fn(a.reshape(ny // b, b, nx // b, b), axis=(1, 3))


def _dist_to(mask):
    """Distance (km) from every 0.25 cell centre to the nearest True pixel."""
    from h2mare.utils.spatial import nearest_on_sphere

    iy, ix = np.nonzero(mask)
    n = len(_W["clat"]) * len(_W["clon"])
    if len(iy) == 0:
        return np.full(n, np.nan, "float32")
    d, _ = nearest_on_sphere(_W["cells_xyz"], _W["lat"][iy], _W["lon"][ix])
    return d.astype("float32")


def _frag(mask):
    lab, n = label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return dict(n=0, px=0, small_share=np.nan, median_len=np.nan)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(n=int(n), px=int(sizes.sum()),
                small_share=float(sizes[sizes < 5].sum() / sizes.sum()),
                median_len=float(np.median(sizes)))


def _day(args):
    values, err = args
    vk, b = _W["vk"], _W["b"]
    fronts, mag, valid = detect_day(values, _W["lat"], _W["lon"], SPEC[vk],
                                    _W["low"], _W["high"], err, _W["err_max"])
    boa = boa_day(values if vk == "sst" else values, _W["lat"], _W["lon"], BOA_THRESHOLD[vk]) \
        if not np.isnan(values).all() else np.zeros(values.shape, bool)
    boa &= valid
    grad = _blocks(np.where(valid, mag, 0.0), b, np.sum) / np.maximum(_blocks(valid, b, np.sum), 1)
    sea = _blocks(valid, b, np.any)
    grad = np.where(sea, grad, np.nan).astype("float32")
    shape = sea.shape
    # Assessed = could have held a front: valid, and not masked for low confidence.
    # A masked day is "not observed", never "no front".
    assessed = valid & ~(err > _W["err_max"]) if err is not None and _W["err_max"] else valid
    return dict(
        fronts=np.packbits(fronts), boa=np.packbits(boa), shape=values.shape,
        assessed=np.packbits(assessed),
        grad=grad, sea=sea,
        front_any=_blocks(fronts, b, np.any), front_frac=_blocks(fronts, b, np.mean).astype("float32"),
        assessed_cell=_blocks(assessed, b, np.mean) >= 0.5,
        fdist=np.where(sea, _dist_to(fronts).reshape(shape), np.nan).astype("float32"),
        fdist_boa=np.where(sea, _dist_to(boa).reshape(shape), np.nan).astype("float32"),
        frag_new=_frag(fronts), frag_boa=_frag(boa),
        masked_share=float(((err > _W["err_max"]) & valid).sum() / valid.sum()) if err is not None and valid.any() else 0.0,
    )


def _stable(prev, cur):
    if prev is None or prev.sum() == 0 or cur.sum() == 0:
        return np.nan
    dist = distance_transform_edt(~cur)
    return float((dist[prev] <= STABLE_PX).mean())


def _nan_gauss(a, sigma):
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0.0), sigma, mode="constant")
    den = gaussian_filter(ok.astype(float), sigma, mode="constant")
    return np.where(ok, num / np.where(den > 0.05, den, np.nan), np.nan).astype("float32")


def main(vk: str):
    from h2mare.config import get_settings

    cfg = get_settings().load_app_config()
    cal = json.loads((HERE / "calibration.json").read_text())[vk]
    low, high = cal["grad_pct"]["75"], cal["grad_pct"]["90"]
    err_max = cal.get("err_pct", {}).get("95")
    spec = SPEC[vk]
    OUT.mkdir(exist_ok=True)

    # 128-day blocks aligned to the store's time chunks; warm-up from the previous year's last block.
    plan = []
    prev = xr.open_zarr(store_file(cfg, vk, YEAR - 1), consolidated=False)
    n_prev = prev.sizes["time"]
    plan.append((YEAR - 1, slice(n_prev - WARMUP, n_prev)))
    n_cur = xr.open_zarr(store_file(cfg, vk, YEAR), consolidated=False).sizes["time"]
    for s in range(0, n_cur, 128):
        plan.append((YEAR, slice(s, min(s + 128, n_cur))))

    ds0 = xr.open_zarr(store_file(cfg, vk, YEAR), consolidated=False)
    lat, lon = ds0.lat.values.astype("float64"), ds0.lon.values.astype("float64")
    b = BLOCK[vk]
    assert len(lat) % b == 0 and len(lon) % b == 0

    times, layers = [], {k: [] for k in ("grad", "front_frac", "fdist", "fdist_boa", "field")}
    front_any_hist, assessed_hist = [], []
    window = deque(maxlen=30)
    win_sum = None
    fdist_persist, stats = [], []
    prev_new = prev_boa = None
    t0 = time.perf_counter()

    with mp.get_context("spawn").Pool(12, initializer=_init, initargs=(vk, lat, lon, low, high, err_max)) as pool:
        for year, sl in plan:
            ds = xr.open_zarr(store_file(cfg, vk, year), consolidated=False).isel(time=sl)
            vals = ds[vk].values
            errs = ds[spec["err"]].values if spec["err"] else [None] * len(vals)
            tt = pd.DatetimeIndex(ds.time.values)
            for t, res in zip(tt, pool.imap(_day, zip(vals, errs), chunksize=2)):
                new = np.unpackbits(res["fronts"])[: np.prod(res["shape"])].reshape(res["shape"]).astype(bool)
                boa = np.unpackbits(res["boa"])[: np.prod(res["shape"])].reshape(res["shape"]).astype(bool)
                seen = np.unpackbits(res["assessed"])[: np.prod(res["shape"])].reshape(res["shape"]).astype(bool)
                dil = binary_dilation(new, np.ones((3, 3), bool))
                if win_sum is None:
                    win_sum = np.zeros(new.shape, np.int16)
                    seen_sum = np.zeros(new.shape, np.int16)
                if len(window) == window.maxlen:
                    win_sum -= window[0][0]
                    seen_sum -= window[0][1]
                window.append((dil.astype(np.int16), seen.astype(np.int16)))
                win_sum += window[-1][0]
                seen_sum += window[-1][1]
                front_any_hist.append(res["front_any"])
                assessed_hist.append(res["assessed_cell"])
                in_year = t.year == YEAR
                if in_year:
                    times.append(t)
                    for k in ("grad", "front_frac", "fdist", "fdist_boa"):
                        layers[k].append(res[k])
                    # frequency over the days each pixel was actually assessed; a pixel
                    # seen on fewer than half the window cannot be called persistent
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pfreq = win_sum / seen_sum
                    persistent = (pfreq >= PERSIST_MIN) & (seen_sum >= len(window) / 2)
                    from h2mare.utils.spatial import nearest_on_sphere  # noqa
                    iy, ix = np.nonzero(persistent)
                    fp = np.full(res["sea"].size, np.nan, "float32")
                    if len(iy):
                        _init(vk, lat, lon, low, high, err_max) if not _W else None
                        d, _ = nearest_on_sphere(_W["cells_xyz"], lat[iy], lon[ix])
                        fp = d.astype("float32")
                    fdist_persist.append(np.where(res["sea"], fp.reshape(res["sea"].shape), np.nan))
                    stats.append(dict(date=str(t.date()), **{f"new_{k}": v for k, v in res["frag_new"].items()},
                                      **{f"boa_{k}": v for k, v in res["frag_boa"].items()},
                                      stable_new=_stable(prev_new, new), stable_boa=_stable(prev_boa, boa),
                                      masked_share=res["masked_share"]))
                prev_new, prev_boa = new, boa
            print(f"[{vk}] {year} days {sl.start}-{sl.stop} done ({time.perf_counter() - t0:.0f}s)", flush=True)

    # field itself on the 0.25 grid (block mean; chl as log10), read in chunk-aligned blocks
    fields = []
    for s in range(0, n_cur, 128):
        v = xr.open_zarr(store_file(cfg, vk, YEAR), consolidated=False)[vk].isel(time=slice(s, min(s + 128, n_cur))).values.astype("float64")
        if spec["transform"] == "log10":
            v = np.log10(np.where(v > 0, v, np.nan))
        ny, nx = v.shape[1:]
        fields.append(np.nanmean(v.reshape(-1, ny // b, b, nx // b, b), axis=(2, 4)).astype("float32"))
    field = np.concatenate(fields)

    fa = np.array(front_any_hist, dtype="float32")  # includes warm-up
    seen = np.array(assessed_hist, dtype="float32")

    def _freq(w):
        out = []
        for i in range(WARMUP, len(fa)):
            n = seen[i - w + 1:i + 1].sum(0)
            f = (fa[i - w + 1:i + 1] * seen[i - w + 1:i + 1]).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.append(np.where(n >= w / 2, f / n, np.nan))
        return np.stack(out)

    freq7, freq30 = _freq(7), _freq(30)
    grad = np.stack(layers["grad"])
    cell_km = 0.25 * R_KM
    act = {f"grad_nbhd{r}": np.stack([_nan_gauss(g, r / cell_km) for g in grad]) for r in (50, 100, 200)}

    clat = lat.reshape(-1, b).mean(1)
    clon = lon.reshape(-1, b).mean(1)
    sea = np.isfinite(grad)
    data = {
        f"{vk}_field": field,
        "grad": grad,
        **act,
        "front_frac": np.stack(layers["front_frac"]),
        "freq7": np.where(sea, freq7, np.nan),
        "freq30": np.where(sea, freq30, np.nan),
        "fdist": np.stack(layers["fdist"]),
        "fdist_persist": np.stack(fdist_persist),
        "fdist_boa": np.stack(layers["fdist_boa"]),
    }
    ds_out = xr.Dataset({k: (("time", "lat", "lon"), v.astype("float32")) for k, v in data.items()},
                        coords={"time": times, "lat": clat, "lon": clon})
    ds_out.attrs.update(low=low, high=high, err_max=err_max or -1, sigma_km=spec["sigma_km"])
    ds_out.to_zarr(OUT / f"{vk}_layers_{YEAR}.zarr", mode="w", consolidated=False)
    pd.DataFrame(stats).to_csv(OUT / f"{vk}_stats_{YEAR}.csv", index=False)
    print(f"[{vk}] saved layers and stats ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
