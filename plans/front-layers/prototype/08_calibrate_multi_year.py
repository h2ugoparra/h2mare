"""
Gradient percentiles per year (2004, 2014, 2024) and per sigma, and analysis_error
percentiles per year. Reads each sample date once and derives every sigma from it.
"""

import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import xarray as xr

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from detect import SPEC, metric_gradient, prepare, read_values, store_file  # noqa: E402

YEARS = (2004, 2014, 2024)
SIGMA_FACTORS = (0.5, 1.0, 1.5)
PCTS = (75, 85, 90, 95)


def _one(args):
    vk, year, date = args
    from h2mare.config import get_settings

    cfg = get_settings().load_app_config()
    spec = SPEC[vk]
    ds = xr.open_zarr(store_file(cfg, vk, year), consolidated=False).sel(time=date)
    lat, lon = ds.lat.values.astype("float64"), ds.lon.values.astype("float64")
    f, gaps = prepare(read_values(ds[vk]), spec["transform"])
    rng = np.random.default_rng(0)
    out = {}
    for fac in SIGMA_FACTORS:
        mag, _, _ = metric_gradient(f, lat, lon, spec["sigma_km"] * fac)
        out[fac] = rng.choice(mag[~gaps], 100_000, replace=False)
    err = None
    if spec["err"]:
        e = read_values(ds[spec["err"]])
        err = rng.choice(e[np.isfinite(e)], 100_000, replace=False)
    return vk, year, out, err


def main():
    jobs = [(vk, y, f"{y}-{m:02d}-{d:02d}") for vk in SPEC for y in YEARS
            for m in range(1, 13) for d in (1, 15)]
    acc: dict = {}
    with mp.get_context("spawn").Pool(6) as pool:
        for vk, year, out, err in pool.imap_unordered(_one, jobs):
            a = acc.setdefault(vk, {}).setdefault(year, {"g": {f: [] for f in SIGMA_FACTORS}, "e": []})
            for fac, v in out.items():
                a["g"][fac].append(v)
            if err is not None:
                a["e"].append(err)
    result = {}
    for vk, years in acc.items():
        result[vk] = {}
        for year, a in sorted(years.items()):
            r = {"grad": {}, "err": None}
            for fac in SIGMA_FACTORS:
                g = np.concatenate(a["g"][fac])
                r["grad"][str(fac)] = {str(q): float(np.percentile(g, q)) for q in PCTS}
            if a["e"]:
                e = np.concatenate(a["e"])
                r["err"] = {str(q): float(np.percentile(e, q)) for q in (90, 95, 99)}
            result[vk][str(year)] = r
    (HERE / "calibration_multi.json").write_text(json.dumps(result, indent=1))
    for vk, years in result.items():
        print(f"\n== {vk} (sigma baseline {SPEC[vk]['sigma_km']} km) ==")
        for year, r in years.items():
            for fac, p in r["grad"].items():
                print(f"  {year} sigma x{fac}: " + "  ".join(f"p{q}={v:.4g}" for q, v in p.items()))
            if r["err"]:
                print(f"  {year} analysis_error: " + "  ".join(f"p{q}={v:.3g}" for q, v in r["err"].items()))


if __name__ == "__main__":
    main()
