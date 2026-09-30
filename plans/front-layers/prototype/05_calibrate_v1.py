"""Stage 0: gradient and analysis_error distributions on sample days -> thresholds."""

import json
import pathlib
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from detect import SPEC, metric_gradient, prepare, store_file  # noqa: E402

from h2mare.config import get_settings  # noqa: E402

OUT = pathlib.Path(__file__).parent
DAYS = [f"2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 15)]


def main():
    cfg = get_settings().load_app_config()
    result = {}
    for vk, spec in SPEC.items():
        ds = xr.open_zarr(store_file(cfg, vk, 2024), consolidated=False)
        lat, lon = ds.lat.values, ds.lon.values
        mags, errs = [], []
        for d in DAYS:
            day = ds.sel(time=d)
            vals = day[vk].values
            f, gaps = prepare(vals, spec["transform"])
            mag, _, _ = metric_gradient(f, lat, lon, spec["sigma_km"])
            sample = mag[~gaps]
            mags.append(np.random.default_rng(0).choice(sample, 200_000, replace=False))
            if spec["err"]:
                e = day[spec["err"]].values
                errs.append(e[np.isfinite(e)])
        m = np.concatenate(mags)
        p = {q: float(np.percentile(m, q)) for q in (50, 75, 90, 95, 99)}
        result[vk] = {"grad_pct": p}
        line = "  ".join(f"p{q}={v:.4g}" for q, v in p.items())
        print(f"{vk}: smoothed |grad| per km  {line}")
        if errs:
            e = np.concatenate(errs)
            ep = {q: float(np.percentile(e, q)) for q in (50, 90, 95, 99)}
            result[vk]["err_pct"] = ep
            print(f"{vk}: analysis_error  " + "  ".join(f"p{q}={v:.3g}" for q, v in ep.items()))
    (OUT / "calibration.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
