"""
Persistence threshold for the distance to persistent fronts (plan §4.4).

Detects fronts over two 45-day windows of 2024 with the configured
front_layers, then, on two days per window with a full 30-day history, reports
the zone share, distance percentiles and Spearman with ffreq30 for each
candidate min_frequency, and the 1% binomial chance-recurrence cut.

    uv run python plans/front-layers/prototype/10_persistent_distance.py sst
"""

import json
import os
import sys
import time

os.environ.setdefault("H2MARE_ROOT", r"C:\Users\h2ugo\h2mare-run")

import numpy as np
import pandas as pd
from scipy.stats import binom, spearmanr

from h2mare import get_settings
from h2mare.processing.core.front_layers import apply_front_layers, persistent_distance
from h2mare.processing.core.fronts import clear_staging
from h2mare.storage.zarr_catalog import ZarrCatalog

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
PERIODS = {"summer": ("2024-06-01", "2024-07-15"), "winter": ("2024-01-01", "2024-02-14")}


def q(a, ps=(10, 25, 50, 75, 90)):
    a = a[np.isfinite(a)]
    return [round(float(np.percentile(a, p)), 1) for p in ps] if a.size else None


def main(var_key):
    spec = get_settings().app_config.variables[var_key].front_layers[var_key]
    needed = [spec.source] + ([spec.confidence.var] if spec.confidence else [])
    cat = ZarrCatalog(var_key)
    results = {}
    for period, (start, end) in PERIODS.items():
        t0 = time.time()
        ds = cat.open_dataset(start_date=start, end_date=end, variables=needed)
        owner = f"measure_{var_key}"
        out = apply_front_layers(ds[needed], {var_key: spec}, owner).load()
        clear_staging(owner)
        lat, lon = out["lat"].values, out["lon"].values
        freq = out[f"{var_key}_ffreq30"]
        days = pd.DatetimeIndex(out.time.values)
        for day in (days[-16], days[-1]):
            f = freq.sel(time=day).values
            # window's proximity masks: the daily chance a pixel has a front nearby
            n_assessed = int(np.isfinite(out[f"{var_key}_front"].sel(time=slice(day - pd.Timedelta(days=29), day)).values).sum(0).max())
            p0 = float(np.nanmean(f))
            crit = {
                n: float(binom.isf(0.01, n, p0) + 1) / n for n in (15, 30)
            }
            row = {
                "p0_mean_freq": round(p0, 3),
                "freq_zero_share": round(float(np.nanmean(f == 0)), 3),
                "freq_pct_50_75_90_95": [round(float(np.nanpercentile(f, p)), 3) for p in (50, 75, 90, 95)],
                "binomial_1pct_threshold_n15_n30": {k: round(v, 3) for k, v in crit.items()},
                "max_assessed": n_assessed,
                "thresholds": {},
            }
            for thr in THRESHOLDS:
                t1 = time.time()
                d = persistent_distance(f, lat, lon, thr)
                ok = np.isfinite(d) & np.isfinite(f)
                rho = spearmanr(d[ok], f[ok]).statistic if ok.any() else None
                row["thresholds"][thr] = {
                    "zone_share": round(float(np.nanmean(f >= thr)), 3),
                    "pdist_p10_25_50_75_90": q(d),
                    "spearman_with_ffreq30": round(float(rho), 3),
                    "secs": round(time.time() - t1, 1),
                }
            results[f"{period} {day:%Y-%m-%d}"] = row
        print(f"{var_key} {period}: {time.time() - t0:.0f}s", flush=True)
    return results


if __name__ == "__main__":
    var_key = sys.argv[1]
    res = main(var_key)
    path = os.path.join(os.path.dirname(__file__), f"persistent_distance_{var_key}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res, indent=1))
