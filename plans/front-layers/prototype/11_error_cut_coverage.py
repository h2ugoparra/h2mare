"""
What the sst confidence cut costs in coverage (plan §4.3).

A pixel whose ``analysis_error`` exceeds the cut is not assessed that day, and
the 30-day frequency is NaN where fewer than half the window's days were
assessed. This reports, per candidate cut and for a summer and a winter 30-day
window in 2004 / 2014 / 2024:

- the share of sea pixels whose frequency would be NaN, over the whole domain
  and over a Gulf Stream / NAC box (35-55N, 60-30W);
- the median daily share of sea pixels masked;
- how much of what stays masked lies north of 55N (ice margins, persistent
  high-latitude cloud);
- how blocky the error field is (share of east-west neighbours with the
  identical value), which is why the holes are squares.

    uv run python plans/front-layers/prototype/11_error_cut_coverage.py
"""

import json
import os

os.environ.setdefault("H2MARE_ROOT", r"C:\Users\h2ugo\h2mare-run")

import numpy as np
import pandas as pd

from h2mare.processing.core.front_layers import read_retrying
from h2mare.storage.zarr_catalog import ZarrCatalog

CUTS = [0.84, 1.0, 1.2, 1.52]
WINDOWS = {"summer": "07-15", "winter": "02-14"}
YEARS = [2004, 2014, 2024]
GS = dict(lat=slice(35, 55), lon=slice(-60, -30))


def main() -> dict:
    cat = ZarrCatalog("sst")
    out = {}
    for year in YEARS:
        for season, md in WINDOWS.items():
            end = pd.Timestamp(f"{year}-{md}")
            start = end - pd.Timedelta(days=29)
            ds = cat.open_dataset(
                start_date=start, end_date=end, variables=["analysis_error"]
            )
            e = read_retrying(ds["analysis_error"].load, f"{year} {season}")
            sea = np.isfinite(e).all("time")
            gs, gsea = e.sel(**GS), sea.sel(**GS)
            n_days = e.sizes["time"]
            row = {"days": n_days, "cuts": {}}
            for cut in CUTS:
                over = (e > cut).sum("time")
                nan = (over > n_days // 2) & sea
                gnan = ((gs > cut).sum("time") > n_days // 2) & gsea
                daily = ((e > cut) & sea).sum(("lat", "lon")) / sea.sum()
                masked = (e > cut) & sea
                north = masked.sel(lat=slice(55, None)).sum() / max(int(masked.sum()), 1)
                row["cuts"][str(cut)] = {
                    "freq_nan_domain": round(float(nan.sum() / sea.sum()), 4),
                    "freq_nan_gulf_stream": round(float(gnan.sum() / gsea.sum()), 4),
                    "daily_masked_median": round(float(daily.median()), 4),
                    "masked_north_of_55N": round(float(north), 3),
                }
            last = e.isel(time=-1).values
            same = [
                np.mean(np.diff(r[np.isfinite(r)]) == 0)
                for r in last[:: max(1, last.shape[0] // 20)]
                if np.isfinite(r).sum() > 10
            ]
            row["identical_lon_neighbours"] = round(float(np.mean(same)), 3)
            out[f"{year} {season}"] = row
            print(f"{year} {season}: done", flush=True)
    return out


if __name__ == "__main__":
    res = main()
    path = os.path.join(os.path.dirname(__file__), "error_cut_coverage.json")
    json.dump(res, open(path, "w"), indent=1)
    for k, r in res.items():
        print(f"\n{k} (identical neighbours {r['identical_lon_neighbours']:.0%})")
        for cut, c in r["cuts"].items():
            print(
                f"  {cut} K: freq NaN domain {c['freq_nan_domain']:.1%}, "
                f"Gulf Stream {c['freq_nan_gulf_stream']:.1%}, daily masked "
                f"{c['daily_masked_median']:.1%}, masked N of 55N "
                f"{c['masked_north_of_55N']:.0%}"
            )
