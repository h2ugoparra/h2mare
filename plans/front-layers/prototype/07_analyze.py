"""Stage 2: statistics, correlations/VIF and maps from the stage-1 outputs."""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from detect import SPEC, boa_day, detect_day, store_file  # noqa: E402

OUT = HERE / "out"
LAYERS = ["field", "lat", "grad", "grad_nbhd50", "grad_nbhd100", "grad_nbhd200",
          "front_frac", "freq7", "freq30", "fdist", "fdist_persist", "fdist_boa"]
PROPOSED = ["field", "lat", "grad", "grad_nbhd100", "freq30", "fdist_persist"]


def vif(X: np.ndarray) -> np.ndarray:
    Z = (X - X.mean(0)) / X.std(0)
    out = []
    for j in range(Z.shape[1]):
        y = Z[:, j]
        A = np.delete(Z, j, axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ coef) ** 2).sum() / (y ** 2).sum()
        out.append(1 / max(1 - r2, 1e-9))
    return np.array(out)


def table(vk: str, report: dict) -> None:
    ds = xr.open_zarr(OUT / f"{vk}_layers_2024.zarr", consolidated=False).load()
    ds = ds.rename({f"{vk}_field": "field"})
    ds["lat"] = ds.lat.broadcast_like(ds["grad"])
    flat = {k: ds[k].values.ravel() for k in LAYERS}
    ok = np.all([np.isfinite(v) for v in flat.values()], axis=0)
    idx = np.random.default_rng(1).choice(np.flatnonzero(ok), 60_000, replace=False)
    df = pd.DataFrame({k: v[idx] for k, v in flat.items()})
    ranks = df.apply(rankdata)
    corr = ranks.corr()
    report[vk]["spearman"] = corr.round(2).to_dict()
    report[vk]["vif_all"] = dict(zip(LAYERS, vif(ranks.values).round(1)))
    report[vk]["vif_proposed"] = dict(zip(PROPOSED, vif(ranks[PROPOSED].values).round(1)))
    q = [10, 25, 50, 75, 90]
    report[vk]["fdist_pct_km"] = {k: dict(zip(q, np.nanpercentile(ds[k].values, q).round(1)))
                                 for k in ("fdist", "fdist_persist", "fdist_boa")}
    print(f"\n== {vk}: Spearman correlations (sample of 60k sea cell-days) ==")
    print(corr.round(2).to_string())
    print(f"\nVIF, all layers: {report[vk]['vif_all']}")
    print(f"VIF, proposed set: {report[vk]['vif_proposed']}")
    print(f"fdist percentiles (km): {report[vk]['fdist_pct_km']}")


def stats(vk: str, report: dict) -> None:
    s = pd.read_csv(OUT / f"{vk}_stats_2024.csv", parse_dates=["date"])
    summ = {}
    for m in ("new", "boa"):
        summ[m] = dict(front_px_median=int(s[f"{m}_px"].median()),
                       components_median=int(s[f"{m}_n"].median()),
                       median_component_px=float(s[f"{m}_median_len"].median()),
                       small_component_px_share=round(float(s[f"{m}_small_share"].median()), 3),
                       next_day_within_2px=round(float(s[f"stable_{m}"].median()), 3))
    summ["masked_share_median"] = round(float(s["masked_share"].median()), 3)
    month = s.groupby(s.date.dt.month)[["new_px", "boa_px"]].median().astype(int)
    report[vk]["summary"] = summ
    report[vk]["front_px_by_month"] = month.to_dict()
    print(f"\n== {vk}: fragmentation / stability (daily medians, 2024) ==")
    print(json.dumps(summ, indent=1))
    print(month.T.to_string())


def maps(vk: str, day: str) -> None:
    from h2mare.config import get_settings

    cfg = get_settings().load_app_config()
    cal = json.loads((HERE / "calibration.json").read_text())[vk]
    spec = SPEC[vk]
    src = xr.open_zarr(store_file(cfg, vk, 2024), consolidated=False).sel(time=day)
    lat, lon = src.lat.values.astype("float64"), src.lon.values.astype("float64")
    vals = src[vk].values
    err = src[spec["err"]].values if spec["err"] else None
    new, _, valid = detect_day(vals, lat, lon, spec, cal["grad_pct"]["75"], cal["grad_pct"]["90"],
                               err, cal.get("err_pct", {}).get("95"))
    boa = boa_day(vals, lat, lon, 0.4 if vk == "sst" else 0.06) & valid
    field = np.log10(np.where(vals > 0, vals, np.nan)) if vk == "chl" else vals
    lay = xr.open_zarr(OUT / f"{vk}_layers_2024.zarr", consolidated=False).sel(time=day)

    ext = [lon[0], lon[-1], lat[0], lat[-1]]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10.5))
    cmap = "RdYlBu_r" if vk == "sst" else "viridis"
    for a, mask, title in ((ax[0, 0], boa, f"current BOA ({boa.mean()*100:.0f}% of pixels)"),
                           (ax[0, 1], new, f"new detector ({new.mean()*100:.1f}% of pixels)")):
        a.imshow(field, origin="lower", extent=ext, cmap=cmap)
        m = np.ma.masked_where(~mask, mask)
        a.imshow(m, origin="lower", extent=ext, cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(f"{vk} {day}: {title}")
    panels = [(ax[0, 2], "grad", "gradient (per km), 0.25 deg", "magma"),
              (ax[1, 0], "freq30", "front frequency, last 30 days", "magma"),
              (ax[1, 1], "fdist_persist", "distance to persistent front (km)", "viridis_r"),
              (ax[1, 2], "fdist_boa", "current fdist (BOA, km)", "viridis_r")]
    for a, k, title, cm in panels:
        v = lay[k].values
        vmax = np.nanpercentile(v, 98)
        im = a.imshow(v, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=vmax)
        a.set_title(title)
        fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / f"map_{vk}_{day}.png", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    report = {}
    for vk in ("sst", "chl"):
        report[vk] = {}
        stats(vk, report)
        table(vk, report)
        for day in ("2024-01-15", "2024-07-15"):
            maps(vk, day)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("\nmaps and report written")
