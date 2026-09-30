"""
scripts/backfill_front_layers.py: a store backfilled file by file must hold
what one continuous convert would have written (plans/front-layers.md §6.6).
"""

import importlib.util
import shutil
from pathlib import Path

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import FrontLayerSpec
from h2mare.processing.core.front_layers import apply_front_layers

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backfill_front_layers.py"

LAT = np.arange(40.0, 42.0, 0.05) + 0.025
LON = np.arange(-20.0, -18.0, 0.05) + 0.025
SPEC = msgspec.convert(
    {
        "source": "sst",
        "sigma_km": 5.0,
        "low_per_km": 0.0155,
        "high_per_km": 0.0299,
        "frequency_days": [4],
    },
    FrontLayerSpec,
)
SPECS = {"sst": SPEC}
OUTPUTS = SPEC.output_names("sst")
DAYS = pd.date_range("2020-01-01", "2020-01-18", freq="D")


def _load():
    spec = importlib.util.spec_from_file_location("backfill_front_layers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sst(days: pd.DatetimeIndex, seed=0) -> xr.Dataset:
    """A step that moves one column a day, so each day's mask differs."""
    fields = np.full((len(days), LAT.size, LON.size), 15.0, dtype="float32")
    for i in range(len(days)):
        fields[i, :, 15 + (i + seed) % 10 :] = 18.0
    return xr.Dataset(
        {"sst": (("time", "lat", "lon"), fields)},
        coords={"time": days, "lat": LAT, "lon": LON},
    )


class _Store:
    """Per-period zarr files, with the script's Writer and a MaskReader over them."""

    def __init__(self, root: Path, field: xr.Dataset, periods: list[pd.DatetimeIndex]):
        self.paths = []
        for i, days in enumerate(periods):
            path = root / f"testvar_{i}.zarr"
            field.sel(time=days).to_zarr(path, consolidated=False)
            self.paths.append(path)
        self.spans = pd.DataFrame(
            {"start": [p[0] for p in periods], "end": [p[-1] for p in periods]},
            index=[str(p) for p in self.paths],
        )

    def write(self, ds: xr.Dataset, path: Path) -> None:
        """Incoming wins for its days; everything else in the file is kept."""
        with xr.open_zarr(path, consolidated=False) as stored:
            merged = stored.load()
        ds = ds.load()
        for var in ds.data_vars:
            if var in merged:
                merged[var].loc[dict(time=ds.time)] = ds[var].values
            else:
                merged[var] = ds[var].reindex(time=merged.time)
        tmp = path.with_suffix(".tmp")
        merged.to_zarr(tmp, mode="w", consolidated=False)
        shutil.rmtree(path)
        tmp.rename(path)

    def read(self, var, start, end):
        """Like ZarrReader: a file without *var* is padded with NaN as long as
        another file in the window holds it."""
        parts, held = [], False
        for path in self.paths:
            with xr.open_zarr(path, consolidated=False) as ds:
                part = ds.sel(time=slice(start, end))
                if var in ds:
                    held = True
                    parts.append(part[var].load())
                else:
                    parts.append(xr.full_like(part["sst"], np.nan).load().rename(var))
        parts = [p for p in parts if p.sizes["time"]]
        return xr.concat(parts, dim="time") if held and parts else None

    def open(self) -> xr.Dataset:
        return xr.concat(
            [xr.open_zarr(p, consolidated=False).load() for p in self.paths], dim="time"
        )


@pytest.fixture
def script():
    return _load()


@pytest.fixture
def periods():
    return [DAYS[:6], DAYS[6:12], DAYS[12:]]


@pytest.mark.usefixtures("interim_dir", "serial_pool")
class TestBackfill:
    def test_file_by_file_equals_one_continuous_run(self, script, tmp_path, periods):
        store = _Store(tmp_path, _sst(DAYS), periods)

        script.backfill(
            "testvar", SPECS, store.paths, store.spans, store.read, store.write
        )

        truth = apply_front_layers(_sst(DAYS), SPECS, "testvar").load()
        got = store.open()
        for var in OUTPUTS:
            np.testing.assert_array_equal(got[var].values, truth[var].values, var)

    def test_one_file_again_brings_the_next_files_frequency_up_to_date(
        self, script, tmp_path, periods
    ):
        """The field in the middle file changed (a REP rewrite): re-running that
        file alone must leave the whole store as a from-scratch run would."""
        store = _Store(tmp_path, _sst(DAYS), periods)
        script.backfill(
            "testvar", SPECS, store.paths, store.spans, store.read, store.write
        )
        changed = _sst(periods[1], seed=3)
        store.write(changed, store.paths[1])

        script.backfill(
            "testvar", SPECS, [store.paths[1]], store.spans, store.read, store.write
        )

        field = _sst(DAYS)
        field["sst"].loc[dict(time=periods[1])] = changed["sst"].values
        truth = apply_front_layers(field, SPECS, "testvar").load()
        got = store.open()
        for var in OUTPUTS:
            np.testing.assert_array_equal(got[var].values, truth[var].values, var)

    def test_files_without_layers_after_the_run_are_left_alone(
        self, script, tmp_path, periods
    ):
        """Backfilling the first file must not plant a stray frequency in a later
        one that has no layers yet."""
        store = _Store(tmp_path, _sst(DAYS), periods)

        script.backfill(
            "testvar", SPECS, store.paths[:1], store.spans, store.read, store.write
        )

        with xr.open_zarr(store.paths[1], consolidated=False) as ds:
            assert not set(OUTPUTS) & set(ds.data_vars)

    def test_the_staging_is_cleared(self, script, tmp_path, periods, interim_dir):
        store = _Store(tmp_path, _sst(DAYS), periods)

        script.backfill(
            "testvar", SPECS, store.paths, store.spans, store.read, store.write
        )

        assert list(interim_dir.glob(".testvar_*")) == []


class TestSelect:
    def test_years_pick_the_files_touching_them(self, script):
        spans = pd.DataFrame(
            {
                "start": pd.to_datetime(["2019-01-01", "2020-01-01", "2021-01-01"]),
                "end": pd.to_datetime(["2019-12-31", "2020-12-31", "2021-06-30"]),
            },
            index=["a.zarr", "b.zarr", "c.zarr"],
        )
        assert script.select(spans, [2020, 2021]) == [Path("b.zarr"), Path("c.zarr")]
        assert len(script.select(spans, None)) == 3

    def test_workers_override_every_layer(self, script):
        specs = script.with_workers({"a": SPEC, "b": SPEC}, 3)
        assert {s.n_workers for s in specs.values()} == {3}
        assert script.with_workers(SPECS, None) is SPECS
