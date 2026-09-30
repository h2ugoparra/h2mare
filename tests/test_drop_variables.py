"""scripts/drop_variables.py: only the named arrays go, and the store still opens."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "drop_variables.py"


def _load():
    spec = importlib.util.spec_from_file_location("drop_variables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load()


def _store(path: Path, consolidated: bool = True) -> Path:
    shape = (3, 4, 5)
    xr.Dataset(
        {
            "sst": (("time", "lat", "lon"), np.ones(shape, "float32")),
            "sst_fdist": (("time", "lat", "lon"), np.zeros(shape, "float32")),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=3),
            "lat": np.arange(4.0),
            "lon": np.arange(5.0),
        },
        attrs={"Conventions": "CF-1.10"},
    ).to_zarr(path, consolidated=consolidated)
    return path


@pytest.mark.parametrize("consolidated", [True, False])
def test_only_the_named_array_goes(script, tmp_path, consolidated):
    path = _store(tmp_path / "a.zarr", consolidated)

    script.drop_file(path, ["sst_fdist"], apply=True)

    assert not (path / "sst_fdist").exists()
    with xr.open_zarr(path, consolidated=consolidated) as ds:
        assert list(ds.data_vars) == ["sst"]
        assert ds.attrs["Conventions"] == "CF-1.10"
        np.testing.assert_array_equal(ds["sst"].values, 1)


def test_a_dry_run_changes_nothing(script, tmp_path):
    path = _store(tmp_path / "a.zarr")

    assert script.drop_file(path, ["sst_fdist"], apply=False) > 0

    with xr.open_zarr(path) as ds:
        assert "sst_fdist" in ds.data_vars


def test_a_file_without_the_array_is_left_alone(script, tmp_path):
    path = _store(tmp_path / "a.zarr")
    assert script.drop_file(path, ["chl_fdist"], apply=True) == 0


def test_coordinates_and_produced_names_are_refused(script):
    reasons = script.refused(["lat", "sst", "sst_fdist"], {"sst"})
    assert reasons == ["'lat' is a coordinate", "'sst' is still produced by the config"]
