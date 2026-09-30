"""
``h2mare parquet2zarr`` — rebuild per-period Zarr files from a Parquet store.

The conversion itself is covered by ``test_parquet2zarr_engine.py``; these pin
the command: its argument checks, what it forwards, and one real round trip
through the CLI, since rebuilding a store is what this command is for.
"""

from datetime import date
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from conftest import make_grid_df
from typer.testing import CliRunner

from h2mare.cli.parquet2zarr import app
from h2mare.storage.parquet_indexer import ParquetIndexer

_runner = CliRunner()
_CONVERT = "h2mare.format_converters.parquet2zarr.convert_parquet_to_zarr"


@pytest.mark.parametrize(
    "args, message",
    [
        (["--start-date", "2021-01-01"], "must be provided together"),
        (["--end-date", "2021-01-01"], "must be provided together"),
        (
            ["--start-date", "2021-02-01", "--end-date", "2021-01-01"],
            "must not be after",
        ),
        (["--date-format", "week"], "--date-format must be"),
        (["--layout", "tiles"], "--layout must be"),
    ],
)
def test_bad_arguments_exit_1_before_converting(tmp_path, args, message):
    with patch(_CONVERT) as convert:
        result = _runner.invoke(
            app, [str(tmp_path / "pq"), str(tmp_path / "out"), *args]
        )

    assert result.exit_code == 1
    assert message in result.output
    convert.assert_not_called()


def test_options_are_forwarded(tmp_path):
    with patch(_CONVERT) as convert:
        result = _runner.invoke(
            app,
            [
                str(tmp_path / "pq"),
                str(tmp_path / "out"),
                "--name", "h2ds",
                "--start-date", "2021-01-01",
                "--end-date", "2021-12-31",
                "-v", "sst",
                "-v", "adt",
                # The old spelling, kept so existing scripts keep working.
                "--time-resolution", "year",
                "--date-format", "yearmonth",
                "--layout", "map",
            ],
        )  # fmt: skip

    assert result.exit_code == 0, result.output
    args, kwargs = convert.call_args
    assert args == (tmp_path / "pq", tmp_path / "out")
    assert kwargs == {
        "name": "h2ds",
        "start_date": "2021-01-01",
        "end_date": "2021-12-31",
        "file_period": "year",
        "date_format": "yearmonth",
        "variables": ["sst", "adt"],
        "layout": "map",
    }


def test_defaults(tmp_path):
    with patch(_CONVERT) as convert:
        result = _runner.invoke(app, [str(tmp_path / "pq"), str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    kwargs = convert.call_args.kwargs
    assert kwargs["name"] == "data"
    assert kwargs["variables"] is None
    assert (kwargs["file_period"], kwargs["date_format"], kwargs["layout"]) == (
        "month",
        "year",
        "timeseries",
    )


def test_round_trip_rebuilds_the_values(tmp_path):
    """Parquet written by the indexer comes back as the same grid of values."""
    dates = [date(2021, 1, d) for d in (1, 2, 3)]
    df = make_grid_df(dates, variables={"sst": 20.0, "adt": 0.5})
    ParquetIndexer(tmp_path / "pq").add_data(df)
    out = tmp_path / "out"

    result = _runner.invoke(app, [str(tmp_path / "pq"), str(out), "--name", "demo"])

    assert result.exit_code == 0, result.output
    files = sorted(out.glob("*.zarr"))
    assert len(files) == 1 and files[0].name.startswith("demo_")
    with xr.open_zarr(files[0], consolidated=False) as ds:
        assert ds.sizes["time"] == 3
        assert {"sst", "adt"} <= set(ds.data_vars)
        row = df.filter(
            (df["lon"] == -5.0) & (df["lat"] == 35.0) & (df["time"] == dates[1])
        )
        got = float(ds["sst"].sel(time="2021-01-02", lon=-5.0, lat=35.0))
        # Parquet stores Float32, so compare at that precision.
        np.testing.assert_allclose(got, row["sst"].item(), rtol=1e-6)


def test_variable_subset_is_honoured(tmp_path):
    df = make_grid_df([date(2021, 1, 1)], variables={"sst": 20.0, "adt": 0.5})
    ParquetIndexer(tmp_path / "pq").add_data(df)
    out = tmp_path / "out"

    result = _runner.invoke(app, [str(tmp_path / "pq"), str(out), "-v", "sst"])

    assert result.exit_code == 0, result.output
    with xr.open_zarr(next(out.glob("*.zarr")), consolidated=False) as ds:
        assert "sst" in ds.data_vars and "adt" not in ds.data_vars
