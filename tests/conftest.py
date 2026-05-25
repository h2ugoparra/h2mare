"""Shared fixtures for h2mare test suite."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import date


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------


def make_grid_df(
    dates: list[date],
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
    variables: dict[str, float] | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Build a minimal gridded Polars DataFrame (time × lon × lat).

    Parameters
    ----------
    dates:     list of date objects
    lons:      longitude values
    lats:      latitude values
    variables: mapping of column name → base value (random noise added).
               Defaults to {"sst": 20.0}.
    seed:      random seed for reproducibility.
    """
    if variables is None:
        variables = {"sst": 20.0}

    rng = np.random.default_rng(seed)
    rows = [
        {
            "time": d,
            "lon": lon,
            "lat": lat,
            **{k: float(base + rng.uniform(-1, 1)) for k, base in variables.items()},
        }
        for d in dates
        for lon in lons
        for lat in lats
    ]
    return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Date))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jan_df():
    """9 rows: 3-day × 3×3 grid, single variable 'sst'."""
    return make_grid_df(
        dates=[date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
    )


@pytest.fixture
def parquet_dir(tmp_path):
    """Empty path for a ParquetIndexer store."""
    return tmp_path / "parquet"


@pytest.fixture
def loaded_indexer(parquet_dir, jan_df):
    """ParquetIndexer with january data pre-loaded."""
    from h2mare.storage.parquet_indexer import ParquetIndexer

    idx = ParquetIndexer(parquet_dir)
    idx.add_data(jan_df)
    return idx
