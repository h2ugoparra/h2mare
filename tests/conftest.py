"""Shared fixtures for h2mare test suite."""

import os
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from h2mare.config import get_settings

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The config under test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def repo_config() -> "object":
    """
    Run the suite against the repo's own config.yaml, whatever the machine has.

    ``H2MARE_ROOT`` is a user-wide setting on a developer box — pointed at
    whichever project is being worked on — and it outranks the repo's .env,
    because python-dotenv does not override an existing variable. Pointed at
    another project it takes the whole suite with it: seven tests failed
    against a config that simply does not define the var_keys they name, which
    says something about the machine and nothing about the code.

    CI has no ``H2MARE_ROOT`` and finds the repo by its config.yaml. This makes
    a local run agree with it, and is why ``get_settings`` documents its cache
    as clearable.
    """
    previous = os.environ.get("H2MARE_ROOT")
    os.environ["H2MARE_ROOT"] = str(REPO)
    get_settings.cache_clear()
    yield
    if previous is None:
        os.environ.pop("H2MARE_ROOT", None)
    else:
        os.environ["H2MARE_ROOT"] = previous
    get_settings.cache_clear()


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


@pytest.fixture
def multivar_df():
    """27 rows: 3 months × 3×3 grid, four variables (sst, chl, mld, adt)."""
    return make_grid_df(
        dates=[date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)],
        variables={"sst": 20.0, "chl": 0.5, "mld": 30.0, "adt": 0.1},
    )


@pytest.fixture
def multivar_indexer(parquet_dir, multivar_df):
    """ParquetIndexer pre-loaded with four variables."""
    from h2mare.storage.parquet_indexer import ParquetIndexer

    idx = ParquetIndexer(parquet_dir)
    idx.add_data(multivar_df)
    return idx


# ---------------------------------------------------------------------------
# Front detection
# ---------------------------------------------------------------------------


class SerialPool:
    """
    Stand-in for ``mp.Pool`` that runs in the calling process.

    Detection is the same code either way, and a real pool costs a process
    spawn per test on Windows; ``test_fronts.py::TestRealPool`` keeps that path
    honest.
    """

    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]


@pytest.fixture
def interim_dir(tmp_path, monkeypatch) -> Path:
    """
    Point INTERIM_DIR at tmp_path and return it.

    Front detection stages there, so without this a test would write into
    whatever store the machine has deployed.
    """
    settings = SimpleNamespace(INTERIM_DIR=tmp_path / "interim")
    monkeypatch.setattr("h2mare.processing.core.fronts.get_settings", lambda: settings)
    return settings.INTERIM_DIR


@pytest.fixture
def serial_pool(monkeypatch):
    """Run front detection in-process rather than across a pool."""
    monkeypatch.setattr(
        "h2mare.processing.core.fronts._pool", lambda n_workers: SerialPool(n_workers)
    )
