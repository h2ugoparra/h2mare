"""Tests for downloader/cmems_utils.py — pure helper functions."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from h2mare.downloader.cmems_utils import (
    CMEMSAPIError,
    _find_time_coordinate,
    _parse_time_values,
    clear_dataset_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_metadata(coord_id: str, min_val, max_val):
    """Build a minimal mock metadata tree with one coordinate."""
    coord = MagicMock()
    coord.coordinate_id = coord_id
    coord.minimum_value = min_val
    coord.maximum_value = max_val

    variable = MagicMock()
    variable.coordinates = [coord]
    service = MagicMock()
    service.variables = [variable]
    part = MagicMock()
    part.services = [service]
    version = MagicMock()
    version.parts = [part]
    dataset = MagicMock()
    dataset.versions = [version]
    product = MagicMock()
    product.datasets = [dataset]
    metadata = MagicMock()
    metadata.products = [product]
    return metadata


# ---------------------------------------------------------------------------
# _find_time_coordinate
# ---------------------------------------------------------------------------

class TestFindTimeCoordinate:

    def test_returns_min_and_max_when_time_coord_present(self):
        metadata = _build_metadata("time", 1_000_000.0, 2_000_000.0)
        result = _find_time_coordinate(metadata)
        assert result is not None
        assert result["minimum_value"] == 1_000_000.0
        assert result["maximum_value"] == 2_000_000.0

    def test_returns_none_when_coord_id_is_not_time(self):
        metadata = _build_metadata("lat", 0.0, 90.0)
        assert _find_time_coordinate(metadata) is None

    def test_returns_none_for_empty_products_list(self):
        metadata = MagicMock()
        metadata.products = []
        assert _find_time_coordinate(metadata) is None

    def test_returns_none_when_products_attr_missing(self):
        metadata = MagicMock(spec=[])  # no 'products' attribute
        assert _find_time_coordinate(metadata) is None


# ---------------------------------------------------------------------------
# _parse_time_values
# ---------------------------------------------------------------------------

class TestParseTimeValues:

    def _ms(self, ts: str) -> float:
        return pd.Timestamp(ts).timestamp() * 1000

    def test_converts_ms_epoch_to_timestamps(self):
        tmin, tmax = _parse_time_values(
            {"minimum_value": self._ms("2020-01-01"), "maximum_value": self._ms("2021-01-01")},
            "test-ds",
        )
        assert tmin == pd.Timestamp("2020-01-01")
        assert tmax == pd.Timestamp("2021-01-01")

    def test_returns_normalized_dates(self):
        tmin, tmax = _parse_time_values(
            {"minimum_value": self._ms("2020-06-15"), "maximum_value": self._ms("2021-06-15")},
            "test-ds",
        )
        assert tmin.hour == 0 and tmin.minute == 0

    def test_raises_when_minimum_value_is_none(self):
        with pytest.raises(CMEMSAPIError, match="Missing time bounds"):
            _parse_time_values({"minimum_value": None, "maximum_value": 1000.0}, "ds")

    def test_raises_when_maximum_value_is_none(self):
        with pytest.raises(CMEMSAPIError, match="Missing time bounds"):
            _parse_time_values({"minimum_value": 1000.0, "maximum_value": None}, "ds")

    def test_raises_when_values_are_not_numeric(self):
        with pytest.raises(CMEMSAPIError, match="Invalid time values"):
            _parse_time_values({"minimum_value": "not_a_number", "maximum_value": "x"}, "ds")

    def test_raises_when_start_after_end(self):
        t1 = self._ms("2020-01-01")
        t2 = self._ms("2021-01-01")
        with pytest.raises(CMEMSAPIError, match="Invalid time range"):
            _parse_time_values({"minimum_value": t2, "maximum_value": t1}, "ds")


# ---------------------------------------------------------------------------
# clear_dataset_cache
# ---------------------------------------------------------------------------

class TestClearDatasetCache:

    def test_no_error_on_empty_cache(self):
        clear_dataset_cache()  # must not raise
