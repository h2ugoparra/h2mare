"""Tests for storage/coverage.py."""

from unittest.mock import patch

import pytest

from h2mare.storage.coverage import get_store_coverage, split_time_range
from h2mare.types import DateRange, TimeResolution


class TestSplitTimeRange:
    def test_monthly_split_single_month(self):
        dr = DateRange("2020-03-01", "2020-03-31")
        chunks = split_time_range(dr, TimeResolution.MONTH)
        assert len(chunks) == 1
        assert chunks[0].start.month == 3
        assert chunks[0].end.month == 3

    def test_monthly_split_multi_month(self):
        dr = DateRange("2020-01-01", "2020-03-31")
        chunks = split_time_range(dr, TimeResolution.MONTH)
        assert len(chunks) == 3
        assert chunks[0].start.month == 1
        assert chunks[2].end.month == 3

    def test_yearly_split_single_year(self):
        dr = DateRange("2020-01-01", "2020-12-31")
        chunks = split_time_range(dr, TimeResolution.YEAR)
        assert len(chunks) == 1
        assert chunks[0].start.year == 2020
        assert chunks[0].end.year == 2020

    def test_yearly_split_multi_year(self):
        dr = DateRange("2020-01-01", "2021-12-31")
        chunks = split_time_range(dr, TimeResolution.YEAR)
        assert len(chunks) == 2
        assert chunks[0].start.year == 2020
        assert chunks[1].start.year == 2021

    def test_monthly_split_partial_month(self):
        dr = DateRange("2020-01-15", "2020-02-10")
        chunks = split_time_range(dr, TimeResolution.MONTH)
        assert len(chunks) == 2

    def test_invalid_split_raises(self):
        dr = DateRange("2020-01-01", "2020-12-31")
        with pytest.raises((ValueError, AttributeError)):
            split_time_range(dr, "weekly")  # type: ignore[arg-type]


class TestGetStoreCoverage:
    def test_returns_date_range_when_data_exists(self):
        mock_coverage = DateRange("2020-01-01", "2020-12-31")
        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            return_value=mock_coverage,
        ):
            result = get_store_coverage("sst")
        assert result is not None
        assert result.start.year == 2020

    def test_returns_none_when_no_data(self):
        with patch("h2mare.storage.coverage.get_zarr_time_coverage", return_value=None):
            result = get_store_coverage("sst")
        assert result is None

    def test_returns_none_on_exception(self):
        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            side_effect=RuntimeError("store unavailable"),
        ):
            result = get_store_coverage("sst")
        assert result is None
