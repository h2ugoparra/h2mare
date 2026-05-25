"""Tests for utils/date_range.py — resolve_date_range."""
from unittest.mock import patch

import pandas as pd
import pytest

from h2mare.utils.date_range import resolve_date_range
from h2mare.types import DateRange


class TestResolveDateRange:

    def test_explicit_dates_returned_as_date_range(self):
        result = resolve_date_range("sst", "2020-01-01", "2020-12-31")
        assert pd.Timestamp(result.start) == pd.Timestamp("2020-01-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2020-12-31")

    def test_start_after_end_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid date range"):
            resolve_date_range("sst", "2025-01-01", "2020-01-01")

    def test_same_start_and_end_is_valid(self):
        result = resolve_date_range("sst", "2021-06-15", "2021-06-15")
        assert pd.Timestamp(result.start) == pd.Timestamp(result.end)

    def test_no_store_coverage_raises_value_error(self):
        with patch("h2mare.utils.date_range.get_store_coverage", return_value=None):
            with pytest.raises(ValueError, match="No existing data"):
                resolve_date_range("sst", None, None)

    def test_start_inferred_as_day_after_store_end(self):
        coverage = DateRange("2020-01-01", "2021-12-31")
        with patch("h2mare.utils.date_range.get_store_coverage", return_value=coverage):
            result = resolve_date_range("sst", None, "2022-06-30")
        assert pd.Timestamp(result.start) == pd.Timestamp("2022-01-01")

    def test_end_inferred_as_today_when_not_provided(self):
        coverage = DateRange("2015-01-01", "2019-12-31")
        now = pd.Timestamp("2024-06-15")
        with (
            patch("h2mare.utils.date_range.get_store_coverage", return_value=coverage),
            patch("pandas.Timestamp.now", return_value=now),
        ):
            result = resolve_date_range("sst", "2020-01-01", None)
        assert pd.Timestamp(result.end) == now.normalize()
