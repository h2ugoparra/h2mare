"""Tests for utils/datetime_utils.py."""

from datetime import date, datetime

import pandas as pd
import pytest

from h2mare.utils.datetime_utils import (
    date_to_standard_string,
    end_of_day,
    more_than_one_year,
    normalize_date,
    normalize_dates,
    to_datetime,
)


class TestToDatetime:
    def test_datetime_passthrough(self):
        dt = datetime(2020, 6, 15, 12, 30)
        assert to_datetime(dt) is dt

    def test_date_object(self):
        result = to_datetime(date(2020, 6, 15))
        assert result == datetime(2020, 6, 15, 0, 0)

    def test_string_iso(self):
        result = to_datetime("2020-06-15")
        assert result == datetime(2020, 6, 15)

    def test_timestamp(self):
        ts = pd.Timestamp("2020-06-15")
        result = to_datetime(ts)
        assert isinstance(result, datetime)
        assert result.year == 2020

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            to_datetime(12345)


class TestNormalizeDate:
    def test_scalar_string(self):
        result = normalize_date("2020-03-15")
        assert result == pd.Timestamp("2020-03-15")
        assert result.hour == 0

    def test_scalar_timestamp(self):
        result = normalize_date(pd.Timestamp("2020-03-15 12:30"))
        assert result.hour == 0

    @pytest.mark.parametrize("bad", [None, pd.NaT, float("nan")])
    def test_unusable_date_names_the_argument(self, bad):
        """
        Regression: pd.Timestamp returns NaT for these and NaT has no
        .normalize(), so the failure surfaced as "AttributeError: 'NaTType'
        object has no attribute 'normalize'" — naming neither the argument nor
        the mistake.
        """
        with pytest.raises(ValueError, match="Not a usable date"):
            normalize_date(bad)


class TestEndOfDay:
    def test_covers_the_last_sub_daily_step(self):
        got = end_of_day("2020-03-15")

        assert got > pd.Timestamp("2020-03-15 23:00")
        assert got < pd.Timestamp("2020-03-16")

    def test_stays_within_the_day_of_a_stamped_input(self):
        # Normalizes first, so a noon-stamped bound does not spill into the
        # following day.
        assert end_of_day(pd.Timestamp("2020-03-15 12:00")) < pd.Timestamp("2020-03-16")


class TestNormalizeDates:
    def test_list_of_strings(self):
        result = normalize_dates(["2020-01-01", "2020-06-30"])
        assert len(result) == 2
        assert all(ts.hour == 0 for ts in result)

    def test_one_bad_entry_in_a_list_is_reported_like_a_lone_one(self):
        """The list branch normalized inline, so it raised AttributeError."""
        with pytest.raises(ValueError, match="Not a usable date"):
            normalize_dates(["2020-01-01", None])

    def test_tuple_of_dates(self):
        result = normalize_dates((date(2020, 1, 1), date(2020, 6, 30)))
        assert len(result) == 2

    def test_scalar_returns_single_element_list(self):
        result = normalize_dates("2020-03-15")
        assert result == [pd.Timestamp("2020-03-15")]

    def test_empty_list_returns_empty_list(self):
        assert normalize_dates([]) == []

    def test_datetime_index(self):
        """Regression: a DatetimeIndex was taken as one date and raised."""
        days = pd.date_range("2025-01-15 06:00", periods=3, freq="D")
        assert normalize_dates(days) == [
            pd.Timestamp("2025-01-15"),
            pd.Timestamp("2025-01-16"),
            pd.Timestamp("2025-01-17"),
        ]

    def test_series_and_array(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-30"]))
        expected = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")]
        assert normalize_dates(dates) == expected
        assert normalize_dates(dates.to_numpy()) == expected


class TestMoreThanOneYear:
    def test_true(self):
        a = pd.Timestamp("2020-01-01")
        b = pd.Timestamp("2021-06-01")
        assert more_than_one_year(a, b)

    def test_false_same_year(self):
        a = pd.Timestamp("2020-01-01")
        b = pd.Timestamp("2020-11-30")
        assert not more_than_one_year(a, b)

    def test_order_independent(self):
        a = pd.Timestamp("2021-06-01")
        b = pd.Timestamp("2020-01-01")
        assert more_than_one_year(a, b)


class TestDateToStandardString:
    def test_string_input(self):
        assert date_to_standard_string("2020-03-15") == "2020-03-15"

    def test_datetime_input(self):
        assert date_to_standard_string(datetime(2020, 3, 15, 12, 0)) == "2020-03-15"

    def test_date_input(self):
        assert date_to_standard_string(date(2020, 3, 15)) == "2020-03-15"

    def test_timestamp_input(self):
        assert date_to_standard_string(pd.Timestamp("2020-03-15")) == "2020-03-15"
