"""Tests for utils/datetime_utils.py."""

import pytest
import pandas as pd
from datetime import date, datetime

from h2mare.utils.datetime_utils import (
    to_datetime,
    normalize_date,
    more_than_one_year,
    date_to_standard_string,
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

    def test_list_of_strings(self):
        result = normalize_date(["2020-01-01", "2020-06-30"])
        assert len(result) == 2
        assert all(ts.hour == 0 for ts in result)

    def test_tuple_of_dates(self):
        result = normalize_date((date(2020, 1, 1), date(2020, 6, 30)))
        assert len(result) == 2


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
