"""Tests for validators.py."""

import msgspec
import pytest

from h2mare.models import AppConfig
from h2mare.types import TimeResolution
from h2mare.validators import (
    validate_time_resolution,
    validate_var_key,
    validate_var_keys,
)

_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "pattern": r".*\.nc",
}
_CONFIG = msgspec.convert({"variables": {"sst": _ENTRY}, "secrets": {}}, AppConfig)


class TestValidateVarKey:
    def test_valid_key_returns_key(self):
        assert validate_var_key("sst", _CONFIG) == "sst"

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="not found"):
            validate_var_key("nonexistent", _CONFIG)


class TestValidateVarKeys:
    def test_all_valid(self):
        validate_var_keys(["sst"], _CONFIG)

    def test_invalid_keys_raises(self):
        with pytest.raises(ValueError, match="not found"):
            validate_var_keys(["sst", "bad_key"], _CONFIG)

    def test_empty_list_passes(self):
        validate_var_keys([], _CONFIG)


class TestValidateTimeResolution:
    def test_enum_passthrough(self):
        assert validate_time_resolution(TimeResolution.MONTH) is TimeResolution.MONTH

    def test_string_month(self):
        assert validate_time_resolution("month") == TimeResolution.MONTH

    def test_string_year(self):
        assert validate_time_resolution("year") == TimeResolution.YEAR

    def test_string_case_insensitive(self):
        assert validate_time_resolution("MONTH") == TimeResolution.MONTH

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid period"):
            validate_time_resolution("daily")

    def test_non_string_non_enum_raises(self):
        with pytest.raises(ValueError, match="Period must be"):
            validate_time_resolution(42)  # type: ignore[arg-type]
