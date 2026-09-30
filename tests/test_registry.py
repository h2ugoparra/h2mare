"""Tests for processing/registry.py — PROCESSORS dict."""

from pathlib import Path

import msgspec
import pytest
import yaml

from h2mare.models import AppConfig
from h2mare.processing.registry import PROCESSORS


class TestProcessorsRegistry:
    _EXPECTED_KEYS = {
        "atm-instante",
        "atm-accum-avg",
        "radiation",
        "waves",
        "chl",
        "sst",
        "ssh",
        "fsle",
    }

    def test_expected_keys_all_present(self):
        assert self._EXPECTED_KEYS == set(PROCESSORS.keys())

    def test_all_values_are_callable(self):
        for key, fn in PROCESSORS.items():
            assert callable(fn), f"PROCESSORS[{key!r}] is not callable"

    def test_no_extra_keys(self):
        assert set(PROCESSORS.keys()) == self._EXPECTED_KEYS

    def test_a_rename_only_var_key_needs_no_processor(self):
        """mld's processor only renamed mlotst, which config now does."""
        assert "mld" not in PROCESSORS


class TestShippedSourceRenames:
    """
    Guard the seam between the tracked config.yaml and the processors.

    A processor is written for the names its var_key publishes — ``process_sst``
    reads ``ds["sst"]`` — and ``source_renames`` is what puts them there. A
    rename to some other name leaves the processor indexing a variable that is
    no longer in the dataset: a KeyError partway through a convert, rather than
    anything config load can see.
    """

    def _shipped(self):
        raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        return msgspec.convert(
            {"variables": raw["variables"], "secrets": {}}, AppConfig, strict=False
        ).variables

    @pytest.mark.parametrize(
        "var_key, renames",
        [
            ("sst", {"analysed_sst": "sst"}),
            ("chl", {"CHL": "chl"}),
            ("mld", {"mlotst": "mld"}),
        ],
    )
    def test_the_shipped_map_is_the_one_the_code_expects(self, var_key, renames):
        assert self._shipped()[var_key].source_renames == renames
