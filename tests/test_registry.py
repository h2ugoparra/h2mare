"""Tests for processing/registry.py — PROCESSORS dict."""

from h2mare.processing.registry import PROCESSORS


class TestProcessorsRegistry:
    _EXPECTED_KEYS = {
        "atm-instante",
        "atm-accum-avg",
        "radiation",
        "waves",
        "chl",
        "sst",
        "mld",
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
