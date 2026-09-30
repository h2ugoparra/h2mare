"""Tests for utils/parallel.py — pool sizing."""

import os

import pytest

from h2mare.config import get_settings
from h2mare.utils.parallel import resolve_n_workers


@pytest.fixture
def machine(monkeypatch):
    """Set the host's CPU count and the H2MARE_MAX_WORKERS ceiling."""

    def _set(cpus: int, ceiling: int | None = None):
        monkeypatch.setattr(os, "cpu_count", lambda: cpus)
        monkeypatch.setattr(get_settings(), "MAX_WORKERS", ceiling)

    return _set


class TestResolveNWorkers:
    def test_the_requested_count_is_used_when_it_fits(self, machine):
        machine(cpus=16)
        assert resolve_n_workers(6, 10) == 6

    def test_none_falls_back_to_the_sites_default(self, machine):
        machine(cpus=16)
        assert resolve_n_workers(None, 10) == 10

    def test_the_default_is_capped_to_the_host(self, machine):
        """What this was added for: DEFAULT_N_WORKERS = 10 on a 4-core box
        started 10 spawn workers, each re-importing h2mare."""
        machine(cpus=4)
        assert resolve_n_workers(None, 10) == 4

    def test_a_request_below_the_cap_is_left_alone(self, machine):
        machine(cpus=4)
        assert resolve_n_workers(2, 10) == 2

    def test_the_env_ceiling_caps_it(self, machine):
        machine(cpus=16, ceiling=3)
        assert resolve_n_workers(None, 10) == 3

    def test_the_ceiling_does_not_raise_a_smaller_ask(self, machine):
        """It is a ceiling, not a value: more workers than a site's default
        comes from that site's own n_workers, not from the environment."""
        machine(cpus=16, ceiling=12)
        assert resolve_n_workers(None, 10) == 10
        assert resolve_n_workers(2, 10) == 2

    def test_the_lower_of_the_two_limits_wins(self, machine):
        machine(cpus=2, ceiling=8)
        assert resolve_n_workers(None, 10) == 2
        machine(cpus=8, ceiling=2)
        assert resolve_n_workers(None, 10) == 2

    def test_an_unknown_cpu_count_falls_back_to_one(self, monkeypatch):
        """os.cpu_count() returns None where the platform cannot tell."""
        monkeypatch.setattr(os, "cpu_count", lambda: None)
        monkeypatch.setattr(get_settings(), "MAX_WORKERS", None)
        assert resolve_n_workers(None, 10) == 1
