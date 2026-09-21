"""Tests for utils/paths.py."""

from unittest.mock import MagicMock, patch

import msgspec
import pytest

from h2mare.models import AppConfig
from h2mare.utils.paths import (
    resolve_download_path,
    resolve_store_path,
    static_layer_path,
    store_root_for,
)

_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r".*\.nc",
}
_CONFIG = msgspec.convert({"variables": {"sst": _ENTRY}, "secrets": {}}, AppConfig)
_VAR_CONFIG = _CONFIG.variables["sst"]


class TestResolveDownloadPath:
    def test_explicit_root_used(self, tmp_path):
        result = resolve_download_path(
            _VAR_CONFIG, download_root=tmp_path, warn_if_missing=False
        )
        assert result == tmp_path.resolve()

    def test_missing_path_still_returns_resolved_path(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = resolve_download_path(
            _VAR_CONFIG, download_root=missing, warn_if_missing=True
        )
        assert result == missing.resolve()

    def test_warn_if_missing_false_skips_check(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        result = resolve_download_path(
            _VAR_CONFIG, download_root=missing, warn_if_missing=False
        )
        assert result == missing.resolve()

    def test_falls_back_to_settings_downloads_dir(self, tmp_path):
        mock_settings = MagicMock()
        mock_settings.DOWNLOADS_DIR = tmp_path
        with patch("h2mare.utils.paths.get_settings", return_value=mock_settings):
            result = resolve_download_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (tmp_path / _VAR_CONFIG.local_folder).resolve()


def _settings(tmp_path, *, store_root=None, overridden=False):
    """
    Stand-in Settings.

    ``store_root_overridden`` must be set explicitly: a MagicMock invents it as
    a truthy attribute, which would make every resolution look like a
    ``--store-path`` run and hide the per-variable root entirely.
    """
    mock_settings = MagicMock()
    mock_settings.STORE_ROOT = store_root
    mock_settings.ZARR_DIR = tmp_path / "zarr_dir"
    mock_settings.store_root_overridden = overridden
    return mock_settings


def _var_config(**overrides):
    """A var config built from _ENTRY with fields overridden."""
    entry = {**_ENTRY, **overrides}
    cfg = msgspec.convert({"variables": {"sst": entry}, "secrets": {}}, AppConfig)
    return cfg.variables["sst"]


class TestResolveStorePath:
    def test_explicit_root_used(self, tmp_path):
        result = resolve_store_path(
            _VAR_CONFIG, store_root=tmp_path, warn_if_missing=False
        )
        assert result == tmp_path.resolve()

    def test_store_dir_used_when_available(self, tmp_path):
        settings = _settings(tmp_path, store_root=tmp_path)
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = resolve_store_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (tmp_path / _VAR_CONFIG.local_folder).resolve()

    def test_falls_back_to_zarr_dir_when_store_root_none(self, tmp_path):
        settings = _settings(tmp_path, store_root=None)
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = resolve_store_path(_VAR_CONFIG, warn_if_missing=False)
        assert result == (settings.ZARR_DIR / _VAR_CONFIG.local_folder).resolve()

    def test_variables_own_root_beats_store_root(self, tmp_path):
        """The whole point of the field: this variable lives on another drive."""
        own = tmp_path / "other_drive"
        var_config = _var_config(store_root=str(own))
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")

        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = resolve_store_path(var_config, warn_if_missing=False)

        assert result == (own / "sst").resolve()

    def test_explicit_argument_still_wins_over_the_variables_own_root(self, tmp_path):
        """An explicit store_root names one store's directory and is used as-is."""
        var_config = _var_config(store_root=str(tmp_path / "other_drive"))
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")

        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = resolve_store_path(
                var_config, store_root=tmp_path / "explicit", warn_if_missing=False
            )

        assert result == (tmp_path / "explicit").resolve()


class TestStoreRootFor:
    """
    Precedence: --store-path > config.yaml store_root > default_root > STORE_ROOT
    > ZARR_DIR. Returns a *root*; callers join local_folder themselves.
    """

    def test_returns_store_root_when_variable_names_none(self, tmp_path):
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            assert store_root_for(_VAR_CONFIG) == tmp_path / "from_env"

    def test_returns_the_variables_own_root(self, tmp_path):
        var_config = _var_config(store_root=str(tmp_path / "own"))
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            assert store_root_for(var_config) == tmp_path / "own"

    def test_override_beats_the_variables_own_root(self, tmp_path):
        """
        --store-path relocates a whole run deliberately. A flag that moved only
        the variables which had not opted out would be a partial relocation.
        """
        var_config = _var_config(store_root=str(tmp_path / "own"))
        settings = _settings(
            tmp_path, store_root=tmp_path / "from_flag", overridden=True
        )
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            assert store_root_for(var_config) == tmp_path / "from_flag"

    def test_default_root_used_when_variable_names_none(self, tmp_path):
        """What keeps PipelineManager and Compiler on the root they were handed."""
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = store_root_for(_VAR_CONFIG, tmp_path / "handed_down")
        assert result == tmp_path / "handed_down"

    def test_variables_own_root_beats_the_default_root(self, tmp_path):
        var_config = _var_config(store_root=str(tmp_path / "own"))
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            result = store_root_for(var_config, tmp_path / "handed_down")
        assert result == tmp_path / "own"

    def test_config_without_the_field_is_unchanged(self, tmp_path):
        """Backward compatibility: an entry declaring nothing resolves as before."""
        settings = _settings(tmp_path, store_root=None)
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            assert store_root_for(_VAR_CONFIG) == settings.ZARR_DIR

    def test_tolerates_a_config_predating_the_field(self, tmp_path):
        """Stand-in configs without the attribute, as step_freq allows for time_step."""
        from types import SimpleNamespace

        stub = SimpleNamespace(local_folder="sst")
        settings = _settings(tmp_path, store_root=tmp_path / "from_env")
        with patch("h2mare.utils.paths.get_settings", return_value=settings):
            assert store_root_for(stub) == tmp_path / "from_env"  # type: ignore[arg-type]


class TestStaticLayerPath:
    def test_joins_the_layer_file_onto_the_store_dir(self, tmp_path):
        cfg = _var_config(layers={"15s": "b15.zarr", "60s": "b60.zarr"})
        assert static_layer_path(cfg, "60s", tmp_path) == tmp_path / "b60.zarr"

    @pytest.mark.parametrize("layer", ["30s", None])
    def test_undeclared_layer_names_the_declared_ones(self, tmp_path, layer):
        cfg = _var_config(layers={"15s": "b15.zarr"})
        with pytest.raises(ValueError, match=r"\['15s'\]"):
            static_layer_path(cfg, layer, tmp_path)
