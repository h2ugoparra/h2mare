"""Tests for config.py — Settings class."""
import warnings
from pathlib import Path

import pytest

from h2mare.config import Settings

_MINIMAL_CONFIG_YAML = """\
global_attrs:
  title: test dataset

variable_attrs:
  sst:
    long_name: Sea Surface Temperature
    units: K

variables:
  sst:
    local_folder: sst
    variables:
      - analysed_sst
    dataset_id_rep: cmems-rep-sst
    source: cmems
    pattern: '.*\.nc'
    subset: true
    bbox:
      - -80
      - 0
      - 10
      - 70
"""


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------

class TestFindProjectRoot:

    def test_h2mare_root_env_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.BASE_DIR == tmp_path.resolve()

    def test_h2mare_root_sets_project_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s._project_mode is True

    def test_base_dir_is_resolved_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.BASE_DIR.is_absolute()


# ---------------------------------------------------------------------------
# _get_store_dir
# ---------------------------------------------------------------------------

class TestGetStoreDir:

    def test_store_root_env_returned_as_path(self, tmp_path, monkeypatch):
        store = tmp_path / "my_store"
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.setenv("STORE_ROOT", str(store))
        s = Settings()
        assert s.STORE_ROOT == store.resolve()

    def test_missing_store_root_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.STORE_ROOT is None


# ---------------------------------------------------------------------------
# load_app_config
# ---------------------------------------------------------------------------

class TestLoadAppConfig:

    def _make_settings(self, tmp_path, monkeypatch) -> Settings:
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        return Settings()

    def test_loads_configured_variable(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        config = s.load_app_config()
        assert "sst" in config.variables

    def test_raises_when_config_yaml_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        with pytest.raises(FileNotFoundError):
            s.load_app_config()

    def test_second_call_returns_same_object(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        config1 = s.load_app_config()
        config2 = s.load_app_config()
        assert config1 is config2

    def test_global_attrs_populated(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        s.load_app_config()
        assert s._global_attrs.get("title") == "test dataset"

    def test_warns_when_aviso_vars_but_missing_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        monkeypatch.delenv("AVISO_FTP_SERVER", raising=False)
        monkeypatch.delenv("AVISO_USERNAME", raising=False)
        monkeypatch.delenv("AVISO_PASSWORD", raising=False)
        aviso_yaml = _MINIMAL_CONFIG_YAML.replace("source: cmems", "source: aviso")
        (tmp_path / "config.yaml").write_text(aviso_yaml)
        s = Settings()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.load_app_config()
        assert any(issubclass(warning.category, RuntimeWarning) for warning in w)


# ---------------------------------------------------------------------------
# get_var_info
# ---------------------------------------------------------------------------

class TestGetVarInfo:

    def _make_settings(self, tmp_path, monkeypatch) -> Settings:
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        return Settings()

    def test_returns_attrs_for_known_var(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        info = s.get_var_info("sst")
        assert info.get("long_name") == "Sea Surface Temperature"
        assert info.get("units") == "K"

    def test_returns_empty_dict_for_unknown_var(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        assert s.get_var_info("nonexistent") == {}


# ---------------------------------------------------------------------------
# get_available_var_keys
# ---------------------------------------------------------------------------

class TestGetAvailableVarKeys:

    def test_returns_list_of_configured_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        s = Settings()
        keys = s.get_available_var_keys()
        assert isinstance(keys, list)
        assert "sst" in keys
