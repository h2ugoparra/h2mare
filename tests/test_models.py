"""Tests for msgspec-based data models (KeyVarConfigEntry, AppConfig)."""

import pytest
import msgspec

from h2mare.models import AppConfig, KeyVarConfigEntry, SecretsConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "pattern": r".*\.nc",
}


# ---------------------------------------------------------------------------
# KeyVarConfigEntry
# ---------------------------------------------------------------------------


class TestKeyVarConfigEntry:
    def test_valid_minimal(self):
        """Required fields only — all optional fields default correctly."""
        entry = msgspec.convert(VALID_ENTRY, KeyVarConfigEntry)
        assert entry.local_folder == "sst"
        assert entry.dataset_id_nrt is None
        assert entry.subset is True
        assert entry.bbox is None
        assert entry.depth_range is None

    def test_valid_with_bbox(self):
        entry = msgspec.convert(
            {**VALID_ENTRY, "bbox": [-10.0, 30.0, 10.0, 50.0]}, KeyVarConfigEntry
        )
        assert entry.bbox == (-10.0, 30.0, 10.0, 50.0)

    def test_valid_with_depth_range(self):
        entry = msgspec.convert(
            {**VALID_ENTRY, "depth_range": [0.0, 500.0]}, KeyVarConfigEntry
        )
        assert entry.depth_range == (0.0, 500.0)

    def test_variables_as_string(self):
        """variables field accepts a plain string (not just a list)."""
        entry = msgspec.convert(
            {**VALID_ENTRY, "variables": "analysed_sst"}, KeyVarConfigEntry
        )
        assert entry.variables == "analysed_sst"

    # --- bbox validation ---

    def test_invalid_lon_too_large(self):
        with pytest.raises(msgspec.ValidationError, match="Longitude"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [170.0, 30.0, 200.0, 50.0]}, KeyVarConfigEntry
            )

    def test_invalid_lon_too_small(self):
        with pytest.raises(msgspec.ValidationError, match="Longitude"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [-200.0, 30.0, 10.0, 50.0]}, KeyVarConfigEntry
            )

    def test_invalid_lat_too_large(self):
        with pytest.raises(msgspec.ValidationError, match="Latitude"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [-10.0, 30.0, 10.0, 100.0]}, KeyVarConfigEntry
            )

    def test_invalid_lat_too_small(self):
        with pytest.raises(msgspec.ValidationError, match="Latitude"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [-10.0, -95.0, 10.0, 50.0]}, KeyVarConfigEntry
            )

    def test_invalid_lon_order(self):
        """lon_min must be strictly less than lon_max."""
        with pytest.raises(msgspec.ValidationError, match="lon_min"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [10.0, 30.0, -10.0, 50.0]}, KeyVarConfigEntry
            )

    def test_invalid_lon_equal(self):
        with pytest.raises(msgspec.ValidationError, match="lon_min"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [10.0, 30.0, 10.0, 50.0]}, KeyVarConfigEntry
            )

    def test_invalid_lat_order(self):
        """lat_min must be strictly less than lat_max."""
        with pytest.raises(msgspec.ValidationError, match="lat_min"):
            msgspec.convert(
                {**VALID_ENTRY, "bbox": [-10.0, 50.0, 10.0, 30.0]}, KeyVarConfigEntry
            )

    def test_bbox_none_skips_validation(self):
        """None bbox must not trigger bbox validation."""
        entry = msgspec.convert({**VALID_ENTRY, "bbox": None}, KeyVarConfigEntry)
        assert entry.bbox is None

    # --- depth_range validation ---

    def test_invalid_depth_range_order(self):
        """depth_min must be strictly less than depth_max."""
        with pytest.raises(msgspec.ValidationError, match="depth_min"):
            msgspec.convert(
                {**VALID_ENTRY, "depth_range": [500.0, 0.0]}, KeyVarConfigEntry
            )

    def test_invalid_depth_range_equal(self):
        with pytest.raises(msgspec.ValidationError, match="depth_min"):
            msgspec.convert(
                {**VALID_ENTRY, "depth_range": [100.0, 100.0]}, KeyVarConfigEntry
            )

    def test_depth_range_none_skips_validation(self):
        entry = msgspec.convert({**VALID_ENTRY, "depth_range": None}, KeyVarConfigEntry)
        assert entry.depth_range is None


# ---------------------------------------------------------------------------
# AppConfig
# ---------------------------------------------------------------------------


class TestAppConfig:
    def test_convert_full_config(self):
        """Full config dict deserializes into nested Struct types."""
        raw = {
            "variables": {
                "sst": VALID_ENTRY,
                "chl": {**VALID_ENTRY, "local_folder": "chl"},
            },
            "secrets": {
                "aviso_ftp_server": None,
                "aviso_username": None,
                "aviso_password": None,
            },
        }
        cfg = msgspec.convert(raw, AppConfig)
        assert isinstance(cfg.variables["sst"], KeyVarConfigEntry)
        assert cfg.variables["sst"].local_folder == "sst"
        assert cfg.variables["chl"].local_folder == "chl"
        assert isinstance(cfg.secrets, SecretsConfig)

    def test_secrets_all_optional(self):
        """SecretsConfig fields all default to None."""
        raw = {
            "variables": {"sst": VALID_ENTRY},
            "secrets": {},
        }
        cfg = msgspec.convert(raw, AppConfig)
        assert cfg.secrets.aviso_ftp_server is None
        assert cfg.secrets.aviso_username is None
        assert cfg.secrets.aviso_password is None

    def test_variables_dict_interface(self):
        """variables is a plain dict — supports keys(), values(), items(), []."""
        raw = {
            "variables": {"sst": VALID_ENTRY},
            "secrets": {},
        }
        cfg = msgspec.convert(raw, AppConfig)
        assert list(cfg.variables.keys()) == ["sst"]
        assert len(list(cfg.variables.values())) == 1
        entry = cfg.variables["sst"]
        assert entry.source == "cmems"

    def test_missing_required_field_raises(self):
        """Missing required field (e.g. source) raises msgspec.ValidationError."""
        incomplete = {k: v for k, v in VALID_ENTRY.items() if k != "source"}
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {"variables": {"sst": incomplete}, "secrets": {}},
                AppConfig,
            )
