"""Tests for msgspec-based data models (KeyVarConfigEntry, AppConfig)."""

from types import SimpleNamespace

import msgspec
import pytest

from h2mare.models import (
    AppConfig,
    KeyVarConfigEntry,
    SecretsConfig,
    depth_column_names,
    depth_levels_for,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "archive_raw": False,
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
        assert entry.store_root is None

    def test_archive_raw_defaults_to_false(self):
        """Optional: omitted means raw files are deleted once converted."""
        entry = {k: v for k, v in VALID_ENTRY.items() if k != "archive_raw"}
        assert msgspec.convert(entry, KeyVarConfigEntry).archive_raw is False

    @pytest.mark.parametrize(
        "root",
        ["/mnt/other_drive", r"D:\data", r"\\server\share"],
        ids=["posix", "windows-drive", "unc"],
    )
    def test_store_root_accepts_an_absolute_path(self, root):
        """
        Both flavours are accepted on either platform: a config written on
        Linux must not be rejected merely for being read on Windows.
        """
        entry = msgspec.convert({**VALID_ENTRY, "store_root": root}, KeyVarConfigEntry)
        assert entry.store_root == root

    def test_relative_store_root_rejected(self):
        """
        A relative root would resolve against the process cwd, so the same
        config would name different drives depending on where it was run.
        """
        with pytest.raises(ValueError, match="store_root must be an absolute path"):
            msgspec.convert(
                {**VALID_ENTRY, "store_root": "some/relative/dir"}, KeyVarConfigEntry
            )

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
        """source_vars field accepts a plain string (not just a list)."""
        entry = msgspec.convert(
            {**VALID_ENTRY, "source_vars": "analysed_sst"}, KeyVarConfigEntry
        )
        assert entry.source_vars == "analysed_sst"

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

    # --- pattern / filename_date_range validation ---

    def test_pattern_optional_defaults_none(self):
        """pattern may be omitted (derived/system vars) and defaults to None."""
        no_pattern = {k: v for k, v in VALID_ENTRY.items() if k != "pattern"}
        entry = msgspec.convert(no_pattern, KeyVarConfigEntry)
        assert entry.pattern is None

    def test_filename_date_range_two_groups_ok(self):
        entry = msgspec.convert(
            {**VALID_ENTRY, "filename_date_range": True, "pattern": r"(\d{8})_(\d{8})"},
            KeyVarConfigEntry,
        )
        assert entry.filename_date_range is True

    def test_filename_date_range_wrong_group_count_raises(self):
        """Range mode unpacks exactly 2 groups — fewer/more must fail at load."""
        with pytest.raises(msgspec.ValidationError, match="exactly 2 capture groups"):
            msgspec.convert(
                {**VALID_ENTRY, "filename_date_range": True, "pattern": r"(\d{8})"},
                KeyVarConfigEntry,
            )

    def test_filename_date_range_without_pattern_raises(self):
        no_pattern = {k: v for k, v in VALID_ENTRY.items() if k != "pattern"}
        with pytest.raises(msgspec.ValidationError, match="with 2 capture groups"):
            msgspec.convert(
                {**no_pattern, "filename_date_range": True}, KeyVarConfigEntry
            )


# ---------------------------------------------------------------------------
# Depth levels
# ---------------------------------------------------------------------------


class TestStaticLayers:
    """compile_layer / extract_layer must name one of the declared layers."""

    _LAYERS = {"15s": "b15.zarr", "0.25deg": "b025.nc"}

    def test_declared_layers_accepted(self):
        entry = msgspec.convert(
            {
                **VALID_ENTRY,
                "layers": self._LAYERS,
                "compile_layer": "0.25deg",
                "extract_layer": "15s",
            },
            KeyVarConfigEntry,
        )
        assert entry.layers == self._LAYERS
        assert entry.extract_layer == "15s"

    @pytest.mark.parametrize("field", ["compile_layer", "extract_layer"])
    def test_undeclared_layer_rejected(self, field):
        with pytest.raises(ValueError, match=f"{field} '60s'"):
            msgspec.convert(
                {**VALID_ENTRY, "layers": self._LAYERS, field: "60s"},
                KeyVarConfigEntry,
            )

    def test_layer_without_layers_rejected(self):
        with pytest.raises(ValueError, match="extract_layer"):
            msgspec.convert({**VALID_ENTRY, "extract_layer": "15s"}, KeyVarConfigEntry)


def _entry(**depth) -> KeyVarConfigEntry:
    return msgspec.convert({**VALID_ENTRY, **depth}, KeyVarConfigEntry)


class TestDepthLevelKeys:
    def test_per_variable_levels_are_accepted(self):
        entry = _entry(depth_levels={"thetao": [0, 50, 100], "uo": [0]})
        assert entry.depth_levels == {"thetao": [0, 50, 100], "uo": [0]}

    @pytest.mark.parametrize(
        "depth",
        [
            {"depth_levels": {"thetao": [0]}, "compile_depth_slices": [0]},
            {"extract_depth_levels": {"thetao": [0]}, "extract_depth_slices": [0]},
        ],
        ids=["compile", "extract"],
    )
    def test_new_and_older_key_together_are_refused(self, depth):
        with pytest.raises(msgspec.ValidationError, match="are both set"):
            _entry(**depth)

    @pytest.mark.parametrize(
        ("depth", "message"),
        [
            ({"depth_levels": {}}, "is empty"),
            ({"depth_levels": {"thetao": []}}, "empty level list"),
            ({"depth_levels": {"thetao": [-5]}}, ">= 0"),
            ({"depth_levels": {"thetao": [0, 0]}}, "duplicate"),
            ({"extract_depth_levels": {"uo": [10, 10]}}, "duplicate"),
            ({"compile_depth_slices": []}, "empty level list"),
            ({"extract_depth_slices": [-1]}, ">= 0"),
        ],
    )
    def test_malformed_levels_are_refused(self, depth, message):
        with pytest.raises(msgspec.ValidationError, match=message):
            _entry(**depth)

    def test_depth_levels_are_distinct_from_depth_range(self):
        """The continuous download band does not imply any levels."""
        assert depth_levels_for("thetao", _entry(depth_range=[0.0, 110.0])) == {}


class TestDepthLevelsFor:
    def test_older_list_form_keys_by_var_key(self):
        """o2/thetao keep their meaning: the store variable is named like the key."""
        entry = _entry(compile_depth_slices=[0, 100])
        assert depth_levels_for("o2", entry) == {"o2": [0, 100]}
        assert depth_levels_for("o2", entry, "extract") == {"o2": [0, 100]}

    def test_older_extract_list_replaces_compile_levels(self):
        entry = _entry(compile_depth_slices=[0, 100, 500], extract_depth_slices=[0])
        assert depth_levels_for("o2", entry, "extract") == {"o2": [0]}
        assert depth_levels_for("o2", entry, "compile") == {"o2": [0, 100, 500]}

    def test_extract_override_is_merged_per_variable(self):
        """Narrowing one field must not drop the others from extraction."""
        entry = _entry(
            depth_levels={"thetao": [0, 50, 100], "so": [0]},
            extract_depth_levels={"thetao": [0]},
        )
        assert depth_levels_for("dyn_rep", entry, "extract") == {
            "thetao": [0],
            "so": [0],
        }
        assert depth_levels_for("dyn_rep", entry) == {
            "thetao": [0, 50, 100],
            "so": [0],
        }

    def test_no_depth_keys_resolve_to_nothing(self):
        assert depth_levels_for("sst", _entry()) == {}
        assert depth_levels_for("sst", _entry(), "extract") == {}

    def test_stand_in_config_without_the_new_fields(self):
        cfg = SimpleNamespace(compile_depth_slices=[100], extract_depth_slices=None)
        assert depth_levels_for("thetao", cfg, "extract") == {"thetao": [100]}

    def test_returned_lists_are_copies(self):
        entry = _entry(depth_levels={"thetao": [0]})
        depth_levels_for("x", entry)["thetao"].append(99)
        assert entry.depth_levels == {"thetao": [0]}

    def test_unknown_purpose_is_refused(self):
        with pytest.raises(ValueError, match="purpose"):
            depth_levels_for("x", _entry(), "convert")

    def test_column_names_follow_the_variable(self):
        assert depth_column_names({"thetao": [0, 50], "uo": [0]}) == [
            "thetao_0",
            "thetao_50",
            "uo_0",
        ]


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

    def _with(self, **entry):
        return {"variables": {"dyn": {**VALID_ENTRY, **entry}}, "secrets": {}}

    def test_depth_columns_missing_from_compiled_vars_are_refused(self):
        raw = self._with(
            depth_levels={"thetao": [0, 50], "uo": [0]},
            compiled_vars=["thetao_0", "thetao_50", "zos"],
        )
        with pytest.raises(msgspec.ValidationError, match=r"\['uo_0'\]"):
            msgspec.convert(raw, AppConfig)

    def test_the_message_gives_the_list_to_write(self):
        """
        The case that prompted it: levels added, compiled_vars left with the
        bare names. "Add them" would have left four names h2ds never holds.
        """
        raw = self._with(
            depth_levels={"thetao": [0, 5], "uo": [0]},
            compiled_vars=["thetao", "uo", "zos", "mlotst"],
        )
        with pytest.raises(msgspec.ValidationError) as err:
            msgspec.convert(raw, AppConfig)

        msg = str(err.value)
        assert "['thetao', 'uo'] are sliced by depth" in msg
        assert "['thetao_0', 'thetao_5', 'uo_0'] are not listed" in msg
        assert "compiled_vars: [thetao_0, thetao_5, uo_0, zos, mlotst]" in msg

    def test_a_bare_sliced_name_is_refused_even_with_its_columns(self):
        raw = self._with(
            depth_levels={"thetao": [0]},
            compiled_vars=["thetao", "thetao_0", "zos"],
        )
        with pytest.raises(msgspec.ValidationError) as err:
            msgspec.convert(raw, AppConfig)

        msg = str(err.value)
        assert "['thetao'] are sliced by depth" in msg
        assert "not listed" not in msg
        assert "compiled_vars: [thetao_0, zos]" in msg

    def test_older_form_is_checked_under_the_var_key_name(self):
        raw = self._with(compile_depth_slices=[100], compiled_vars=["thetao_100"])
        with pytest.raises(msgspec.ValidationError, match=r"\['dyn_100'\]"):
            msgspec.convert(raw, AppConfig)

    def test_regrid_naming_an_unpublished_column_is_refused(self):
        """A typo would leave the real column averaged and look configured."""
        raw = self._with(
            compiled_vars=["ac_track", "ac_dist_km"],
            regrid={"ac_trak": "nearest"},
        )
        with pytest.raises(msgspec.ValidationError, match=r"\['ac_trak'\]"):
            msgspec.convert(raw, AppConfig)

    def test_regrid_accepts_published_columns(self):
        raw = self._with(
            compiled_vars=["ac_track", "ac_dist_km"],
            regrid={"ac_track": "nearest"},
        )
        assert msgspec.convert(raw, AppConfig).variables["dyn"].regrid == {
            "ac_track": "nearest"
        }

    def test_regrid_rejects_an_unknown_method(self):
        raw = self._with(compiled_vars=["ac_track"], regrid={"ac_track": "mean"})
        with pytest.raises(msgspec.ValidationError, match="Invalid enum value 'mean'"):
            msgspec.convert(raw, AppConfig)

    def test_regrid_without_compiled_vars_skips_the_name_check(self):
        raw = self._with(regrid={"anything": "nearest"})
        assert msgspec.convert(raw, AppConfig).variables["dyn"].regrid

    def test_source_renames_onto_a_published_name_is_accepted(self):
        raw = self._with(
            source_renames={"mlotst": "mld"},
            compiled_vars=["mld"],
        )
        assert msgspec.convert(raw, AppConfig).variables["dyn"].source_renames == {
            "mlotst": "mld"
        }

    def test_source_renames_onto_an_unpublished_name_is_refused(self):
        """The map is the link between source_vars and compiled_vars; a target
        nothing publishes means one of the two is wrong."""
        raw = self._with(source_renames={"mlotst": "mdl"}, compiled_vars=["mld"])
        with pytest.raises(msgspec.ValidationError, match=r"\['mdl'\]"):
            msgspec.convert(raw, AppConfig)

    def test_a_renamed_source_left_in_compiled_vars_is_refused(self):
        """compiled_vars keeping the old name names a column no store holds."""
        raw = self._with(
            source_renames={"mlotst": "mld"},
            compiled_vars=["mld", "mlotst"],
        )
        with pytest.raises(msgspec.ValidationError, match=r"\['mlotst'\] ?, which"):
            msgspec.convert(raw, AppConfig)

    def test_source_renames_onto_a_depth_sliced_name_is_accepted(self):
        """A renamed 3-D variable is published as its depth columns."""
        raw = self._with(
            source_renames={"to": "thetao"},
            depth_levels={"thetao": [0]},
            compiled_vars=["thetao_0"],
        )
        assert msgspec.convert(raw, AppConfig).variables["dyn"].source_renames

    def test_a_self_mapping_rename_is_refused(self):
        raw = self._with(source_renames={"mld": "mld"})
        with pytest.raises(msgspec.ValidationError, match="onto itself"):
            msgspec.convert(raw, AppConfig)

    def test_two_sources_onto_one_name_are_refused(self):
        raw = self._with(source_renames={"mlotst": "mld", "mlotst_cvg": "mld"})
        with pytest.raises(msgspec.ValidationError, match=r"more than one source"):
            msgspec.convert(raw, AppConfig)

    def test_a_rename_colliding_with_a_derived_layer_is_refused(self):
        raw = self._with(
            source_renames={"mld_stdev": "mld_std"},
            derived_vars={"mld_std": {"op": "rolling_std", "source": "mld"}},
        )
        with pytest.raises(msgspec.ValidationError, match=r"\['mld_std'\]"):
            msgspec.convert(raw, AppConfig)

    def test_source_renames_without_compiled_vars_skips_the_name_check(self):
        raw = self._with(source_renames={"mlotst": "anything"})
        assert msgspec.convert(raw, AppConfig).variables["dyn"].source_renames

    def test_consistent_depth_columns_are_accepted(self):
        raw = self._with(
            depth_levels={"thetao": [0, 50]},
            compiled_vars=["thetao_0", "thetao_50", "zos", "mlotst"],
        )
        assert msgspec.convert(raw, AppConfig).variables["dyn"].depth_levels

    def test_undeclared_compiled_vars_skip_the_check(self):
        """compiled_vars: None means 'not yet declared', not 'publishes nothing'."""
        raw = self._with(depth_levels={"thetao": [0]})
        assert msgspec.convert(raw, AppConfig).variables["dyn"].compiled_vars is None

    def test_extract_only_levels_need_not_be_compiled(self):
        raw = self._with(
            depth_levels={"thetao": [0]},
            extract_depth_levels={"uo": [0]},
            compiled_vars=["thetao_0"],
        )
        assert msgspec.convert(raw, AppConfig)

    def test_missing_required_field_raises(self):
        """Missing required field (e.g. source) raises msgspec.ValidationError."""
        incomplete = {k: v for k, v in VALID_ENTRY.items() if k != "source"}
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {"variables": {"sst": incomplete}, "secrets": {}},
                AppConfig,
            )
