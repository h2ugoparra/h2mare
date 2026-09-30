"""
CF compliance of the metadata table in ``config.yaml``.

These check the table itself, not the code that applies it — that is
``test_xarray_helpers.py::TestApplyCfAttrs``. The distinction matters because a
unit string is only ever wrong in config: no amount of correct plumbing makes
``degrees Celsius`` mean a temperature.

The repo's ``config.yaml`` is read by path rather than through ``get_settings``.
Settings resolve through ``H2MARE_ROOT``, so what ``get_settings`` returns is
whatever config the machine has deployed — on a developer's box that is a
different file, and on CI it is none. The checked-in table is the one this repo
is responsible for.

Units are parsed with **udunits2** (via ``cf-units``), which is the parser CF
itself defers to. Parsing alone is not enough: udunits accepts ``degrees
Celsius`` and resolves it to ``0.0174 K.rad``, an angle times a temperature,
because a space means multiplication. So every ``standard_name`` is also checked
for whether its canonical units are *convertible* to the ones declared, which is
what catches that class of error.

Standard names are checked against a vendored snapshot of the CF table — see
``scripts/refresh_cf_standard_names.py`` for why and how to regenerate it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cf_units
import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config.yaml"
CF_NAMES = REPO / "tests" / "fixtures" / "cf_standard_names.json"

#: CF Appendix C modifiers, appended to a standard name after a space. A
#: modified name keeps the base quantity's units, except for the counts.
_MODIFIERS = {
    "detection_minimum",
    "number_of_observations",
    "standard_error",
    "status_flag",
}


@pytest.fixture(scope="module")
def variable_attrs() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["variable_attrs"]


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canonical_units() -> dict:
    return json.loads(CF_NAMES.read_text(encoding="utf-8"))["canonical_units"]


def _ids(attrs: dict) -> list[str]:
    return sorted(attrs)


class TestUnits:
    def test_every_units_string_parses_under_udunits(self, variable_attrs):
        """
        gke declared m-2.s-2 for a specific kinetic energy and tcc declared
        '(0-1)', which is not a unit at all.
        """
        bad = {}
        for var in _ids(variable_attrs):
            declared = variable_attrs[var].get("units")
            if declared is None:
                continue
            try:
                cf_units.Unit(str(declared))
            except Exception as exc:
                bad[var] = f"{declared!r}: {exc}"
        assert not bad, f"units that udunits2 rejects: {bad}"

    def test_a_label_variable_may_have_no_units(self, variable_attrs):
        """
        The eddy track ids are ordinal labels; they used to declare 'ordinal',
        which does not parse. CF allows a label variable to carry no units, and
        that is the honest form.
        """
        for var in ("ac_track", "c_track"):
            assert "units" not in variable_attrs[var]


class TestStandardNames:
    def test_every_standard_name_is_a_current_cf_name(
        self, variable_attrs, canonical_units
    ):
        """
        A name absent from the snapshot has either been mistyped or never
        verified. Regenerate with scripts/refresh_cf_standard_names.py, which
        refuses names the published table does not have and names it has
        deprecated.
        """
        unknown = {}
        for var in _ids(variable_attrs):
            declared = variable_attrs[var].get("standard_name")
            if declared is None:
                continue
            base, _, modifier = str(declared).partition(" ")
            if modifier and modifier not in _MODIFIERS:
                unknown[var] = f"{modifier!r} is not a CF modifier"
            elif base not in canonical_units:
                unknown[var] = f"{base!r} is not in the verified CF snapshot"
        assert not unknown, f"standard_name problems: {unknown}"

    def test_declared_units_are_convertible_to_the_canonical_ones(
        self, variable_attrs, canonical_units
    ):
        """
        The check that catches a unit which parses but means something else.
        tp with standard_name precipitation_amount would fail here: its
        canonical kg m-2 is a mass per area and tp is a depth in mm.
        """
        clashes = {}
        for var in _ids(variable_attrs):
            info = variable_attrs[var]
            declared, units = info.get("standard_name"), info.get("units")
            if declared is None or units is None:
                continue
            base = str(declared).partition(" ")[0]
            canonical = canonical_units.get(base)
            if not canonical:
                continue
            if not cf_units.Unit(str(units)).is_convertible(cf_units.Unit(canonical)):
                clashes[var] = f"{units!r} vs canonical {canonical!r} of {base}"
        assert not clashes, f"units incompatible with their standard_name: {clashes}"

    @pytest.mark.parametrize(
        "var", ["sst", "sst_std", "thetao", "thetao_100", "analysis_error"]
    )
    def test_a_temperature_converts_on_the_right_scale(self, variable_attrs, var):
        """
        Regression on 'degrees Celsius', and the reason convertibility alone is
        not enough to catch it. A space is multiplication in udunits, so it
        resolves to 0.0174 K.rad — and because radians are *dimensionless*,
        udunits happily reports that as convertible to K. What separates the two
        is the offset: a real Celsius scale sends 0 to 273.15 K, while the
        angle-times-temperature reading sends it to 0.
        """
        declared = variable_attrs[var]["units"]
        unit = cf_units.Unit(declared)
        accepted = (cf_units.Unit("K"), cf_units.Unit("degree_C"))
        assert any(unit == ok for ok in accepted), (
            f"{var} declares {declared!r}, which is neither kelvin nor a Celsius "
            f"scale. Allowing 'convertible to K' is not enough here: "
            f"'degrees Celsius' resolves to 0.0174 K.rad, and radians are "
            f"dimensionless, so udunits calls that convertible to K too."
        )


class TestCellMethods:
    def test_cell_methods_name_a_dimension_and_a_method(self, variable_attrs):
        malformed = {
            var: variable_attrs[var]["cell_methods"]
            for var in _ids(variable_attrs)
            if variable_attrs[var].get("cell_methods")
            and ":" not in str(variable_attrs[var]["cell_methods"])
        }
        assert not malformed, f"cell_methods must read 'dim: method': {malformed}"

    def test_the_rolling_window_features_claim_no_cell_method(self, variable_attrs):
        """
        Their cell is not the day the axis labels them with, so a time: method
        would misdescribe rather than under-describe them. The comment carries
        the window instead.
        """
        for var in ("ekman_7d", "ekman_anom", "n_upwell_events_7d"):
            assert "cell_methods" not in variable_attrs[var]
            assert variable_attrs[var]["comment"]


class TestCoverage:
    def test_every_compiled_variable_has_an_entry(self, config):
        """
        A variable with no entry reaches the store with whatever attrs xarray
        inherited, which for anything derived is none at all.
        """
        attrs = config["variable_attrs"]
        missing = sorted(
            f"{key}:{name}"
            for key, var_config in config["variables"].items()
            for name in (var_config.get("compiled_vars") or [])
            if name.strip() and name.strip() not in attrs
        )
        assert not missing, f"compiled_vars with no variable_attrs entry: {missing}"

    def test_every_entry_has_a_long_name(self, config):
        """The one attribute the plotting helpers read, and the CF fallback when
        there is no standard_name."""
        missing = [
            var
            for var, info in sorted(config["variable_attrs"].items())
            if not info.get("long_name")
        ]
        assert not missing, f"entries with no long_name: {missing}"


class TestNativeOverrides:
    def test_overrides_name_real_var_keys_and_real_variables(self, config):
        attrs = config["variable_attrs"]
        problems = []
        for var_key, per_var in config.get("native_attr_overrides", {}).items():
            if var_key not in config["variables"]:
                problems.append(f"{var_key} is not a configured var_key")
                continue
            for name in per_var:
                if name not in attrs:
                    problems.append(f"{var_key}:{name} has no variable_attrs entry")
        assert not problems, f"native_attr_overrides problems: {problems}"

    def test_an_overridden_unit_still_parses(self, config):
        """msl is Pa natively and hPa in h2ds; both have to be real units."""
        bad = {}
        for var_key, per_var in config.get("native_attr_overrides", {}).items():
            for name, override in per_var.items():
                units = override.get("units")
                if units is None:
                    continue
                try:
                    cf_units.Unit(str(units))
                except Exception as exc:
                    bad[f"{var_key}:{name}"] = f"{units!r}: {exc}"
        assert not bad, f"native override units that udunits2 rejects: {bad}"

    def test_an_override_only_touches_attributes_the_table_defines(self, config):
        """
        A null override removes an attribute. Naming one the table does not set
        removes nothing, which reads as an instruction that quietly does not
        apply — most likely a typo for one that does.
        """
        attrs = config["variable_attrs"]
        stray = []
        for var_key, per_var in config.get("native_attr_overrides", {}).items():
            for name, override in per_var.items():
                for key, value in override.items():
                    if value is None and key not in attrs.get(name, {}):
                        stray.append(f"{var_key}:{name}:{key}")
        assert not stray, f"overrides removing an attribute nothing sets: {stray}"


class TestGlobalAttrs:
    def test_the_facts_about_a_file_are_not_declared_in_config(self, config):
        """
        They are computed per file by provenance.refresh_root_attrs. A value
        here would be a second, stale source — config carried
        time_coverage_resolution: P1D long after four stores went hourly.
        """
        computed = {
            "Conventions",
            "product_version",
            "history",
            "time_coverage_start",
            "time_coverage_end",
            "time_coverage_duration",
            "time_coverage_resolution",
            "geospatial_lat_min",
            "geospatial_lat_max",
            "geospatial_lon_min",
            "geospatial_lon_max",
        }
        declared = computed & set(config["global_attrs"])
        assert not declared, (
            f"global_attrs declares what is computed per file: {sorted(declared)}"
        )

    def test_the_acdd_fields_a_reader_needs_are_present(self, config):
        required = ("title", "summary", "keywords", "creator_name", "license", "source")
        missing = [key for key in required if not config["global_attrs"].get(key)]
        assert not missing, f"global_attrs missing: {missing}"


class TestFrontLayers:
    """
    The detection threshold lives in ``boa_fronts`` and is quoted in the
    layer's ``comment``. Nothing makes the prose follow the number, so a
    threshold changed in one place and not the other would leave every file
    written afterwards describing itself wrongly.
    """

    def _declared(self, config) -> list[tuple[str, str, float]]:
        return [
            (var_key, name, spec["threshold"])
            for var_key, entry in config["variables"].items()
            for name, spec in (entry.get("boa_fronts") or {}).items()
        ]

    def test_the_layers_are_in_the_table(self, config):
        missing = [
            f"{var_key}:{name}"
            for var_key, name, _ in self._declared(config)
            if name not in config["variable_attrs"]
        ]
        assert self._declared(config), "no boa_fronts entries left to check"
        assert not missing, f"front layers with no variable_attrs entry: {missing}"

    def test_each_comment_states_the_configured_threshold(self, config):
        attrs = config["variable_attrs"]
        drifted = {}
        for var_key, name, threshold in self._declared(config):
            comment = attrs.get(name, {}).get("comment", "")
            if str(threshold) not in comment:
                drifted[f"{var_key}:{name}"] = (
                    f"threshold {threshold} not in {comment!r}"
                )
        assert not drifted, f"comments that no longer state their threshold: {drifted}"

    def test_each_comment_names_the_detection_algorithm(self, config):
        """The threshold means nothing without it: it is a BOA gradient cut."""
        attrs = config["variable_attrs"]
        unnamed = [
            f"{var_key}:{name}"
            for var_key, name, _ in self._declared(config)
            if "belkin" not in attrs.get(name, {}).get("comment", "").lower()
        ]
        assert not unnamed, f"front layers whose comment omits the algorithm: {unnamed}"
