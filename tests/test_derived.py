"""Tests for processing/derived.py and the derived_vars config entry."""

from pathlib import Path

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from h2mare.models import AppConfig, DerivedVarSpec
from h2mare.processing.derived import apply_derived_vars

REPO = Path(__file__).resolve().parent.parent


def _spec(**kw) -> DerivedVarSpec:
    return msgspec.convert(kw, DerivedVarSpec)


def _grid(values: np.ndarray, name: str = "adt", depth: bool = False) -> xr.DataArray:
    dims = ["time", "depth", "lat", "lon"] if depth else ["time", "lat", "lon"]
    coords = {
        "time": pd.date_range("2020-01-01", periods=values.shape[0], freq="D"),
        "lat": np.arange(values.shape[-2], dtype=float),
        "lon": np.arange(values.shape[-1], dtype=float),
    }
    if depth:
        coords["depth"] = np.arange(values.shape[1], dtype=float)
    return xr.DataArray(values, dims=dims, coords=coords, name=name)


def _noisy(shape, seed=0) -> np.ndarray:
    a = np.random.default_rng(seed).random(shape).astype("float32")
    a[..., 3:6, 4:9] = np.nan  # a coastline-like hole
    return a


class TestRollingStd:
    def test_matches_the_former_ssh_and_sst_code(self):
        """Both hardcoded versions this replaces, bit for bit, holes included —
        so stores written before and after the move agree."""
        da = _grid(_noisy((2, 12, 15))).chunk({"time": 1, "lat": -1, "lon": -1})
        ssh_style = (
            da.rolling(lon=3, lat=3, center=True, min_periods=1)
            .std(skipna=True)
            .compute()
        )
        sst_style = (
            da.rolling(lon=3, lat=3, center=True, min_periods=1)
            .construct(lon="lon_win", lat="lat_win")
            .std(dim=["lon_win", "lat_win"], skipna=True)
            .astype("float32")
            .compute()
        )
        out = apply_derived_vars(
            da.to_dataset(),
            {"adt_std": _spec(op="rolling_std", source="adt")},
            "ssh",
        )["adt_std"].compute()

        assert out.dtype == np.float32
        np.testing.assert_array_equal(out.values, ssh_style.values)
        np.testing.assert_array_equal(out.values, sst_style.values)

    def test_stays_lazy(self):
        da = _grid(_noisy((2, 6, 6))).chunk({"time": 1})
        out = apply_derived_vars(
            da.to_dataset(), {"s": _spec(op="rolling_std", source="adt")}, "ssh"
        )
        assert out["s"].chunks is not None

    def test_keeps_depth_axis_and_windows_each_level_alone(self):
        a = _noisy((1, 2, 6, 6))
        a[:, 1] = 5.0  # a constant level has zero spread wherever it has data
        out = apply_derived_vars(
            _grid(a, "uo", depth=True).to_dataset(),
            {"s": _spec(op="rolling_std", source="uo")},
            "dyn",
        )["s"]
        assert out.dims == ("time", "depth", "lat", "lon")
        np.testing.assert_array_equal(out.isel(depth=1).values, 0.0)

    def test_window_is_honoured(self):
        a = np.zeros((1, 7, 7), dtype="float32")
        a[0, 0, 0] = 1.0
        ds = _grid(a).to_dataset()
        # The spike at (0, 0) is inside a 5-cell window centred on (2, 2) but
        # not a 3-cell one.
        three = apply_derived_vars(
            ds.copy(), {"s": _spec(op="rolling_std", source="adt")}, "k"
        )["s"]
        five = apply_derived_vars(
            ds.copy(), {"s": _spec(op="rolling_std", source="adt", window=5)}, "k"
        )["s"]
        assert float(three[0, 2, 2]) == 0.0
        assert float(five[0, 2, 2]) > 0.0

    def test_refuses_data_without_lon_lat(self):
        ds = xr.Dataset({"a": (["time", "y", "x"], np.ones((1, 3, 3)))})
        with pytest.raises(ValueError, match="lon/lat"):
            apply_derived_vars(ds, {"s": _spec(op="rolling_std", source="a")}, "k")


class TestKineticEnergy:
    def test_is_half_speed_squared(self):
        ds = xr.Dataset(
            {
                "ugos": _grid(np.full((1, 2, 2), 3.0, dtype="float32")),
                "vgos": _grid(np.full((1, 2, 2), 4.0, dtype="float32")),
            }
        )
        out = apply_derived_vars(
            ds,
            {"gke": _spec(op="kinetic_energy", source=["ugos", "vgos"])},
            "ssh",
        )["gke"]
        np.testing.assert_array_equal(out.values, 12.5)
        assert out.dtype == np.float32

    def test_works_per_depth_level(self):
        u = np.stack([np.full((3, 3), 1.0), np.full((3, 3), 2.0)])[None]
        ds = xr.Dataset(
            {
                "uo": _grid(u, "uo", depth=True),
                "vo": _grid(np.zeros_like(u), "vo", depth=True),
            }
        )
        out = apply_derived_vars(
            ds, {"ke": _spec(op="kinetic_energy", source=["uo", "vo"])}, "dyn"
        )["ke"]
        assert out.dims == ("time", "depth", "lat", "lon")
        np.testing.assert_array_equal(out.isel(depth=0).values, 0.5)
        np.testing.assert_array_equal(out.isel(depth=1).values, 2.0)

    def test_land_stays_nan(self):
        u = np.array([[[np.nan, 1.0]]])
        ds = xr.Dataset({"u": _grid(u, "u"), "v": _grid(np.ones_like(u), "v")})
        out = apply_derived_vars(
            ds, {"ke": _spec(op="kinetic_energy", source=["u", "v"])}, "k"
        )["ke"]
        assert np.isnan(out.values[0, 0, 0])
        assert out.values[0, 0, 1] == 1.0


def _uv_depth(depths=(0.5, 47.0, 109.7)) -> xr.Dataset:
    """uo = level index + 1, vo = 0, so ke at level i is (i+1)**2 / 2."""
    u = np.stack([np.full((2, 3), i + 1.0, dtype="float32") for i in range(3)])[None]
    coords = {
        "time": pd.date_range("2020-01-01", periods=1),
        "depth": list(depths),
        "lat": [0.0, 1.0],
        "lon": [0.0, 1.0, 2.0],
    }
    dims = ["time", "depth", "lat", "lon"]
    return xr.Dataset({"uo": (dims, u), "vo": (dims, np.zeros_like(u))}, coords=coords)


class TestDepthSelection:
    def test_writes_one_2d_variable_per_level(self):
        out = apply_derived_vars(
            _uv_depth(),
            {"ke": _spec(op="kinetic_energy", source=["uo", "vo"], depth=[0, 110])},
            "dyn",
        )
        assert "ke" not in out
        for name, expected in [("ke_0", 0.5), ("ke_110", 4.5)]:
            assert out[name].dims == ("time", "lat", "lon")
            assert "depth" not in out[name].coords
            np.testing.assert_array_equal(out[name].values, expected)

    def test_matches_the_nearest_depth_and_keeps_the_requested_name(self):
        """Same rule as depth_levels: ke_50 off the 47 m level."""
        out = apply_derived_vars(
            _uv_depth(),
            {"ke": _spec(op="kinetic_energy", source=["uo", "vo"], depth=[50])},
            "dyn",
        )
        np.testing.assert_array_equal(out["ke_50"].values, 2.0)

    def test_unselected_levels_are_never_computed(self):
        """The point of `depth`: levels not asked for are not even read."""
        import dask.array as dsa
        from dask.delayed import delayed

        def _boom():
            raise AssertionError("an unselected level was computed")

        good = dsa.from_array(np.ones((1, 2, 3), dtype="float32"))
        bad = dsa.from_delayed(delayed(_boom)(), shape=(1, 2, 3), dtype="float32")
        u = dsa.stack([good, bad], axis=1)  # time, depth, lat, lon
        dims = ["time", "depth", "lat", "lon"]
        ds = xr.Dataset(
            {"uo": (dims, u), "vo": (dims, u)},
            coords={"depth": [0.5, 100.0], "lat": [0.0, 1.0], "lon": [0.0, 1.0, 2.0]},
        )
        out = apply_derived_vars(
            ds,
            {"ke": _spec(op="kinetic_energy", source=["uo", "vo"], depth=[0])},
            "dyn",
        )
        np.testing.assert_array_equal(out["ke_0"].values, 1.0)

    def test_rolling_std_at_a_level(self):
        out = apply_derived_vars(
            _uv_depth(),
            {"s": _spec(op="rolling_std", source="uo", depth=[0])},
            "dyn",
        )
        assert out["s_0"].dims == ("time", "lat", "lon")
        np.testing.assert_array_equal(out["s_0"].values, 0.0)

    def test_source_without_depth_axis_is_refused(self):
        ds = xr.Dataset(
            {"u": _grid(np.ones((1, 2, 2))), "v": _grid(np.ones((1, 2, 2)))}
        )
        with pytest.raises(ValueError, match=r"\['u', 'v'\] has no depth axis"):
            apply_derived_vars(
                ds,
                {"ke": _spec(op="kinetic_energy", source=["u", "v"], depth=[0])},
                "k",
            )

    def test_output_passes_through_depth_slicing(self):
        """ke_0 is 2-D, so compile/extract slice uo beside it and leave it be."""
        from h2mare.storage.xarray_helpers import select_depth_levels

        ds = apply_derived_vars(
            _uv_depth(),
            {"ke": _spec(op="kinetic_energy", source=["uo", "vo"], depth=[0])},
            "dyn",
        )
        out = select_depth_levels(ds, {"uo": [0], "vo": [0]}, "dyn")
        assert set(out.data_vars) == {"uo_0", "vo_0", "ke_0"}


class TestApplyDerivedVars:
    def _ds(self):
        return xr.Dataset({"adt": _grid(_noisy((1, 5, 5)))})

    @pytest.mark.parametrize("derived", [None, {}])
    def test_nothing_declared_leaves_the_dataset_alone(self, derived):
        ds = self._ds()
        out = apply_derived_vars(ds.copy(), derived, "k")
        xr.testing.assert_identical(out, ds)

    def test_missing_source_names_the_entry_and_what_is_there(self):
        with pytest.raises(ValueError, match=r"derived_vars\.ke reads \['uo'\].*adt"):
            apply_derived_vars(
                self._ds(),
                {"ke": _spec(op="kinetic_energy", source=["uo", "adt"])},
                "k",
            )

    def test_an_entry_can_read_an_earlier_one(self):
        out = apply_derived_vars(
            self._ds(),
            {
                "adt_std": _spec(op="rolling_std", source="adt"),
                "adt_std_std": _spec(op="rolling_std", source="adt_std"),
            },
            "k",
        )
        assert {"adt_std", "adt_std_std"} <= set(out.data_vars)


class TestDerivedVarSpec:
    def test_window_defaults_to_three(self):
        assert _spec(op="rolling_std", source="adt").window == 3

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"op": "rolling_std", "source": ["a", "b"]}, "reads 1"),
            ({"op": "kinetic_energy", "source": "u"}, "reads 2"),
            ({"op": "kinetic_energy", "source": ["u", "v", "w"]}, "reads 2"),
            ({"op": "rolling_std", "source": "a", "window": 4}, "odd"),
            ({"op": "rolling_std", "source": "a", "window": 0}, "odd"),
            ({"op": "kinetic_energy", "source": ["u", "v"], "window": 3}, "no window"),
            ({"op": "rolling_std", "source": "a", "depth": []}, "empty"),
            ({"op": "rolling_std", "source": "a", "depth": [-5]}, ">= 0"),
            ({"op": "rolling_std", "source": "a", "depth": [0, 0]}, "duplicate"),
        ],
    )
    def test_refuses_malformed_entries(self, kw, match):
        with pytest.raises(msgspec.ValidationError, match=match):
            _spec(**kw)

    def test_refuses_unknown_op(self):
        with pytest.raises(msgspec.ValidationError):
            _spec(op="rolling_mean", source="a")

    def test_refuses_misspelt_field(self):
        with pytest.raises(msgspec.ValidationError, match="windw"):
            _spec(op="rolling_std", source="a", windw=5)

    def test_decodes_inside_a_variable_entry(self):
        cfg = msgspec.convert(
            {
                "variables": {
                    "ssh": {
                        "local_folder": "ssh",
                        "source_vars": ["ugos", "vgos"],
                        "dataset_id_rep": "x",
                        "source": "cmems",
                        "archive_raw": False,
                        "derived_vars": {
                            "gke": {"op": "kinetic_energy", "source": ["ugos", "vgos"]}
                        },
                    }
                },
                "secrets": {},
            },
            AppConfig,
        )
        derived = cfg.variables["ssh"].derived_vars
        assert derived is not None
        assert derived["gke"].sources == ["ugos", "vgos"]


def _dyn_config(**entry) -> AppConfig:
    return msgspec.convert(
        {
            "variables": {
                "dyn": {
                    "local_folder": "dyn",
                    "source_vars": ["uo", "vo"],
                    "dataset_id_rep": "x",
                    "source": "cmems",
                    "archive_raw": False,
                    **entry,
                }
            },
            "secrets": {},
        },
        AppConfig,
    )


_KE_AT_0 = {"op": "kinetic_energy", "source": ["uo", "vo"], "depth": [0]}
_KE_FULL = {"op": "kinetic_energy", "source": ["uo", "vo"]}


class TestDerivedNamesAtLoad:
    def test_depth_entry_beside_depth_levels_for_other_vars_loads(self):
        cfg = _dyn_config(
            depth_levels={"uo": [0]},
            derived_vars={"ke": _KE_AT_0},
            compiled_vars=["uo_0", "ke_0"],
        )
        derived = cfg.variables["dyn"].derived_vars
        assert derived is not None
        assert derived["ke"].output_names("ke") == ["ke_0"]

    def test_full_depth_entry_sliced_by_depth_levels_loads(self):
        _dyn_config(depth_levels={"ke": [0]}, derived_vars={"ke": _KE_FULL})

    @pytest.mark.parametrize("key", ["depth_levels", "extract_depth_levels"])
    def test_depth_levels_cannot_slice_a_depth_entry(self, key):
        with pytest.raises(ValueError, match="stored without a depth axis"):
            _dyn_config(**{key: {"ke": [0]}}, derived_vars={"ke": _KE_AT_0})

    def test_same_column_from_both_sides_is_refused(self):
        with pytest.raises(ValueError, match=r"\['uo_0'\] are written both"):
            _dyn_config(
                depth_levels={"uo": [0]},
                derived_vars={"uo_0": {"op": "rolling_std", "source": "vo"}},
            )

    def test_two_entries_writing_one_name_are_refused(self):
        with pytest.raises(ValueError, match="both write 'ke_0'"):
            _dyn_config(
                derived_vars={
                    "ke": _KE_AT_0,
                    "ke_0": {"op": "rolling_std", "source": "uo"},
                }
            )


class TestRepoConfig:
    """Guard the tracked config.yaml. These layers used to be hardcoded in the
    ssh/sst processors; with the names moved to config, dropping an entry would
    silently stop them being written."""

    @pytest.fixture(scope="class")
    def config(self) -> AppConfig:
        raw = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8"))
        return msgspec.convert(
            {"variables": raw["variables"], "secrets": {}}, AppConfig
        )

    @pytest.mark.parametrize(
        "var_key, expected",
        [
            ("sst", {"sst_std": ("rolling_std", ["sst"], 3)}),
            (
                "ssh",
                {
                    "adt_std": ("rolling_std", ["adt"], 3),
                    "sla_std": ("rolling_std", ["sla"], 3),
                    "gke": ("kinetic_energy", ["ugos", "vgos"], None),
                },
            ),
        ],
    )
    def test_declares_the_formerly_hardcoded_layers(self, config, var_key, expected):
        derived = config.variables[var_key].derived_vars or {}
        got = {n: (s.op.value, s.sources, s.window) for n, s in derived.items()}
        assert got == expected
