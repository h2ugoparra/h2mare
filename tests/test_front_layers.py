"""Tests for processing/core/front_layers.py and the front_layers config entry.

The design and the evidence behind each rule are in plans/front-layers.md; the
section numbers below refer to it.
"""

from unittest.mock import patch

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import AppConfig, FrontLayerSpec, native_only_vars
from h2mare.processing.core.front_layers import (
    RollingFrequency,
    apply_front_layers,
    detect_fronts,
    front_frequency,
    hysteresis,
    metric_gradient,
    proximity_mask,
    recompute_following_frequency,
    smooth_km,
)
from h2mare.types import DateRange

LAT = np.arange(40.0, 42.0, 0.05) + 0.025
LON = np.arange(-20.0, -18.0, 0.05) + 0.025
PARAMS = dict(transform="none", sigma_km=5.0, low_per_km=0.0155, high_per_km=0.0299)


def _uniform(value=18.0):
    return np.full((LAT.size, LON.size), value)


def _step(cold=15.0, warm=18.0, at=20):
    f = np.full((LAT.size, LON.size), cold)
    f[:, at:] = warm
    return f


# ---------------------------------------------------------------------------
# Detector (§3.2)
# ---------------------------------------------------------------------------


class TestDetector:
    def test_a_coastline_is_not_a_front(self):
        """Gaps are filled from the nearest valid cell, not with a constant."""
        field = _uniform()
        field[:, :10] = np.nan
        mask, grad = detect_fronts(field, LAT, LON, **PARAMS)

        assert np.nansum(mask) == 0
        assert np.isnan(mask[:, :10]).all()  # land is not assessed
        assert np.isnan(grad[:, :10]).all()

    def test_a_step_gives_one_thin_line_at_the_step(self):
        mask, _ = detect_fronts(_step(at=20), LAT, LON, **PARAMS)

        cols = np.unique(np.nonzero(mask == 1)[1])
        assert set(cols) <= {19, 20}  # thinned to the ridge, a tie is 2 px
        rows = np.unique(np.nonzero(mask == 1)[0])
        assert len(rows) == LAT.size  # continuous along the front

    def test_a_uniform_field_has_no_fronts(self):
        mask, grad = detect_fronts(_uniform(), LAT, LON, **PARAMS)
        assert np.nansum(mask) == 0
        assert np.nanmax(grad) == pytest.approx(0.0, abs=1e-9)

    def test_a_day_without_data_is_not_assessed(self):
        mask, grad = detect_fronts(
            np.full((LAT.size, LON.size), np.nan), LAT, LON, **PARAMS
        )
        assert np.isnan(mask).all() and np.isnan(grad).all()

    def test_low_confidence_pixels_are_never_fronts(self):
        conf = np.zeros((LAT.size, LON.size))
        conf[:, 15:25] = 2.0  # the step sits here
        mask, _ = detect_fronts(
            _step(at=20), LAT, LON, confidence=conf, confidence_max=0.84, **PARAMS
        )
        assert np.isnan(mask[:, 15:25]).all()
        assert np.nansum(mask) == 0

    def test_log10_makes_a_relative_step_count_the_same_at_any_level(self):
        """chl gradients otherwise scale with concentration (§1.3)."""
        low = detect_fronts(
            _step(0.1, 0.2), LAT, LON, **{**PARAMS, "transform": "log10"}
        )[1]
        high = detect_fronts(
            _step(1.0, 2.0), LAT, LON, **{**PARAMS, "transform": "log10"}
        )[1]
        np.testing.assert_allclose(np.nanmax(low), np.nanmax(high), rtol=1e-6)


class TestHysteresis:
    def test_a_weak_ridge_touching_a_strong_one_is_kept(self):
        mag = np.zeros((5, 10))
        mag[2, 1:9] = 1.0  # weak ridge
        mag[2, 8] = 3.0  # strong end
        thin = mag > 0
        out = hysteresis(mag, thin, low=0.5, high=2.0)
        assert out[2, 1:9].all()

    def test_an_isolated_weak_ridge_is_dropped(self):
        mag = np.zeros((5, 10))
        mag[2, 1:5] = 1.0
        mag[2, 7] = 3.0  # strong, but not connected
        out = hysteresis(mag, mag > 0, low=0.5, high=2.0)
        assert not out[2, 1:5].any() and out[2, 7]


class TestMetricUnits:
    """The Sobel ran in index space: an east-west gradient counted cos(lat) as
    strong as the same north-south one (§1.3)."""

    lat = np.arange(59.0, 61.0, 0.05)
    lon = np.arange(-20.0, -18.0, 0.05)

    def _ramps(self, g=0.02):
        dy = 0.05 * 111.2
        dx = 0.05 * 111.2 * np.cos(np.deg2rad(self.lat))[:, None]
        y_km = np.arange(self.lat.size)[:, None] * dy + 0 * self.lon
        x_km = np.cumsum(np.broadcast_to(dx, (self.lat.size, self.lon.size)), axis=1)
        return g * x_km, g * y_km

    def test_equal_gradients_have_equal_magnitude_at_60n(self):
        east_west, north_south = self._ramps()
        mag_ew = metric_gradient(east_west, self.lat, self.lon)[0][5:-5, 5:-5]
        mag_ns = metric_gradient(north_south, self.lat, self.lon)[0][5:-5, 5:-5]
        np.testing.assert_allclose(mag_ew, 0.02, rtol=1e-3)
        np.testing.assert_allclose(mag_ns, 0.02, rtol=1e-3)

    def test_sigma_is_the_same_width_in_km_east_west_at_0n_and_60n(self):
        widths = []
        for lat0 in (0.0, 60.0):
            lat = np.arange(lat0 - 1, lat0 + 1, 0.05)
            lon = np.arange(-20.0, -10.0, 0.05)
            f = np.zeros((lat.size, lon.size))
            f[:, lon.size // 2] = 1.0
            row = smooth_km(f, lat, lon, 10.0)[lat.size // 2]
            x_km = (
                (np.arange(lon.size) - lon.size // 2)
                * 0.05
                * 111.2
                * np.cos(np.deg2rad(lat[lat.size // 2]))
            )
            widths.append(np.sqrt((row * x_km**2).sum() / row.sum()))
        assert widths[0] == pytest.approx(10.0, rel=0.05)
        assert widths[1] == pytest.approx(10.0, rel=0.05)


# ---------------------------------------------------------------------------
# Frequency (§4.3, §4.4)
# ---------------------------------------------------------------------------


class TestFrequency:
    def test_fronts_over_assessed_days(self):
        masks = np.array([[[1.0]], [[0.0]], [[1.0]], [[0.0]]])
        assert front_frequency(masks, 2)[0, 0] == pytest.approx(0.5)

    def test_a_masked_day_does_not_lower_the_frequency(self):
        """Regression (prototype v1): masked days counted as "no front", which
        punched square holes where the confidence mask applied."""
        masks = np.array([[[1.0]], [[np.nan]], [[1.0]], [[np.nan]]])
        assert front_frequency(masks, 2)[0, 0] == pytest.approx(1.0)

    def test_too_few_assessed_days_is_nan(self):
        masks = np.array([[[1.0]], [[np.nan]], [[np.nan]], [[np.nan]]])
        assert np.isnan(front_frequency(masks, 2)[0, 0])

    def test_proximity_is_a_box_in_km(self):
        mask = np.zeros((LAT.size, LON.size), "float32")
        mask[20, 20] = 1.0
        near = proximity_mask(mask, LAT, LON, 12.5)
        rows, cols = np.nonzero(near == 1)
        # 12.5 km is ~2 cells north-south (5.56 km) and ~3 east-west at 41N (4.2 km)
        assert rows.min() == 18 and rows.max() == 22
        assert cols.min() == 17 and cols.max() == 23

    def test_windows_are_calendar_days(self):
        """A day missing from the axis is not assessed; it does not stretch the
        window back over older days."""
        rolling = RollingFrequency([4], LAT[:2], LON[:2], 0.0)
        one, zero = np.ones((2, 2), "float32"), np.zeros((2, 2), "float32")
        for d, m in (("2020-01-01", one), ("2020-01-02", one), ("2020-01-05", zero)):
            rolling.push(pd.Timestamp(d), m)
        # window of 4 ending Jan 5 = Jan 2..5: Jan 2 (front) and Jan 5 (none)
        assert rolling.frequency(4)[0, 0] == pytest.approx(0.5)

    def test_days_must_arrive_in_order(self):
        rolling = RollingFrequency([4], LAT[:2], LON[:2], 0.0)
        rolling.push(pd.Timestamp("2020-01-02"), np.ones((2, 2)))
        with pytest.raises(ValueError, match="increasing order"):
            rolling.push(pd.Timestamp("2020-01-01"), np.ones((2, 2)))


# ---------------------------------------------------------------------------
# Convert step and history (§6.3)
# ---------------------------------------------------------------------------

SPEC = msgspec.convert(
    {
        "source": "sst",
        **{k: v for k, v in PARAMS.items() if k != "transform"},
        "frequency_days": [4],
        "frequency_radius_km": 0.0,
    },
    FrontLayerSpec,
)


def _sst(days: pd.DatetimeIndex, seed=0) -> xr.Dataset:
    """A step that moves one column a day, so each day's masks differ."""
    fields = np.stack([_step(at=15 + (i + seed) % 10) for i in range(len(days))])
    return xr.Dataset(
        {"sst": (("time", "lat", "lon"), fields)},
        coords={"time": days, "lat": LAT, "lon": LON},
    )


class _Store:
    """Stands in for a var_key's store: the MaskReader over layers written so far."""

    def __init__(self):
        self.ds: xr.Dataset | None = None

    def write(self, ds: xr.Dataset) -> None:
        ds = ds.load()
        if self.ds is None:
            self.ds = ds
            return
        keep = self.ds.sel(time=~self.ds.time.isin(ds.time))
        self.ds = xr.concat([keep, ds], dim="time").sortby("time")

    def read(self, var, start, end):
        if self.ds is None or var not in self.ds:
            return None
        out = self.ds[var].sel(time=slice(start, end))
        return out if out.sizes["time"] else None


@pytest.mark.usefixtures("interim_dir", "serial_pool")
class TestHistory:
    days = pd.date_range("2020-01-01", "2020-01-20", freq="D")

    def test_outputs_land_under_their_names(self):
        out = apply_front_layers(_sst(self.days), {"sst": SPEC}, "sst")
        assert {"sst_front", "sst_grad", "sst_ffreq4"} <= set(out.data_vars)
        assert out["sst_front"].sizes == out["sst"].sizes

    def test_seeding_across_a_period_boundary_equals_one_run(self):
        whole = apply_front_layers(_sst(self.days), {"sst": SPEC}, "sst").load()

        store = _Store()
        first, second = self.days[:10], self.days[10:]
        store.write(apply_front_layers(_sst(first), {"sst": SPEC}, "sst"))
        split = apply_front_layers(
            _sst(second, seed=10), {"sst": SPEC}, "sst", read_masks=store.read
        ).load()

        np.testing.assert_array_equal(
            split["sst_ffreq4"].values, whole["sst_ffreq4"].sel(time=second).values
        )

    def test_an_unseeded_start_is_nan_until_half_a_window_is_assessed(self):
        out = apply_front_layers(_sst(self.days), {"sst": SPEC}, "sst").load()
        freq = out["sst_ffreq4"]
        assert np.isnan(freq.isel(time=0)).all()  # 1 assessed day of 4
        assert np.isfinite(freq.isel(time=1)).any()  # 2 of 4

    def test_rewriting_a_window_refreshes_the_following_days(self):
        """REP replacing NRT: the frequency after the rewritten days must match
        a from-scratch run over the new data (§6.3)."""
        store = _Store()
        store.write(apply_front_layers(_sst(self.days), {"sst": SPEC}, "sst"))

        rewritten = self.days[5:10]
        new = apply_front_layers(
            _sst(rewritten, seed=3), {"sst": SPEC}, "sst", read_masks=store.read
        )
        store.write(new)
        refreshed = recompute_following_frequency(
            {"sst": SPEC}, store.read, rewritten[-1]
        )
        store.write(refreshed)

        truth_fields = _sst(self.days).copy()
        truth_fields["sst"].loc[dict(time=rewritten)] = _sst(rewritten, seed=3)[
            "sst"
        ].values
        truth = apply_front_layers(truth_fields, {"sst": SPEC}, "sst").load()

        after = self.days[10:13]  # the window's length minus one
        assert set(pd.DatetimeIndex(refreshed.time.values)) == set(after)
        np.testing.assert_array_equal(
            store.ds["sst_ffreq4"].sel(time=after).values,
            truth["sst_ffreq4"].sel(time=after).values,
        )

    def test_nothing_follows_the_end_of_the_store(self):
        store = _Store()
        store.write(apply_front_layers(_sst(self.days), {"sst": SPEC}, "sst"))
        assert (
            recompute_following_frequency({"sst": SPEC}, store.read, self.days[-1])
            is None
        )

    def test_a_missing_source_is_named(self):
        spec = msgspec.structs.replace(SPEC, source="chl")
        with pytest.raises(ValueError, match=r"reads \['chl'\]"):
            apply_front_layers(_sst(self.days), {"x": spec}, "sst")

    def test_a_missing_confidence_variable_is_named(self):
        spec = msgspec.convert(
            {
                **msgspec.to_builtins(SPEC),
                "confidence": {"var": "analysis_error", "max": 0.84},
            },
            FrontLayerSpec,
        )
        with pytest.raises(ValueError, match="analysis_error"):
            apply_front_layers(_sst(self.days), {"sst": spec}, "sst")


# ---------------------------------------------------------------------------
# Config (§6.4)
# ---------------------------------------------------------------------------

_LAYER = {
    "source": "sst",
    "sigma_km": 5.0,
    "low_per_km": 0.0155,
    "high_per_km": 0.0299,
}


def _load(**kw):
    entry = {
        "local_folder": "CMEMS_SST",
        "source_vars": ["analysed_sst"],
        "dataset_id_rep": "x",
        "source": "cmems",
        **kw,
    }
    return msgspec.convert({"variables": {"sst": entry}, "secrets": {}}, AppConfig)


class TestConfig:
    def test_a_daily_store_loads(self):
        cfg = _load(front_layers={"sst": _LAYER})
        spec = cfg.variables["sst"].front_layers["sst"]
        assert spec.frequency_days == [30] and spec.frequency_radius_km == 12.5

    @pytest.mark.parametrize(
        "change, match",
        [
            ({"transform": "ln"}, "transform"),
            ({"sigma_km": 0}, "sigma_km"),
            ({"low_per_km": 0.05}, "low_per_km < high_per_km"),
            ({"frequency_days": [1]}, "frequency_days"),
            ({"frequency_days": [30, 30]}, "repeats"),
            ({"frequency_radius_km": -1}, "frequency_radius_km"),
            ({"confidence": {"var": "e", "max": 0}}, "confidence max"),
            ({"typo": 1}, "typo"),
        ],
    )
    def test_malformed_entries_are_refused(self, change, match):
        with pytest.raises(msgspec.ValidationError, match=match):
            _load(front_layers={"sst": {**_LAYER, **change}})

    def test_an_hourly_store_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="needs a daily store"):
            _load(time_step="hourly", front_layers={"sst": _LAYER})

    def test_a_name_another_mechanism_writes_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="both by front_layers"):
            _load(
                derived_vars={"sst_grad": {"op": "rolling_std", "source": "sst"}},
                front_layers={"sst": _LAYER},
            )

    def test_published_outputs_must_be_in_compiled_vars(self):
        with pytest.raises(msgspec.ValidationError, match=r"add them to compiled_vars"):
            _load(compiled_vars=["sst", "sst_grad"], front_layers={"sst": _LAYER})

    def test_the_mask_may_not_be_compiled(self):
        with pytest.raises(msgspec.ValidationError, match="never compiled"):
            _load(
                compiled_vars=["sst", "sst_grad", "sst_ffreq30", "sst_front"],
                front_layers={"sst": _LAYER},
            )

    def test_a_rename_onto_an_output_is_refused(self):
        with pytest.raises(msgspec.ValidationError, match="front_layers also write"):
            _load(
                source_renames={"analysed_sst": "sst_grad"},
                front_layers={"sst": _LAYER},
            )

    def test_native_only_vars_names_the_mask(self):
        cfg = _load(front_layers={"sst": _LAYER})
        assert native_only_vars(cfg.variables["sst"]) == {"sst_front"}


class TestCompileDropsTheMask:
    def test_the_daily_mask_does_not_reach_h2ds(self, tmp_path):
        from h2mare.processing.compiler import Compiler

        cfg = msgspec.convert(
            {
                "variables": {
                    "h2ds": {
                        "local_folder": "h2ds",
                        "source_vars": [],
                        "dataset_id_rep": "h2ds",
                        "source": "h2mare",
                        "bbox": (-20, 40, -18, 42),
                    },
                    "sst": {
                        "local_folder": "sst",
                        "source_vars": ["analysed_sst"],
                        "dataset_id_rep": "x",
                        "source": "cmems",
                        "front_layers": {"sst": _LAYER},
                    },
                },
                "secrets": {},
            },
            AppConfig,
        )
        with patch("h2mare.processing.compiler.ZarrCatalog"):
            compiler = Compiler(
                var_key="h2ds",
                app_config=cfg,
                remote_store_root=tmp_path / "r",
                local_store_root=tmp_path / "l",
            )
        produced = xr.Dataset(
            {
                v: (("time",), [1.0])
                for v in ("sst", "sst_front", "sst_grad", "sst_ffreq30")
            },
            coords={"time": [pd.Timestamp("2020-01-01")]},
        )
        with (
            patch(
                "h2mare.processing.compiler_registry.COMPILE_PROCESSORS",
                {"sst": lambda *a: produced},
            ),
            patch("h2mare.processing.compiler.ZarrCatalog"),
            patch.object(compiler, "_has_overlap", return_value=True),
        ):
            out = compiler._process_variable(
                "sst", DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"))
            )

        assert "sst_front" not in out.data_vars
        assert {"sst", "sst_grad", "sst_ffreq30"} <= set(out.data_vars)


class TestRepoConfig:
    """The shipped values are the ones plans/front-layers.md §0 adopts, with the
    evidence for each; change them together."""

    @pytest.fixture(scope="class")
    def cfg(self):
        import pathlib

        import yaml

        path = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return msgspec.convert(
            {"variables": data["variables"], "secrets": {}}, AppConfig
        )

    @pytest.mark.parametrize(
        "var_key, expected",
        [
            (
                "sst",
                dict(
                    transform="none",
                    sigma_km=5.0,
                    low_per_km=0.0155,
                    high_per_km=0.0299,
                    frequency_days=[30],
                    frequency_radius_km=12.5,
                ),
            ),
            (
                "chl",
                dict(
                    transform="log10",
                    sigma_km=7.0,
                    low_per_km=0.0035,
                    high_per_km=0.0070,
                    frequency_days=[30],
                    frequency_radius_km=12.5,
                ),
            ),
        ],
    )
    def test_shipped_values_match_the_plan(self, cfg, var_key, expected):
        spec = cfg.variables[var_key].front_layers[var_key]
        for field, value in expected.items():
            assert getattr(spec, field) == value, field

    def test_sst_is_masked_by_its_analysis_error(self, cfg):
        conf = cfg.variables["sst"].front_layers["sst"].confidence
        assert (conf.var, conf.max) == ("analysis_error", 1.52)
        assert cfg.variables["chl"].front_layers["chl"].confidence is None

    @pytest.mark.parametrize("var_key", ["sst", "chl"])
    def test_the_persistent_distance_is_published_at_half_the_days(self, cfg, var_key):
        """§4.4: 0.5 clears chance recurrence and keeps it a distance."""
        entry = cfg.variables[var_key]
        pd_ = entry.front_layers[var_key].persistent_distance
        assert (pd_.window, pd_.min_frequency) == (30, 0.5)
        assert f"{var_key}_pdist30" in entry.compiled_vars


class TestReadRetry:
    """The store's drive returns a transient EINVAL under load (§6.6)."""

    @pytest.fixture(autouse=True)
    def _no_wait(self, monkeypatch):
        from tenacity import wait_none

        monkeypatch.setattr(
            "h2mare.processing.core.front_layers.READ_WAIT", wait_none()
        )

    def test_a_transient_oserror_is_retried(self):
        from h2mare.processing.core.front_layers import read_retrying

        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise OSError(22, "Invalid argument")
            return "ok"

        assert read_retrying(flaky, "x") == "ok"
        assert len(calls) == 2

    def test_a_missing_file_is_not_waited_for(self):
        from h2mare.processing.core.front_layers import read_retrying

        calls = []

        def missing():
            calls.append(1)
            raise FileNotFoundError("gone")

        with pytest.raises(FileNotFoundError):
            read_retrying(missing, "x")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Distance to persistent fronts (§4.4, optional)
# ---------------------------------------------------------------------------


def _brute_distance(freq, threshold):
    """Every pixel to every zone pixel: what the edge-only tree must equal."""
    from h2mare.utils.spatial import haversine_min_distance_kdtree

    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    zone = np.isfinite(freq) & (np.nan_to_num(freq) >= threshold)
    d = haversine_min_distance_kdtree(
        np.column_stack([la.ravel(), lo.ravel()]),
        np.column_stack([la[zone], lo[zone]]),
    ).reshape(freq.shape)
    return np.where(np.isfinite(freq), d, np.nan)


class TestPersistentDistance:
    def _freq(self):
        rng = np.random.default_rng(0)
        f = rng.uniform(0, 0.25, (LAT.size, LON.size))
        f[10:14, 5:30] = 0.6  # a band of persistent fronts
        f[30:33, 30:33] = 0.5  # and a patch
        f[:, :3] = np.nan  # not assessed
        return f

    def test_equals_the_distance_to_every_zone_pixel(self):
        from h2mare.processing.core.front_layers import persistent_distance

        f = self._freq()
        got = persistent_distance(f, LAT, LON, 0.3)
        np.testing.assert_allclose(got, _brute_distance(f, 0.3), rtol=1e-5, atol=1e-3)

    def test_zero_inside_nan_where_not_assessed(self):
        from h2mare.processing.core.front_layers import persistent_distance

        f = self._freq()
        got = persistent_distance(f, LAT, LON, 0.3)
        assert (got[10:14, 5:30] == 0).all()
        assert np.isnan(got[:, :3]).all()
        assert np.isfinite(got[:, 3:]).all()

    def test_no_zone_is_no_distance(self):
        from h2mare.processing.core.front_layers import persistent_distance

        got = persistent_distance(np.full((LAT.size, LON.size), 0.1), LAT, LON, 0.3)
        assert np.isnan(got).all()


def _pdist_spec(min_frequency=0.5):
    return msgspec.convert(
        {
            **msgspec.to_builtins(SPEC),
            "persistent_distance": {"window": 4, "min_frequency": min_frequency},
        },
        FrontLayerSpec,
    )


@pytest.mark.usefixtures("interim_dir", "serial_pool")
class TestPersistentDistanceLayer:
    days = pd.date_range("2020-01-01", "2020-01-20", freq="D")

    def test_the_layer_is_the_distance_of_that_days_frequency(self):
        from h2mare.processing.core.front_layers import persistent_distance

        out = apply_front_layers(_sst(self.days), {"sst": _pdist_spec()}, "sst").load()
        assert "sst_pdist4" in out.data_vars
        for day in self.days[[1, 9, 19]]:
            np.testing.assert_array_equal(
                out["sst_pdist4"].sel(time=day).values,
                persistent_distance(
                    out["sst_ffreq4"].sel(time=day).values, LAT, LON, 0.5
                ),
            )

    def test_a_refresh_recomputes_it_with_the_frequency(self):
        spec = {"sst": _pdist_spec()}
        store = _Store()
        store.write(apply_front_layers(_sst(self.days), spec, "sst"))
        rewritten = self.days[5:10]
        store.write(
            apply_front_layers(
                _sst(rewritten, seed=3), spec, "sst", read_masks=store.read
            )
        )
        refreshed = recompute_following_frequency(spec, store.read, rewritten[-1])
        assert refreshed is not None and "sst_pdist4" in refreshed
        store.write(refreshed)

        truth_fields = _sst(self.days).copy()
        truth_fields["sst"].loc[dict(time=rewritten)] = _sst(rewritten, seed=3)[
            "sst"
        ].values
        truth = apply_front_layers(truth_fields, spec, "sst").load()
        after = self.days[10:13]
        np.testing.assert_array_equal(
            store.ds["sst_pdist4"].sel(time=after).values,
            truth["sst_pdist4"].sel(time=after).values,
        )


class TestPersistentDistanceConfig:
    def test_names(self):
        spec = _pdist_spec()
        assert spec.pdist_names("sst") == ["sst_pdist4"]
        assert "sst_pdist4" in spec.output_names("sst")
        assert "sst_pdist4" in spec.published_names("sst")
        assert SPEC.pdist_names("sst") == []

    @pytest.mark.parametrize(
        "pd_, match",
        [
            ({"window": 7, "min_frequency": 0.3}, "not one of frequency_days"),
            ({"window": 30, "min_frequency": 0}, "min_frequency"),
            ({"window": 30, "min_frequency": 1.5}, "min_frequency"),
            ({"window": 30}, "min_frequency"),
        ],
    )
    def test_malformed_entries_are_refused(self, pd_, match):
        with pytest.raises(msgspec.ValidationError, match=match):
            _load(front_layers={"sst": {**_LAYER, "persistent_distance": pd_}})

    def test_it_must_be_compiled_when_declared(self):
        layer = {**_LAYER, "persistent_distance": {"window": 30, "min_frequency": 0.3}}
        with pytest.raises(msgspec.ValidationError, match="add them to compiled_vars"):
            _load(
                compiled_vars=["sst", "sst_grad", "sst_ffreq30"],
                front_layers={"sst": layer},
            )
        _load(
            compiled_vars=["sst", "sst_grad", "sst_ffreq30", "sst_pdist30"],
            front_layers={"sst": layer},
        )
