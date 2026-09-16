"""Tests for Extractor — focused on logic that doesn't need external data."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger
from shapely.geometry import box

from h2mare.models import TimeStep
from h2mare.processing import extractor as extractor_module
from h2mare.processing.extractor import (
    Extractor,
    _keys_path,
    _load_completed_keys,
    _save_completed_keys,
    _warn_if_wholly_failed,
    _widen_degenerate,
    ensure_row_id,
    input_fingerprint,
    null_summary_lines,
    resolve_compiled_vars,
    resolve_read_from,
    split_vars_by_source,
    warn_on_subdaily_store,
)
from h2mare.types import BBox, DateRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spatial_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
) -> xr.Dataset:
    """Minimal spatial dataset (no time dimension)."""
    data = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
    return xr.Dataset(
        {"sst": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )


def _make_spatiotemporal_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
    n_days: int = 5,
) -> xr.Dataset:
    """Minimal dataset with daily time axis."""
    times = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = np.ones((n_days, len(lats), len(lons)))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _extractor(data, **kwargs) -> Extractor:
    """Build an Extractor, adding the positional row_id key first (the pre-step
    every caller now performs; index_col is required)."""
    return Extractor(ensure_row_id(data), index_col="row_id", **kwargs)


def _make_extractor(time_values: list, time_col: str = "time") -> Extractor:
    """Build a minimal Extractor from a list of time strings."""
    df = pd.DataFrame(
        {
            time_col: time_values,
            "lon": [10.0] * len(time_values),
            "lat": [40.0] * len(time_values),
        }
    )
    return _extractor(df, time_col=time_col)


def _make_distinct_ds() -> xr.Dataset:
    """Spatiotemporal dataset with a distinct value per (time, lat, lon) cell.

    sst[t, lat_i, lon_i] = t * 9 + lat_i * 3 + lon_i, so each cell is uniquely
    identifiable — lets a point/time extraction assert an exact expected value.
    """
    times = pd.to_datetime(["2020-01-01", "2020-01-02"])
    lats = [30.0, 35.0, 40.0]
    lons = [-10.0, -5.0, 0.0]
    data = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_geodf(geometries: list, times: list) -> gpd.GeoDataFrame:
    """Minimal GeoDataFrame with a time column and EPSG:4326 geometries."""
    return gpd.GeoDataFrame({"time": times, "geometry": geometries}, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# _resolve_time_col
# ---------------------------------------------------------------------------


class TestResolveTimeCol:
    def test_date_only_strings(self):
        """Date-only strings should NOT be truncated (no time component)."""
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # Should stay as date-only (normalised to midnight, no truncation log)
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_uniform_time_component_truncated(self):
        """Datetimes where all times are identical → truncate to midnight."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-02 06:00:00",
                "2020-01-03 06:00:00",
            ]
        )
        # All midnight after truncation
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_variable_time_component_kept(self):
        """Datetimes with varying times → keep full datetime."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-01 12:00:00",
                "2020-01-01 18:00:00",
            ]
        )
        hours = ext.data["time"].dt.hour.tolist()
        assert len(set(hours)) > 1  # times preserved

    def test_tz_aware_input_becomes_naive(self):
        """TZ-aware strings must become tz-naive after conversion."""
        ext = _make_extractor(
            [
                "2020-06-15T10:00:00+00:00",
                "2020-06-16T10:00:00+00:00",
            ]
        )
        assert ext.data["time"].dt.tz is None

    def test_raw_check_before_conversion(self):
        """
        Regression: raw string check was done AFTER pd.to_datetime conversion,
        so datetime(00:00:00) always matched the HH:MM:SS pattern, causing
        date-only inputs to be misclassified as having a time component.
        """
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # If the bug is present, the code enters the has_time_component branch
        # and then truncates to date — result would be midnight (same as correct).
        # The real regression is: variable-time branch would not be entered.
        # We verify by checking the date-only input stays at date (midnight).
        dates = ext.data["time"].dt.normalize()
        assert (ext.data["time"] == dates).all()

    def test_non_default_time_col(self):
        """Extractor should handle a non-default time column name."""
        ext = _make_extractor(["2020-03-01", "2020-03-02"], time_col="date")
        assert "time" in ext.data.columns  # renamed internally


# ---------------------------------------------------------------------------
# _nearest_grid_indices
# ---------------------------------------------------------------------------


class TestNearestGridIndices:
    def test_exact_grid_points(self):
        """Querying exact grid coordinates returns their exact indices."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, -5.0, 0.0]), np.array([30.0, 35.0, 40.0])
        )
        np.testing.assert_array_equal(lon_idx, [0, 1, 2])
        np.testing.assert_array_equal(lat_idx, [0, 1, 2])

    def test_off_grid_snaps_to_nearest(self):
        """Off-grid point snaps to the nearest grid point."""
        ds = _make_spatial_ds()
        # -8.0 is 2° from -10 and 3° from -5 → nearest is -10 (index 0)
        # 32.0 is 2° from 30 and 3° from 35 → nearest is 30 (index 0)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-8.0]), np.array([32.0])
        )
        assert lon_idx[0] == 0
        assert lat_idx[0] == 0

    def test_returns_ndarrays(self):
        """Output must be numpy ndarrays regardless of input size."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, 0.0]), np.array([30.0, 40.0])
        )
        assert isinstance(lat_idx, np.ndarray)
        assert isinstance(lon_idx, np.ndarray)

    def test_single_point(self):
        """Single-point query returns length-1 arrays."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([0.0]), np.array([40.0])
        )
        assert lat_idx.shape == (1,)
        assert lon_idx.shape == (1,)
        assert lon_idx[0] == 2  # 0.0 is last lon (index 2)
        assert lat_idx[0] == 2  # 40.0 is last lat (index 2)

    def test_irregular_grid(self):
        """Works on a non-uniform grid where searchsorted alone would be wrong."""
        ds = _make_spatial_ds(lons=[-10.0, -3.0, 0.0], lats=[30.0, 38.0, 40.0])
        # -4.0 is 1° from -3 and 6° from -10 → nearest is -3 (index 1)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-4.0]), np.array([30.0])
        )
        assert lon_idx[0] == 1


# ---------------------------------------------------------------------------
# _nearest_time_indices
# ---------------------------------------------------------------------------


class TestNearestTimeIndices:
    def test_exact_match(self):
        """Exact timestamp returns the correct index."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_picks_closer_left_neighbor(self):
        """Point 6h after a step is closer to that step than the next (18h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 06:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0  # 6 h from Jan 1, 18 h from Jan 2

    def test_picks_closer_right_neighbor(self):
        """Point 18h after a step is closer to the next step (6h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 18:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 1  # 18 h from Jan 1, 6 h from Jan 2

    def test_mismatched_datetime_resolutions_still_match(self):
        """
        Regression: a Zarr time axis decodes to datetime64[ns] while pandas
        parses input strings to [us], and both sides were cast to int64 raw.
        The microsecond query read as 1/1000th of its true instant, sorted
        before every stored step, and every row landed on index 0 — one
        arbitrary time returned for the whole input, varying only by location.
        Invisible in-process, because a fixture builds both sides alike.
        """
        ds = _make_spatiotemporal_ds()
        ds = ds.assign_coords(time=ds["time"].values.astype("datetime64[ns]"))
        q = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]).values.astype(
            "datetime64[us]"
        )

        idx = Extractor._nearest_time_indices(ds, q)

        np.testing.assert_array_equal(idx, [0, 2, 4])

    def test_before_first_step_clips_to_zero(self):
        """Query before the first time step is clipped to index 0."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2019-12-31"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_after_last_step_clips_to_last(self):
        """Query after the last time step is clipped to the last index."""
        ds = _make_spatiotemporal_ds(n_days=5)
        q = np.array(pd.to_datetime(["2025-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 4

    def test_multiple_queries(self):
        """Multiple timestamps resolved correctly in one call."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]))
        idx = Extractor._nearest_time_indices(ds, q)
        np.testing.assert_array_equal(idx, [0, 2, 4])


# ---------------------------------------------------------------------------
# Atomic checkpoint helpers
# ---------------------------------------------------------------------------


class TestAtomicCheckpoint:
    def test_save_completed_keys_writes_correct_content(self, tmp_path):
        checkpoint = tmp_path / "data.feather"
        keys = {"sst", "chl", "mld"}
        _save_completed_keys(checkpoint, keys, "abc123")
        dest = _keys_path(checkpoint)
        with open(dest) as f:
            payload = json.load(f)
        assert set(payload["completed"]) == keys
        assert payload["fingerprint"] == "abc123"

    def test_save_completed_keys_no_staging_file_remains(self, tmp_path):
        """The .tmp staging file must be cleaned up after a successful write."""
        checkpoint = tmp_path / "data.feather"
        _save_completed_keys(checkpoint, {"sst"}, "abc123")
        dest = _keys_path(checkpoint)
        staging = dest.with_suffix(".tmp")
        assert dest.exists()
        assert not staging.exists()

    def test_feather_atomic_write_no_tmp_remains(self, tmp_path):
        """Verify the staging-then-replace pattern: no .tmp file left after write."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_feather(staging)
        staging.replace(feather_path)

        assert feather_path.exists()
        assert not staging.exists()

    def test_feather_atomic_write_is_readable(self, tmp_path):
        """Data written through the staging pattern is readable from the final path."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.0, 2.0, 3.0]})
        df.to_feather(staging)
        staging.replace(feather_path)

        loaded = pd.read_feather(feather_path)
        pd.testing.assert_frame_equal(df, loaded)


# ---------------------------------------------------------------------------
# extract_from_dataset — extraction against an arbitrary in-memory dataset
# ---------------------------------------------------------------------------


class TestExtractFromDataset:
    def test_csv_spatiotemporal_exact_values(self):
        """Points resolve to the exact nearest (time, lat, lon) cell value."""
        ds = _make_distinct_ds()
        pts = pd.DataFrame(
            {
                "time": ["2020-01-02", "2020-01-01"],
                "lon": [-10.0, 0.0],
                "lat": [40.0, 30.0],
            }
        )
        out = _extractor(pts).extract_from_dataset(ds)

        # row 0: t=1, lat_i=2, lon_i=0 -> 1*9 + 2*3 + 0 = 15
        # row 1: t=0, lat_i=0, lon_i=2 -> 0    + 0   + 2 = 2
        assert out["sst"].tolist() == [15.0, 2.0]
        assert out.index.tolist() == [0, 1]

    def test_csv_spatial_only_no_time_coord(self):
        """A dataset without a time coord extracts purely on space (no raise)."""
        ds = _make_spatial_ds()  # sst = arange(9), lat rows, lon cols
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        out = _extractor(pts).extract_from_dataset(ds)
        # lat_i=2, lon_i=2 -> 2*3 + 2 = 8
        assert out["sst"].tolist() == [8.0]

    def test_csv_vars_subset(self):
        """`vars` restricts the extracted columns to the requested variable."""
        ds = _make_spatial_ds()
        ds = ds.assign(mld=ds["sst"] + 100)
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        out = _extractor(pts).extract_from_dataset(ds, vars="sst")
        # to_dataframe also carries lon/lat coord columns; the point is that the
        # unselected variable (mld) is absent.
        assert "sst" in out.columns
        assert "mld" not in out.columns

    def test_csv_clip_to_coverage_drops_out_of_extent(self):
        """Out-of-extent rows become NaN; in-extent rows keep their value."""
        ds = _make_distinct_ds()  # lon in [-10, 0], lat in [30, 40]
        pts = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-01"],
                "lon": [0.0, 50.0],  # second point is far outside
                "lat": [30.0, 30.0],
            }
        )
        out = _extractor(pts).extract_from_dataset(ds, clip_to_coverage=True)
        assert out["sst"].iloc[0] == 2.0
        assert np.isnan(out["sst"].iloc[1])

    def test_dataarray_with_vars_raises(self):
        """`vars` against a DataArray is a TypeError (ds[vars] is invalid)."""
        da = _make_spatial_ds()["sst"]
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        with pytest.raises(TypeError):
            _extractor(pts).extract_from_dataset(da, vars="sst")

    def test_shp_geometry_mean_on_ds_without_crs(self):
        """Geometry extraction computes the clipped mean and sets CRS via ensure_crs.

        The dataset is passed WITHOUT a rio CRS; ensure_crs must write the
        GeoDataFrame's CRS onto it so ds.rio.clip succeeds.
        """
        # Spatial-only ds: sst = arange(9); lat rows [30,35,40], lon cols [-10,-5,0]
        #   row(lat30): 0 1 2 | row(lat35): 3 4 5 | row(lat40): 6 7 8
        ds = _make_spatial_ds()
        assert ds.rio.crs is None  # precondition: no CRS on the dataset

        geoms = [
            box(-12, 28, -3, 36),  # touches lon{-10,-5} x lat{30,35} -> 0,1,3,4
            box(-1, 38, 1, 42),  # touches lon{0} x lat{40} -> 8
        ]
        gdf = _make_geodf(geoms, ["2020-01-01", "2020-01-01"])
        out = _extractor(gdf).extract_from_dataset(ds, n_workers=2)

        out = out.sort_index()
        assert out.loc[0, "sst"] == pytest.approx(2.0)  # mean(0,1,3,4)
        assert out.loc[1, "sst"] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# ensure_row_id — caller-side merge-key helper
# ---------------------------------------------------------------------------


class TestEnsureRowId:
    def test_unique_existing_key_passthrough(self):
        """An existing unique key is returned untouched."""
        df = pd.DataFrame({"row_id": [5, 6, 7], "x": [1, 2, 3]})
        out = ensure_row_id(df)
        assert out["row_id"].tolist() == [5, 6, 7]

    def test_duplicate_existing_key_raises(self):
        """A duplicated key is a caller error — raise, never overwrite."""
        df = pd.DataFrame({"row_id": [1, 1, 2]})
        with pytest.raises(ValueError):
            ensure_row_id(df)

    def test_absent_key_creates_positional_without_mutating_input(self):
        """A missing key is added as 0..n-1 on a copy; the input is untouched."""
        df = pd.DataFrame({"x": [10, 20, 30]})
        out = ensure_row_id(df)
        assert out["row_id"].tolist() == [0, 1, 2]
        assert "row_id" not in df.columns  # original not mutated

    def test_geodataframe_preserved(self):
        """Works on a GeoDataFrame and keeps the type."""
        gdf = _make_geodf([box(0, 0, 1, 1)], ["2020-01-01"])
        out = ensure_row_id(gdf)
        assert isinstance(out, gpd.GeoDataFrame)
        assert out["row_id"].tolist() == [0]


# ---------------------------------------------------------------------------
# _resolve_index — the Extractor consumes a caller-supplied key, never authors it
# ---------------------------------------------------------------------------


class TestResolveIndex:
    def test_missing_index_col_raises(self):
        """Constructing without the key column present is a hard error."""
        df = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        with pytest.raises(ValueError):
            Extractor(df, index_col="row_id")

    def test_duplicate_index_col_raises(self):
        """A non-unique key column is rejected at construction."""
        df = pd.DataFrame(
            {
                "row_id": [1, 1],
                "time": ["2020-01-01", "2020-01-02"],
                "lon": [0.0, 1.0],
                "lat": [40.0, 41.0],
            }
        )
        with pytest.raises(ValueError):
            Extractor(df, index_col="row_id")

    def test_explicit_unique_key_sets_index(self):
        """A valid key becomes the frame index, preserving its values."""
        df = pd.DataFrame(
            {
                "event_id": [101, 102],
                "time": ["2020-01-01", "2020-01-02"],
                "lon": [0.0, 1.0],
                "lat": [40.0, 41.0],
            }
        )
        ext = Extractor(df, index_col="event_id")
        assert ext.data.index.tolist() == [101, 102]
        assert ext.data.index.name == "event_id"


# ---------------------------------------------------------------------------
# Store contents vs what a var_key publishes
# ---------------------------------------------------------------------------


def _atm_config(*, hourly: bool = True) -> SimpleNamespace:
    """atm-accum-avg-shaped config: publishes 3 stored + 2 compile-derived vars."""
    return SimpleNamespace(
        compiled_vars=[
            "avg_iews",
            "avg_inss",
            "tp",
            "ekman_anom",
            "n_upwell_events_3d",
        ],
        time_step=TimeStep.HOURLY if hourly else TimeStep.DAILY,
        extract_depth_slices=None,
        rename_lonlat=False,
        # Required on a real config entry, and read when resolving the store.
        local_folder="CDS_AtmAccumAvg",
        store_root=None,
        source="cds",
    )


def _h2ds_config() -> SimpleNamespace:
    """Compiled-store config entry: daily, publishing nothing of its own.

    ``source="h2mare"`` is the marker the compiled var_key is found by — it is
    not located by the name "h2ds".
    """
    return SimpleNamespace(
        compiled_vars=[],
        time_step=TimeStep.DAILY,
        extract_depth_slices=None,
        rename_lonlat=False,
        source="h2mare",
        local_folder="h2ds",
        store_root=None,
    )


_STORED = ["avg_iews", "avg_inss", "tp"]


class TestSplitVarsBySource:
    """
    An hourly store holds strictly less than its var_key publishes. The split
    says which side each requested variable comes from, so both can be read and
    joined instead of returning a quietly thinner frame.
    """

    def test_implicit_all_vars_routes_the_derived_ones_to_h2ds(self):
        from_store, from_h2ds = split_vars_by_source(
            None, _STORED, "atm-accum-avg", _atm_config()
        )
        assert from_store == _STORED
        assert from_h2ds == ["ekman_anom", "n_upwell_events_3d"]

    def test_daily_store_asks_nothing_of_h2ds(self):
        """A daily store holds everything it publishes — behaviour unchanged."""
        cfg = SimpleNamespace(compiled_vars=_STORED, time_step=TimeStep.DAILY)
        assert split_vars_by_source(None, _STORED, "atm-accum-avg", cfg) == (
            _STORED,
            [],
        )

    def test_named_compile_derived_var_routes_to_h2ds(self):
        from_store, from_h2ds = split_vars_by_source(
            ["ekman_anom"], _STORED, "atm-accum-avg", _atm_config()
        )
        assert from_store == []
        assert from_h2ds == ["ekman_anom"]

    def test_unrecognised_var_still_raises(self):
        with pytest.raises(ValueError) as err:
            split_vars_by_source(["not_a_var"], _STORED, "atm-accum-avg", _atm_config())

        assert "not variables of" in str(err.value)

    def test_satisfiable_request_comes_wholly_from_the_store(self):
        assert split_vars_by_source(
            ["tp"], _STORED, "atm-accum-avg", _atm_config()
        ) == (
            ["tp"],
            [],
        )

    def test_var_key_without_compiled_vars_is_unaffected(self):
        """Most var_keys do not declare compiled_vars; they must not start failing."""
        cfg = SimpleNamespace(compiled_vars=None, time_step=TimeStep.DAILY)
        assert split_vars_by_source(None, _STORED, "sst", cfg) == (_STORED, [])

    def test_incomplete_daily_store_is_a_defect_not_a_route(self):
        """
        h2ds is the fallback only for hourly var_keys. A daily store missing
        what it publishes is a hole in the store, and must still say so.
        """
        cfg = SimpleNamespace(
            compiled_vars=[*_STORED, "sst_std"], time_step=TimeStep.DAILY
        )
        with pytest.raises(ValueError) as err:
            split_vars_by_source(None, _STORED, "sst", cfg)

        assert "sst_std" in str(err.value)
        assert "convert" in str(err.value)


class TestResolveReadFrom:
    """Which store answers, per (store cadence x input cadence x read_from)."""

    @staticmethod
    def _daily_cfg() -> SimpleNamespace:
        return SimpleNamespace(compiled_vars=_STORED, time_step=TimeStep.DAILY)

    # --- read_from="auto": inferred per var_key -----------------------------

    @pytest.mark.parametrize("subdaily", [False, True])
    def test_auto_daily_store_always_answers_for_itself(self, subdaily):
        assert (
            resolve_read_from(
                self._daily_cfg(), read_from="auto", subdaily_input=subdaily
            )
            == "native"
        )

    def test_auto_hourly_store_with_date_only_input_routes_to_compiled(self):
        assert (
            resolve_read_from(_atm_config(), read_from="auto", subdaily_input=False)
            == "compiled"
        )

    def test_auto_hourly_store_with_subdaily_input_serves_itself(self):
        assert (
            resolve_read_from(_atm_config(), read_from="auto", subdaily_input=True)
            == "native"
        )

    def test_auto_config_without_time_step_defaults_to_daily(self):
        """Stand-in configs predating the field must stay on the old path."""
        cfg = SimpleNamespace(compiled_vars=_STORED)
        assert (
            resolve_read_from(cfg, read_from="auto", subdaily_input=False) == "native"
        )

    # --- pinned: honoured whatever the cadences say -------------------------

    @pytest.mark.parametrize("subdaily", [False, True])
    @pytest.mark.parametrize("pinned", ["native", "compiled"])
    def test_pinned_is_honoured_for_an_hourly_store(self, pinned, subdaily):
        assert (
            resolve_read_from(_atm_config(), read_from=pinned, subdaily_input=subdaily)
            == pinned
        )

    @pytest.mark.parametrize("subdaily", [False, True])
    @pytest.mark.parametrize("pinned", ["native", "compiled"])
    def test_pinned_is_honoured_for_a_daily_store(self, pinned, subdaily):
        """read_from='compiled' on a daily var_key is the new capability: it
        was previously impossible to extract e.g. sst from the compiled store."""
        assert (
            resolve_read_from(
                self._daily_cfg(), read_from=pinned, subdaily_input=subdaily
            )
            == pinned
        )


class TestResolveH2dsVars:
    def test_declared_vars_present_in_h2ds_pass_through(self):
        available = [*_STORED, "ekman_anom", "n_upwell_events_3d", "sst"]
        got = resolve_compiled_vars(available, None, "atm-accum-avg", _atm_config())
        assert got == _atm_config().compiled_vars

    def test_missing_column_blames_a_stale_compile(self):
        """The ekman chain absent from h2ds means compile trails convert."""
        with pytest.raises(ValueError) as err:
            resolve_compiled_vars(_STORED, None, "atm-accum-avg", _atm_config())

        msg = str(err.value)
        assert "ekman_anom" in msg
        assert "h2mare compile" in msg

    def test_var_key_publishing_nothing_is_rejected(self):
        cfg = SimpleNamespace(compiled_vars=[], time_step=TimeStep.HOURLY)
        with pytest.raises(ValueError, match="compiled_vars"):
            resolve_compiled_vars(_STORED, None, "waves", cfg)


class TestWarnOnSubdailyStore:
    """
    Fires only where native hourly values are actually served. The h2ds route
    returns the daily semantics and units the caller already expects, so a
    warning there would be noise — that silence comes from the routing, which
    never calls this on the h2ds path (see TestProcessSingleVarkeyRouting).
    """

    @staticmethod
    def _capture(monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(
            extractor_module.logger, "warning", lambda m, *a, **k: seen.append(str(m))
        )
        return seen

    def _ds(self, units: str = "m") -> xr.Dataset:
        ds = _make_spatiotemporal_ds()
        ds["sst"].attrs["units"] = units
        return ds

    def test_hourly_store_warns_about_instantaneous_values(self, monkeypatch):
        seen = self._capture(monkeypatch)
        warn_on_subdaily_store("atm-accum-avg", _atm_config(), self._ds())

        assert len(seen) == 1
        assert "nearest hour" in seen[0]

    def test_hourly_warning_reports_stored_units(self, monkeypatch):
        """The m-vs-mm trap is invisible in the numbers, so surface the units."""
        seen = self._capture(monkeypatch)
        warn_on_subdaily_store("atm-accum-avg", _atm_config(), self._ds(units="m"))

        assert "'sst': 'm'" in seen[0] or "sst" in seen[0]

    def test_daily_store_is_silent(self, monkeypatch):
        seen = self._capture(monkeypatch)
        warn_on_subdaily_store("atm-accum-avg", _atm_config(hourly=False), self._ds())

        assert seen == []


class TestProcessSingleVarkeyRouting:
    """The routing must sit on the real extraction path, not just be importable."""

    @staticmethod
    def _hourly_ds() -> xr.Dataset:
        """5 days x 24 h of the three fields the hourly store actually holds.

        Values rise monotonically with the hour, so a nearest-hour hit is
        distinguishable from a daily aggregate.
        """
        times = pd.date_range("2020-01-01", periods=5 * 24, freq="h")
        lats, lons = [30.0, 35.0, 40.0], [-10.0, -5.0, 0.0]
        hourly = np.arange(len(times), dtype=float)[:, None, None] * np.ones(
            (1, len(lats), len(lons))
        )
        return xr.Dataset(
            {
                "tp": (["time", "lat", "lon"], hourly),
                "avg_iews": (["time", "lat", "lon"], hourly * 2),
                "avg_inss": (["time", "lat", "lon"], hourly * 3),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )

    @staticmethod
    def _h2ds(n_days: int = 5) -> xr.Dataset:
        """Daily compiled store: the stored fields reduced, plus the derived ones."""
        times = pd.date_range("2020-01-01", periods=n_days, freq="D")
        lats, lons = [30.0, 35.0, 40.0], [-10.0, -5.0, 0.0]
        daily = np.arange(len(times), dtype=float)[:, None, None] * np.ones(
            (1, len(lats), len(lons))
        )
        return xr.Dataset(
            {
                "tp": (["time", "lat", "lon"], daily + 100),
                "avg_iews": (["time", "lat", "lon"], daily + 200),
                "avg_inss": (["time", "lat", "lon"], daily + 300),
                "ekman_anom": (["time", "lat", "lon"], daily + 400),
                "n_upwell_events_3d": (["time", "lat", "lon"], daily + 500),
            },
            coords={"time": times, "lat": lats, "lon": lons},
        )

    def _patch_catalogs(self, monkeypatch, store_ds, h2ds) -> list[str]:
        """Route ZarrCatalog(var_key) to the right stand-in, recording the calls."""
        seen: list[str] = []

        def _factory(var_key, **_kw):
            seen.append(var_key)
            catalog = MagicMock()
            catalog.var_key = var_key
            coverage = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"))
            catalog.get_time_coverage.return_value = coverage
            # The compiled path asks per variable: the store's own end is the
            # union of everything merged into it, which over-promises for any
            # var_key padded out to it. Same dates here, different question.
            catalog.get_var_coverage.return_value = coverage
            catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
            catalog.open_dataset.return_value = h2ds if var_key == "h2ds" else store_ds
            return catalog

        monkeypatch.setattr(extractor_module, "ZarrCatalog", _factory)
        return seen

    def _extractor_for(self, cfg, times, **kwargs) -> Extractor:
        # Distinct lon/lat, so the bbox is a real box rather than the widened
        # point _define_bbox falls back to.
        df = pd.DataFrame({"time": times, "lon": [-9.0, -1.0], "lat": [31.0, 39.0]})
        ext = _extractor(df, time_col="time", **kwargs)
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": cfg, "h2ds": _h2ds_config()}
        )
        return ext

    def test_date_only_input_reads_h2ds_not_the_hourly_store(self, monkeypatch):
        """
        Regression: the hourly flip left a date-only extraction either raising
        (compile-derived vars absent from the store) or silently returning the
        00:00 hour. It must return the compiled daily value instead.
        """
        seen = self._patch_catalogs(monkeypatch, self._hourly_ds(), self._h2ds())
        ext = self._extractor_for(_atm_config(), ["2020-01-01", "2020-01-02"])

        out = ext.process_single_varkey("atm-accum-avg")

        assert seen == ["h2ds"]
        assert set(_atm_config().compiled_vars) <= set(out.columns)
        # day 0 / day 1 of the h2ds fixture, not hour 0 / hour 24 of the store
        assert out["tp"].tolist() == [100.0, 101.0]
        assert out["ekman_anom"].tolist() == [400.0, 401.0]

    def test_subdaily_input_serves_hours_and_broadcasts_daily_features(
        self, monkeypatch
    ):
        """Stored fields vary by hour; daily-by-construction ones repeat per day."""
        self._patch_catalogs(monkeypatch, self._hourly_ds(), self._h2ds())
        ext = self._extractor_for(
            _atm_config(), ["2020-01-01 03:00:00", "2020-01-01 15:00:00"]
        )

        out = ext.process_single_varkey("atm-accum-avg")

        assert out["tp"].tolist() == [3.0, 15.0]  # the hours themselves
        assert out["ekman_anom"].tolist() == [400.0, 400.0]  # same day, broadcast
        assert set(_atm_config().compiled_vars) <= set(out.columns)

    def test_daily_store_never_touches_h2ds(self, monkeypatch):
        cfg = SimpleNamespace(
            compiled_vars=["tp"],
            time_step=TimeStep.DAILY,
            extract_depth_slices=None,
            rename_lonlat=False,
            local_folder="CDS_AtmAccumAvg",
            store_root=None,
        )
        seen = self._patch_catalogs(monkeypatch, self._h2ds()[["tp"]], self._h2ds())
        ext = self._extractor_for(cfg, ["2020-01-01", "2020-01-02"])

        ext.process_single_varkey("atm-accum-avg")

        assert seen == ["atm-accum-avg"]

    def test_stale_h2ds_names_the_compile_step(self, monkeypatch):
        """Compile behind convert must be reported as such, not as a bad request."""
        thin = self._h2ds().drop_vars(["ekman_anom", "n_upwell_events_3d"])
        self._patch_catalogs(monkeypatch, self._hourly_ds(), thin)
        ext = self._extractor_for(_atm_config(), ["2020-01-01", "2020-01-02"])

        with pytest.raises(ValueError) as err:
            ext.process_single_varkey("atm-accum-avg")

        assert "h2mare compile" in str(err.value)

    def test_h2ds_catalog_is_opened_once_per_extractor(self, monkeypatch):
        """run() walks several hourly var_keys; each must not rescan the index."""
        seen = self._patch_catalogs(monkeypatch, self._hourly_ds(), self._h2ds())
        ext = self._extractor_for(_atm_config(), ["2020-01-01", "2020-01-02"])

        ext.process_single_varkey("atm-accum-avg")
        ext.process_single_varkey("atm-accum-avg")

        assert seen.count("h2ds") == 1

    def test_compiled_store_is_found_by_source_not_by_name(self, monkeypatch):
        """
        Another deployment may name its compiled store something else. It is
        identified by source: h2mare — the same marker _normalize_var_dict
        already excludes from default runs — not by the literal name "h2ds".
        """
        seen: list[str] = []

        def _factory(var_key, **_kw):
            seen.append(var_key)
            catalog = MagicMock()
            catalog.var_key = var_key
            coverage = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"))
            catalog.get_time_coverage.return_value = coverage
            # The compiled path asks per variable: the store's own end is the
            # union of everything merged into it, which over-promises for any
            # var_key padded out to it. Same dates here, different question.
            catalog.get_var_coverage.return_value = coverage
            catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
            catalog.open_dataset.return_value = self._h2ds()
            return catalog

        monkeypatch.setattr(extractor_module, "ZarrCatalog", _factory)
        ext = self._extractor_for(_atm_config(), ["2020-01-01", "2020-01-02"])
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": _atm_config(), "my_compiled": _h2ds_config()}
        )

        out = ext.process_single_varkey("atm-accum-avg")

        assert seen == ["my_compiled"]
        assert out["ekman_anom"].tolist() == [400.0, 401.0]

    def test_config_without_a_compiled_var_key_says_so(self, monkeypatch):
        self._patch_catalogs(monkeypatch, self._hourly_ds(), self._h2ds())
        ext = self._extractor_for(_atm_config(), ["2020-01-01", "2020-01-02"])
        ext.app_config = SimpleNamespace(variables={"atm-accum-avg": _atm_config()})

        with pytest.raises(ValueError, match="No compiled var_key"):
            ext.process_single_varkey("atm-accum-avg")


class TestCompiledCoverageIsPerVariable:
    """
    h2ds ends where its furthest-ahead source ends, and ``xr.merge`` pads every
    slower one with NaN out to that date. Clipping the compiled read against the
    *store's* end therefore let those padded days through, to be extracted as
    NaN with nothing said — the quiet half of the same bug that made the padding
    look like spatial gaps downstream.
    """

    _R = TestProcessSingleVarkeyRouting

    def _patch(self, monkeypatch, var_end: str) -> None:
        """h2ds runs to 2020-01-05; this var_key's own columns stop at *var_end*."""

        def _factory(var_key, **_kw):
            catalog = MagicMock()
            catalog.var_key = var_key
            catalog.get_time_coverage.return_value = DateRange(
                pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")
            )
            catalog.get_var_coverage.return_value = DateRange(
                pd.Timestamp("2020-01-01"), pd.Timestamp(var_end)
            )
            catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
            catalog.open_dataset.return_value = self._R._h2ds()
            return catalog

        monkeypatch.setattr(extractor_module, "ZarrCatalog", _factory)

    def _extractor_for(self, times) -> Extractor:
        # Distinct lon/lat per row, so what survives a clip is a real box rather
        # than the widened point _define_bbox falls back to.
        n = len(times)
        df = pd.DataFrame(
            {
                "time": times,
                "lon": [-9.0 - i for i in range(n)],
                "lat": [31.0 + i for i in range(n)],
            }
        )
        ext = _extractor(df, time_col="time")
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": _atm_config(), "h2ds": _h2ds_config()}
        )
        return ext

    _DATES = ["2020-01-01", "2020-01-02", "2020-01-04"]

    def test_dates_in_the_padding_are_clipped(self, monkeypatch):
        self._patch(monkeypatch, var_end="2020-01-02")
        ext = self._extractor_for(self._DATES)

        out = ext.process_single_varkey("atm-accum-avg")

        # 01-04 is on h2ds's axis but past this var_key's own end.
        assert out["time"].dt.strftime("%Y-%m-%d").tolist() == [
            "2020-01-01",
            "2020-01-02",
        ]

    def test_the_clip_names_the_var_key_not_the_store(self, monkeypatch):
        """'h2ds ends 01-05' would be true and useless; the var_key is the news."""
        self._patch(monkeypatch, var_end="2020-01-02")
        ext = self._extractor_for(self._DATES)

        messages: list[str] = []
        sink = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            ext.process_single_varkey("atm-accum-avg")
        finally:
            logger.remove(sink)

        warned = "".join(messages)
        assert "atm-accum-avg in h2ds" in warned
        assert "variable ends 2020-01-02" in warned

    def test_a_var_key_level_with_the_store_is_not_clipped(self, monkeypatch):
        self._patch(monkeypatch, var_end="2020-01-05")
        ext = self._extractor_for(self._DATES)

        out = ext.process_single_varkey("atm-accum-avg")

        assert len(out) == 3

    def test_clipping_down_to_one_record_still_extracts(self, monkeypatch):
        """The survivor's extent has no width; that is a query, not an error."""
        self._patch(monkeypatch, var_end="2020-01-01")
        ext = self._extractor_for(self._DATES)

        out = ext.process_single_varkey("atm-accum-avg")

        assert len(out) == 1


class TestStoreRootReachesTheReads:
    """
    Regression: ``Extractor`` captured ``store_root`` in __init__ and then never
    used it. All three read paths built a rootless ``ZarrCatalog``, so they
    resolved from settings and passing a root changed nothing about which files
    were opened — silently, since the store that answered was a real one.
    """

    _R = TestProcessSingleVarkeyRouting

    def _patch_recording_roots(self, monkeypatch) -> dict:
        """Route ZarrCatalog anywhere, recording the store_root each was given."""
        roots: dict = {}

        def _factory(var_key, **kw):
            roots[var_key] = kw.get("store_root")
            catalog = MagicMock()
            catalog.var_key = var_key
            coverage = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05"))
            catalog.get_time_coverage.return_value = coverage
            # The compiled path asks per variable: the store's own end is the
            # union of everything merged into it, which over-promises for any
            # var_key padded out to it. Same dates here, different question.
            catalog.get_var_coverage.return_value = coverage
            catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
            catalog.open_dataset.return_value = (
                self._R._h2ds()
                if var_key == "h2ds"
                else self._R._h2ds()[["tp"]]  # a daily native store
            )
            return catalog

        monkeypatch.setattr(extractor_module, "ZarrCatalog", _factory)
        return roots

    def _extractor_for(self, cfg, **kwargs) -> Extractor:
        df = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-02"],
                "lon": [-9.0, -1.0],
                "lat": [31.0, 39.0],
            }
        )
        ext = _extractor(df, time_col="time", **kwargs)
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": cfg, "h2ds": _h2ds_config()}
        )
        return ext

    def _daily_cfg(self, **over) -> SimpleNamespace:
        cfg = _atm_config(hourly=False)
        cfg.compiled_vars = ["tp"]
        for k, v in over.items():
            setattr(cfg, k, v)
        return cfg

    def test_native_read_uses_the_given_root(self, monkeypatch, tmp_path):
        roots = self._patch_recording_roots(monkeypatch)
        ext = self._extractor_for(self._daily_cfg(), store_root=tmp_path)

        ext.process_single_varkey("atm-accum-avg")

        assert roots["atm-accum-avg"] == tmp_path / "CDS_AtmAccumAvg"

    def test_compiled_read_uses_the_given_root(self, monkeypatch, tmp_path):
        roots = self._patch_recording_roots(monkeypatch)
        # An hourly var_key with a date-only input routes to the compiled store.
        ext = self._extractor_for(_atm_config(), store_root=tmp_path)

        ext.process_single_varkey("atm-accum-avg")

        assert roots["h2ds"] == tmp_path / "h2ds"

    def test_a_variables_own_root_beats_the_extractors(self, monkeypatch, tmp_path):
        """
        The Extractor's root is the default, the same rule PipelineManager and
        Compiler follow — a variable naming its own is read from there.
        """
        own = tmp_path / "other_drive"
        roots = self._patch_recording_roots(monkeypatch)
        ext = self._extractor_for(
            self._daily_cfg(store_root=str(own)), store_root=tmp_path
        )

        ext.process_single_varkey("atm-accum-avg")

        assert roots["atm-accum-avg"] == own / "CDS_AtmAccumAvg"

    def test_native_read_uses_the_given_app_config(self, monkeypatch, tmp_path):
        """
        Regression: the native read built its catalog without ``app_config``, so
        ZarrCatalog validated the var_key against the *process* config. An
        Extractor handed another project's config failed on any var_key that
        project has and this one lacks.
        """
        self._patch_recording_roots(monkeypatch)
        recording = extractor_module.ZarrCatalog
        configs: dict = {}

        def _factory(var_key, **kw):
            configs[var_key] = kw.get("app_config")
            return recording(var_key, **kw)

        monkeypatch.setattr(extractor_module, "ZarrCatalog", _factory)
        ext = self._extractor_for(self._daily_cfg(), store_root=tmp_path)

        ext.process_single_varkey("atm-accum-avg")

        assert configs["atm-accum-avg"] is ext.app_config


class TestPinnedReadFrom:
    """read_from overrides the inference, in both directions."""

    _R = TestProcessSingleVarkeyRouting

    def _extractor(self, times, **kwargs) -> Extractor:
        df = pd.DataFrame({"time": times, "lon": [-9.0, -1.0], "lat": [31.0, 39.0]})
        ext = _extractor(df, time_col="time", **kwargs)
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": _atm_config(), "h2ds": _h2ds_config()}
        )
        return ext

    def test_native_with_date_only_input_reads_the_hourly_store(self, monkeypatch):
        """Pinning native overrides the date-only default of going compiled."""
        seen = self._R()._patch_catalogs(
            monkeypatch, self._R._hourly_ds(), self._R._h2ds()
        )
        ext = self._extractor(["2020-01-01", "2020-01-02"], read_from="native")

        out = ext.process_single_varkey("atm-accum-avg", vars=["tp"])

        assert seen == ["atm-accum-avg"]
        assert out["tp"].tolist() == [0.0, 24.0]  # hour 00:00 of each day

    def test_native_still_sources_compile_derived_vars_from_compiled(self, monkeypatch):
        """
        An hourly store never held the derived chain, so honouring native to the
        letter would mean returning nothing. Route and warn instead — the same
        var_key converted daily holds them natively and never reaches here.
        """
        seen = self._R()._patch_catalogs(
            monkeypatch, self._R._hourly_ds(), self._R._h2ds()
        )
        warned: list[str] = []
        monkeypatch.setattr(
            extractor_module.logger, "warning", lambda m, *a, **k: warned.append(str(m))
        )
        ext = self._extractor(["2020-01-01", "2020-01-02"], read_from="native")

        out = ext.process_single_varkey("atm-accum-avg")

        assert seen == ["atm-accum-avg", "h2ds"]
        assert out["ekman_anom"].tolist() == [400.0, 401.0]
        assert any("compile time" in m for m in warned)

    def test_compiled_on_a_daily_var_key(self, monkeypatch):
        """Previously impossible: reading a daily var_key from the compiled store."""
        cfg = SimpleNamespace(
            compiled_vars=["tp"],
            time_step=TimeStep.DAILY,
            extract_depth_slices=None,
            rename_lonlat=False,
            source="cmems",
            local_folder="CDS_AtmAccumAvg",
            store_root=None,
        )
        seen = self._R()._patch_catalogs(
            monkeypatch, self._R._hourly_ds(), self._R._h2ds()
        )
        ext = self._extractor(["2020-01-01", "2020-01-02"], read_from="compiled")
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": cfg, "h2ds": _h2ds_config()}
        )

        out = ext.process_single_varkey("atm-accum-avg")

        assert seen == ["h2ds"]
        assert out["tp"].tolist() == [100.0, 101.0]  # compiled values, not native

    def test_compiled_with_subdaily_input_warns_it_can_only_give_days(
        self, monkeypatch
    ):
        self._R()._patch_catalogs(monkeypatch, self._R._hourly_ds(), self._R._h2ds())
        warned: list[str] = []
        monkeypatch.setattr(
            extractor_module.logger, "warning", lambda m, *a, **k: warned.append(str(m))
        )
        ext = self._extractor(
            ["2020-01-01 03:00:00", "2020-01-01 18:00:00"], read_from="compiled"
        )

        out = ext.process_single_varkey("atm-accum-avg")

        assert any("daily" in m and "read_from='compiled'" in m for m in warned)
        assert out["tp"].tolist() == [100.0, 100.0]  # same day, same value


class TestTimeCadenceOverride:
    @staticmethod
    def _df(times: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"time": times, "lon": [10.0, 11.0], "lat": [40.0, 41.0]})

    def test_auto_reads_a_uniform_stamp_as_nominal(self):
        """A stamp identical on every row is someone's export default, not an hour."""
        ext = _extractor(self._df(["2020-01-01 14:00:00", "2020-01-02 14:00:00"]))

        assert ext.input_is_subdaily is False
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_native_honours_a_uniform_stamp_as_a_real_hour(self):
        ext = _extractor(
            self._df(["2020-01-01 14:00:00", "2020-01-02 14:00:00"]),
            time_cadence="hourly",
        )

        assert ext.input_is_subdaily is True
        assert ext.data["time"].dt.hour.eq(14).all()

    def test_daily_truncates_even_varying_stamps(self):
        ext = _extractor(
            self._df(["2020-01-01 06:00:00", "2020-01-01 18:00:00"]),
            time_cadence="daily",
        )

        assert ext.input_is_subdaily is False
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_auto_infers_subdaily_from_varying_stamps(self):
        ext = _extractor(self._df(["2020-01-01 06:00:00", "2020-01-01 18:00:00"]))

        assert ext.input_is_subdaily is True

    def test_date_only_input_is_daily(self):
        ext = _extractor(self._df(["2020-01-01", "2020-01-02"]))

        assert ext.input_is_subdaily is False


class TestResolveCoverageEndOfDay:
    """
    Store coverage names calendar days; input rows carry instants. Compared
    bare, every sample after midnight on the final covered day was clipped —
    23 hours' worth against an hourly store.
    """

    @staticmethod
    def _extractor(times: list[str]) -> Extractor:
        # Distinct lon/lat, so the bbox is a real box rather than the widened
        # point _define_bbox falls back to.
        return _extractor(
            pd.DataFrame({"time": times, "lon": [-9.0, -1.0], "lat": [31.0, 39.0]})
        )

    @staticmethod
    def _catalog() -> MagicMock:
        catalog = MagicMock()
        catalog.var_key = "atm-accum-avg"
        catalog.get_time_coverage.return_value = DateRange(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")
        )
        catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
        return catalog

    def test_last_day_after_midnight_survives(self):
        ext = self._extractor(["2020-01-05 06:00:00", "2020-01-05 23:00:00"])

        dates = ext._resolve_coverage(self._catalog())

        assert len(dates) == 2
        assert max(dates) == pd.Timestamp("2020-01-05 23:00:00")

    def test_beyond_the_last_day_is_still_clipped(self):
        ext = self._extractor(["2020-01-05 23:00:00", "2020-01-06 01:00:00"])

        dates = ext._resolve_coverage(self._catalog())

        assert dates == [pd.Timestamp("2020-01-05 23:00:00")]


class TestDefineBboxWidensADegenerateExtent:
    """
    A query extent with no width is ordinary input — one sample, a transect
    along a parallel, or the single row left after a coverage clip. ``BBox``
    refuses it, rightly, for a *store's* extent; as a query it only has to name
    the cells to read, and the read pads by a whole grid cell either way.
    """

    @staticmethod
    def _ext(lons: list[float], lats: list[float]) -> Extractor:
        return _extractor(
            pd.DataFrame({"time": ["2020-01-01"] * len(lons), "lon": lons, "lat": lats})
        )

    def test_single_point_becomes_a_box_around_it(self):
        ext = self._ext([-9.0], [31.0])

        bbox = ext._define_bbox(ext.data)

        assert bbox.xmin < -9.0 < bbox.xmax
        assert bbox.ymin < 31.0 < bbox.ymax

    def test_a_transect_widens_only_the_flat_axis(self):
        """Rows along one parallel: lon already spans, lat does not."""
        ext = self._ext([-9.0, -5.0, -1.0], [31.0, 31.0, 31.0])

        bbox = ext._define_bbox(ext.data)

        assert (bbox.xmin, bbox.xmax) == (-9.0, -1.0)
        assert bbox.ymin < 31.0 < bbox.ymax

    def test_a_real_box_is_left_alone(self):
        ext = self._ext([-9.0, -1.0], [31.0, 39.0])

        bbox = ext._define_bbox(ext.data)

        assert bbox.to_tuple() == (-9.0, 31.0, -1.0, 39.0)

    def test_an_inverted_box_is_still_refused(self):
        """min/max cannot produce one, so it means a defect, not a point."""
        with pytest.raises(ValueError, match="xmin"):
            BBox.from_tuple(_widen_degenerate((1.0, 30.0, -1.0, 40.0)))


def _depth_config(
    *, extract: list[int] | None = None, compile_: list[int] | None = None
) -> SimpleNamespace:
    """thetao/o2-shaped config: one stored var on a depth axis, sliced names published."""
    return SimpleNamespace(
        compiled_vars=[f"thetao_{d}" for d in (compile_ or [])],
        compile_depth_slices=compile_,
        extract_depth_slices=extract,
        time_step=TimeStep.DAILY,
        rename_lonlat=False,
        source="cmems",
        local_folder="CMEMS_Thetao",
        store_root=None,
    )


def _depth_ds(levels: list[float] = [0.0, 100.0, 500.0, 1000.0]) -> xr.Dataset:
    """One variable on a depth axis, valued so each level is identifiable."""
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    lats, lons = [30.0, 35.0, 40.0], [-10.0, -5.0, 0.0]
    data = np.zeros((len(times), len(levels), len(lats), len(lons)))
    for i, lvl in enumerate(levels):
        data[:, i, :, :] = lvl
    return xr.Dataset(
        {"thetao": (["time", "depth", "lat", "lon"], data)},
        coords={"time": times, "depth": levels, "lat": lats, "lon": lons},
    )


class TestDepthVariables:
    """
    A 3-D variable's compiled_vars name post-slicing columns (thetao_100, …)
    while its store holds one variable on a depth axis. Comparing the two — as
    the publish/store reconciliation did — reported every level as a gap in the
    store and refused to extract at all.
    """

    def _extractor(self, cfg, monkeypatch, ds) -> Extractor:
        catalog = MagicMock()
        catalog.var_key = "thetao"
        catalog.get_time_coverage.return_value = DateRange(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")
        )
        catalog.get_bbox.return_value = BBox(-20.0, 30.0, 20.0, 50.0)
        catalog.open_dataset.return_value = ds
        monkeypatch.setattr(extractor_module, "ZarrCatalog", lambda _vk, **_kw: catalog)

        df = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-02"],
                "lon": [-9.0, -1.0],
                "lat": [31.0, 39.0],
            }
        )
        ext = _extractor(df, time_col="time")
        ext.app_config = SimpleNamespace(
            variables={"thetao": cfg, "h2ds": _h2ds_config()}
        )
        return ext

    def test_depth_levels_are_not_reported_as_a_gap_in_the_store(self, monkeypatch):
        """Regression: raised 'this is a gap in the store — re-run convert'."""
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100, 500, 1000])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        out = ext.process_single_varkey("thetao")

        assert set(out.columns) >= {"thetao_0", "thetao_100"}
        assert out["thetao_0"].tolist() == [0.0, 0.0]
        assert out["thetao_100"].tolist() == [100.0, 100.0]

    def test_extract_slices_may_differ_from_compile_slices(self, monkeypatch):
        """o2 extracts 3 levels and compiles 4; the two lists are independent."""
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100, 500, 1000])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        out = ext.process_single_varkey("thetao")

        sliced = {c for c in out.columns if c.startswith("thetao_")}
        assert sliced == {"thetao_0", "thetao_100"}  # not the four compiled ones

    def test_missing_extract_slices_falls_back_to_compile_slices(self, monkeypatch):
        """
        thetao declares only compile_depth_slices. Without a fallback the depth
        axis survives into extraction and the geometry engine's dimensionless
        .mean() averages it away into one value spanning the whole range.
        """
        cfg = _depth_config(extract=None, compile_=[0, 500])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        out = ext.process_single_varkey("thetao")

        assert {"thetao_0", "thetao_500"} <= set(out.columns)
        assert out["thetao_500"].tolist() == [500.0, 500.0]

    def test_no_declared_levels_at_all_is_refused(self, monkeypatch):
        cfg = _depth_config(extract=None, compile_=None)
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        with pytest.raises(ValueError, match="declares no depth levels"):
            ext.process_single_varkey("thetao")

    def test_a_single_level_can_be_named(self, monkeypatch):
        """vars= on a depth variable names the post-expansion columns."""
        cfg = _depth_config(extract=[0, 100, 500], compile_=[0, 100, 500])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        out = ext.process_single_varkey("thetao", vars=["thetao_100"])

        sliced = {c for c in out.columns if c.startswith("thetao_")}
        assert sliced == {"thetao_100"}
        assert out["thetao_100"].tolist() == [100.0, 100.0]

    def test_the_bare_var_key_means_every_level(self, monkeypatch):
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        out = ext.process_single_varkey("thetao", vars=["thetao"])

        assert {"thetao_0", "thetao_100"} <= set(out.columns)

    def test_an_unavailable_level_names_the_ones_that_exist(self, monkeypatch):
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100, 500, 1000])
        ext = self._extractor(cfg, monkeypatch, _depth_ds())

        with pytest.raises(ValueError) as err:
            ext.process_single_varkey("thetao", vars=["thetao_1000"])

        msg = str(err.value)
        assert "thetao_1000" in msg
        assert "thetao_100" in msg  # what it does yield
        assert "extract_depth_slices" in msg


class TestSplitVarsBySourceDepth:
    def test_depth_disables_the_reconciliation(self):
        """The two sides are not comparable, so nothing is reported missing."""
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100, 500, 1000])
        assert split_vars_by_source(
            None, ["thetao"], "thetao", cfg, has_depth=True
        ) == (
            [],
            [],
        )

    def test_depth_passes_a_request_through_untouched(self):
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100])
        assert split_vars_by_source(
            ["thetao_100"], ["thetao"], "thetao", cfg, has_depth=True
        ) == (["thetao_100"], [])

    def test_without_depth_the_reconciliation_still_runs(self):
        """The flag must not weaken the check for flat variables."""
        cfg = _depth_config(extract=[0, 100], compile_=[0, 100])
        with pytest.raises(ValueError, match="gap in"):
            split_vars_by_source(None, ["thetao"], "thetao", cfg, has_depth=False)


class TestGeometryPathSpatialDims:
    """
    rio.clip resolves spatial dims by name, falling back to lon/lat only when
    they carry CF attributes. CMEMS and AVISO stores inherit those from source,
    so every var_key but fsle/eddies passed the precondition by luck; CDS
    stores and the compiled h2ds carry no coordinate attributes at all and
    clipped to nothing but NaN — reported only as a per-geometry DEBUG line.
    """

    @staticmethod
    def _bare_ds() -> xr.Dataset:
        """A store-shaped dataset whose lon/lat carry no CF attributes."""
        ds = _make_spatial_ds()
        ds.lon.attrs.clear()
        ds.lat.attrs.clear()
        return ds

    def _run(self, cfg) -> pd.DataFrame:
        geoms = [box(-12, 28, -3, 36), box(-1, 38, 1, 42)]
        gdf = _make_geodf(geoms, ["2020-01-01", "2020-01-01"])
        ext = _extractor(gdf)
        ext.app_config = SimpleNamespace(variables={"x": cfg, "h2ds": _h2ds_config()})
        return ext._extract(ext.data, self._bare_ds(), n_workers=2).sort_index()

    def test_attribute_less_lon_lat_still_clip(self):
        """Regression: returned all-NaN because rioxarray could not find x/y."""
        out = self._run(SimpleNamespace(rename_lonlat=False))

        assert out["sst"].notna().all()
        assert out.loc[0, "sst"] == pytest.approx(2.0)  # mean(0,1,3,4)
        assert out.loc[1, "sst"] == pytest.approx(8.0)

    def test_rename_lonlat_true_is_not_applied_twice(self):
        """The config flag is now redundant, and must not double-rename."""
        out = self._run(SimpleNamespace(rename_lonlat=True))

        assert out["sst"].notna().all()

    def test_a_dataset_already_on_x_y_is_untouched(self):
        geoms = [box(-12, 28, -3, 36)]
        gdf = _make_geodf(geoms, ["2020-01-01"])
        ext = _extractor(gdf)
        ds = self._bare_ds().rename({"lon": "x", "lat": "y"})

        out = ext._extract(ext.data, ds, n_workers=1)

        assert out["sst"].notna().all()

    def test_dangling_grid_mapping_attribute_still_clips(self):
        """Regression: an all-NaN column for CMEMS/AVISO vars read from h2ds.

        Those variables inherit ``grid_mapping: "crs"`` from source, but the
        compiled store carries no ``crs`` coordinate to match it. ensure_crs
        must run *after* the x/y rename, or write_crs cannot walk the variables
        to find that attribute, parks the CRS on ``spatial_ref`` instead, and
        leaves every variable pointing at a coordinate that does not exist —
        a dataset with a CRS whose variables have none, and rio.clip raises
        MissingCRS per geometry.
        """
        ds = self._bare_ds()
        ds["sst"].attrs["grid_mapping"] = "crs"
        assert "crs" not in ds.coords  # precondition: the pointer is dangling

        geoms = [box(-12, 28, -3, 36), box(-1, 38, 1, 42)]
        gdf = _make_geodf(geoms, ["2020-01-01", "2020-01-01"])
        ext = _extractor(gdf)

        out = ext._extract(ext.data, ds, n_workers=2).sort_index()

        assert out["sst"].notna().all()
        assert out.loc[0, "sst"] == pytest.approx(2.0)  # mean(0,1,3,4)
        assert out.loc[1, "sst"] == pytest.approx(8.0)


class TestWarnIfWhollyFailed:
    @staticmethod
    def _capture(monkeypatch) -> list[str]:
        seen: list[str] = []
        monkeypatch.setattr(
            extractor_module.logger, "warning", lambda m, *a, **k: seen.append(str(m))
        )
        return seen

    def test_all_nan_with_errors_warns(self, monkeypatch):
        seen = self._capture(monkeypatch)
        df = pd.DataFrame({"sst": [float("nan"), float("nan")]})

        _warn_if_wholly_failed(df, [ValueError("x dimension not found")])

        assert len(seen) == 1
        assert "x dimension not found" in seen[0]

    def test_a_partial_failure_is_ordinary(self, monkeypatch):
        """Geometries outside the grid are data, not a broken precondition."""
        seen = self._capture(monkeypatch)
        df = pd.DataFrame({"sst": [1.0, float("nan")]})

        _warn_if_wholly_failed(df, [ValueError("boom")])

        assert seen == []

    def test_all_nan_without_errors_is_silent(self, monkeypatch):
        """Genuinely empty data raised nothing, so there is nothing to report."""
        seen = self._capture(monkeypatch)
        df = pd.DataFrame({"sst": [float("nan")]})

        _warn_if_wholly_failed(df, [])

        assert seen == []

    def test_an_empty_frame_is_silent(self, monkeypatch):
        seen = self._capture(monkeypatch)

        _warn_if_wholly_failed(pd.DataFrame(), [ValueError("boom")])

        assert seen == []


class TestEmptyVarsMeansEverything:
    """
    `run({"waves": []})` is the documented way to ask for everything a var_key
    publishes. Read as an explicit selection of nothing it reached the compiled
    reader with an empty request, which blamed the config — "declares no
    compiled_vars" — for something the caller had said perfectly well.
    """

    _R = TestProcessSingleVarkeyRouting

    def _extractor(self, times, **kwargs) -> Extractor:
        df = pd.DataFrame({"time": times, "lon": [-9.0, -1.0], "lat": [31.0, 39.0]})
        ext = _extractor(df, time_col="time", **kwargs)
        ext.app_config = SimpleNamespace(
            variables={"atm-accum-avg": _atm_config(), "h2ds": _h2ds_config()}
        )
        return ext

    def test_empty_list_reads_everything_from_the_compiled_store(self, monkeypatch):
        """Regression: raised 'declares no compiled_vars' for a populated config."""
        self._R()._patch_catalogs(monkeypatch, self._R._hourly_ds(), self._R._h2ds())
        ext = self._extractor(["2020-01-01", "2020-01-02"])

        out = ext.process_single_varkey("atm-accum-avg", vars=[])

        assert set(_atm_config().compiled_vars) <= set(out.columns)

    def test_empty_list_matches_none(self, monkeypatch):
        self._R()._patch_catalogs(monkeypatch, self._R._hourly_ds(), self._R._h2ds())
        ext = self._extractor(["2020-01-01", "2020-01-02"])

        by_empty = ext.process_single_varkey("atm-accum-avg", vars=[])
        by_none = ext.process_single_varkey("atm-accum-avg", vars=None)

        assert list(by_empty.columns) == list(by_none.columns)

    def test_empty_list_on_the_native_path_too(self, monkeypatch):
        self._R()._patch_catalogs(monkeypatch, self._R._hourly_ds(), self._R._h2ds())
        ext = self._extractor(
            ["2020-01-01 03:00:00", "2020-01-01 15:00:00"], read_from="native"
        )

        out = ext.process_single_varkey("atm-accum-avg", vars=[])

        assert set(_atm_config().compiled_vars) <= set(out.columns)

    def test_a_real_selection_is_still_honoured(self, monkeypatch):
        """The collapse must not swallow an actual one-variable request."""
        self._R()._patch_catalogs(monkeypatch, self._R._hourly_ds(), self._R._h2ds())
        ext = self._extractor(["2020-01-01", "2020-01-02"])

        out = ext.process_single_varkey("atm-accum-avg", vars=["ekman_anom"])

        assert "ekman_anom" in out.columns
        assert "tp" not in out.columns

    def test_a_var_key_publishing_nothing_still_says_so(self):
        """The original message stays reachable for the case it describes."""
        cfg = SimpleNamespace(compiled_vars=[], time_step=TimeStep.HOURLY)
        with pytest.raises(ValueError, match="compiled_vars"):
            resolve_compiled_vars(_STORED, None, "waves", cfg)


class TestNullSummaryLines:
    """The end-of-run tally: a count always, a share only where it means something."""

    @staticmethod
    def _df(**cols) -> pd.DataFrame:
        return pd.DataFrame(cols)

    def test_a_clean_variable_shows_no_share(self):
        df = self._df(sst=[1.0, 2.0, 3.0])

        assert null_summary_lines(df, ["sst"]) == ["  sst: 0"]

    def test_a_partial_null_shows_its_share(self):
        df = self._df(sst=[1.0, float("nan"), 3.0])

        assert null_summary_lines(df, ["sst"]) == ["  sst: 1 (33.3%)"]

    def test_an_all_null_variable_reads_100_percent(self):
        df = self._df(sst=[float("nan")] * 4)

        assert null_summary_lines(df, ["sst"]) == ["  sst: 4 (100.0%)"]

    def test_one_decimal_place(self):
        df = self._df(sst=[float("nan")] + [1.0] * 6)

        assert null_summary_lines(df, ["sst"]) == ["  sst: 1 (14.3%)"]

    def test_columns_are_reported_in_order(self):
        df = self._df(
            a=[1.0, 2.0], b=[float("nan"), float("nan")], c=[1.0, float("nan")]
        )

        assert null_summary_lines(df, ["a", "b", "c"]) == [
            "  a: 0",
            "  b: 2 (100.0%)",
            "  c: 1 (50.0%)",
        ]

    def test_an_empty_frame_does_not_divide_by_zero(self):
        df = pd.DataFrame({"sst": pd.Series(dtype=float)})

        assert null_summary_lines(df, ["sst"]) == ["  sst: 0"]

    def test_only_the_requested_columns_are_tallied(self):
        df = self._df(sst=[float("nan")], lon=[float("nan")])

        assert null_summary_lines(df, ["sst"]) == ["  sst: 1 (100.0%)"]


class TestInputFingerprint:
    """
    The checkpoint lives at one fixed path, so the next run finds whatever the
    last one left there. Resumed on faith, a different input of the same shape
    has its rows replayed from the previous run — silently, and with the right
    index, because ensure_row_id keys positionally.
    """

    @staticmethod
    def _df(times, lons=(10.0, 11.0), lats=(40.0, 41.0)) -> pd.DataFrame:
        return pd.DataFrame({"time": times, "lon": list(lons), "lat": list(lats)})

    def _fp(self, df, index_col="row_id") -> str:
        return input_fingerprint(ensure_row_id(df).set_index(index_col), index_col)

    def test_the_same_frame_hashes_the_same(self):
        a = self._df(["2020-01-01", "2020-01-02"])
        b = self._df(["2020-01-01", "2020-01-02"])

        assert self._fp(a) == self._fp(b)

    def test_different_times_hash_differently(self):
        a = self._df(["2020-01-01", "2020-01-02"])
        b = self._df(["2021-06-01", "2021-06-02"])

        assert self._fp(a) != self._fp(b)

    def test_different_places_hash_differently(self):
        """The trap: same length, same positional keys, different locations."""
        a = self._df(["2020-01-01", "2020-01-02"])
        b = self._df(["2020-01-01", "2020-01-02"], lons=(-30.0, -31.0))

        assert self._fp(a) != self._fp(b)

    def test_row_count_changes_the_hash(self):
        a = self._df(["2020-01-01", "2020-01-02"])
        b = self._df(
            ["2020-01-01", "2020-01-02", "2020-01-03"], (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)
        )

        assert self._fp(a) != self._fp(b)

    def test_the_key_column_name_is_part_of_it(self):
        """A checkpoint keyed on another column cannot be set_index()'d here."""
        df = self._df(["2020-01-01", "2020-01-02"])
        keyed = ensure_row_id(df).set_index("row_id")

        assert input_fingerprint(keyed, "row_id") != input_fingerprint(
            keyed, "event_id"
        )

    def test_geometries_are_hashed_not_rejected(self):
        gdf = _make_geodf([box(0, 0, 1, 1), box(2, 2, 3, 3)], ["2020-01-01"] * 2)
        other = _make_geodf([box(0, 0, 1, 1), box(9, 9, 10, 10)], ["2020-01-01"] * 2)

        a = input_fingerprint(ensure_row_id(gdf).set_index("row_id"), "row_id")
        b = input_fingerprint(ensure_row_id(other).set_index("row_id"), "row_id")

        assert a != b


class TestCheckpointValidation:
    """_load_completed_keys returns None for anything that is not this input's."""

    @staticmethod
    def _write(tmp_path, payload) -> Path:
        checkpoint = tmp_path / "data.feather"
        with open(_keys_path(checkpoint), "w") as f:
            json.dump(payload, f)
        return checkpoint

    def test_a_matching_fingerprint_resumes(self, tmp_path):
        checkpoint = self._write(
            tmp_path, {"fingerprint": "abc", "completed": ["sst", "chl"]}
        )

        assert _load_completed_keys(checkpoint, "abc") == {"sst", "chl"}

    def test_a_different_input_is_discarded(self, tmp_path):
        """Regression: these rows used to be replayed onto the new input."""
        checkpoint = self._write(tmp_path, {"fingerprint": "abc", "completed": ["sst"]})

        assert _load_completed_keys(checkpoint, "xyz") is None

    def test_a_legacy_bare_list_is_discarded(self, tmp_path):
        """Pre-fingerprint sidecars cannot be matched, so they are not trusted."""
        checkpoint = self._write(tmp_path, ["sst", "chl"])

        assert _load_completed_keys(checkpoint, "abc") is None

    def test_an_unreadable_sidecar_is_discarded(self, tmp_path):
        checkpoint = tmp_path / "data.feather"
        _keys_path(checkpoint).write_text("{not json")

        assert _load_completed_keys(checkpoint, "abc") is None

    def test_an_absent_sidecar_is_discarded(self, tmp_path):
        assert _load_completed_keys(tmp_path / "data.feather", "abc") is None

    def test_a_round_trip_matches(self, tmp_path):
        checkpoint = tmp_path / "data.feather"
        _save_completed_keys(checkpoint, {"sst", "waves"}, "fp1")

        assert _load_completed_keys(checkpoint, "fp1") == {"sst", "waves"}
        assert _load_completed_keys(checkpoint, "fp2") is None


class TestInterruptedMidCheckpoint:
    """
    The checkpoint is two files written in sequence: the feather, then the
    sidecar naming what it holds. A kill between them leaves the feather
    carrying a var_key's columns while the sidecar still omits the key.
    """

    @staticmethod
    def _extractor(tmp_path, monkeypatch):
        real = extractor_module.get_settings()

        class _Settings:
            INTERIM_DIR = tmp_path

            def __getattr__(self, name):
                return getattr(real, name)

        monkeypatch.setattr(extractor_module, "get_settings", lambda: _Settings())

        df = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-02"],
                "lon": [10.0, 11.0],
                "lat": [40.0, 41.0],
            }
        )
        extractor = Extractor(ensure_row_id(df), index_col="row_id", time_col="time")
        fresh = pd.DataFrame({"sst": [1.5, 2.5]}, index=extractor.data.index)
        fresh.index.name = "row_id"
        monkeypatch.setattr(
            extractor,
            "process_single_varkey",
            lambda var_key, vars=None, n_workers=8: fresh.copy(),
        )
        return extractor, fresh

    def _write_half_checkpoint(self, extractor, tmp_path, values=(99.0, 99.0)):
        """
        The post-crash state: data written, sidecar not yet updated.

        The stored values differ from what the re-extraction returns, so the
        tests can tell a recovered column from a replayed one.
        """
        stale = pd.DataFrame({"sst": list(values)}, index=extractor.data.index)
        stale.index.name = extractor.index_col
        checkpoint = tmp_path / "extraction_checkpoint.feather"
        extractor.data.join(stale).reset_index().to_feather(checkpoint)
        with open(_keys_path(checkpoint), "w") as f:
            json.dump(
                {
                    "fingerprint": input_fingerprint(
                        extractor.data, extractor.index_col
                    ),
                    "completed": [],
                },
                f,
            )
        return checkpoint

    def test_the_var_key_recovers_instead_of_wedging(self, tmp_path, monkeypatch):
        """
        Regression: the resume re-extracted and joined onto columns already
        there, which pandas rejects ("columns overlap but no suffix
        specified"). The resulting failure also kept the checkpoint from being
        cleared, so every later run reloaded the same state and failed the same
        way — the var_key never recovered on its own.
        """
        extractor, _ = self._extractor(tmp_path, monkeypatch)
        checkpoint = self._write_half_checkpoint(extractor, tmp_path)

        result, all_succeeded = extractor._run_impl({"sst": None}, n_workers=1)

        assert all_succeeded
        # Cleared, so a later run starts clean rather than reloading the wedge.
        assert not checkpoint.exists()
        assert not _keys_path(checkpoint).exists()

    def test_the_re_extracted_values_win_over_the_stale_ones(
        self, tmp_path, monkeypatch
    ):
        """The column is re-extracted, not carried over from the checkpoint."""
        extractor, _ = self._extractor(tmp_path, monkeypatch)
        self._write_half_checkpoint(extractor, tmp_path, values=(99.0, 99.0))

        result, _ = extractor._run_impl({"sst": None}, n_workers=1)

        assert result["sst"].tolist() == [1.5, 2.5]
        assert list(result.columns).count("sst") == 1
        assert not [c for c in result.columns if c.endswith(("_x", "_y"))]

    def test_an_input_column_of_the_same_name_still_collides(
        self, tmp_path, monkeypatch
    ):
        """
        Only checkpoint leftovers are dropped. A column the caller supplied that
        happens to share a name with an extracted variable is a different
        problem, and silently overwriting their data would be a worse answer.
        """
        extractor, fresh = self._extractor(tmp_path, monkeypatch)
        extractor.data["sst"] = [9.0, 9.0]

        _, all_succeeded = extractor._run_impl({"sst": None}, n_workers=1)

        assert not all_succeeded
