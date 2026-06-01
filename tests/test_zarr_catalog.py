"""Tests for ZarrCatalog: build_file_path, dataset column, and provenance sidecars."""

import json

import msgspec
import numpy as np
import pandas as pd
import xarray as xr

from h2mare.models import AppConfig
from h2mare.storage.zarr_catalog import ZarrCatalog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "variables": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "pattern": r".*\.nc",
}


def _make_app_config() -> AppConfig:
    return msgspec.convert(
        {"variables": {"sst": _ENTRY}, "secrets": {}},
        AppConfig,
    )


def _make_catalog(tmp_path) -> ZarrCatalog:
    return ZarrCatalog(
        "sst",
        app_config=_make_app_config(),
        store_root=tmp_path,
        auto_refresh=False,
    )


def _make_ds(start: str = "2020-01-01", n_days: int = 5) -> xr.Dataset:
    times = pd.date_range(start, periods=n_days, freq="D")
    data = np.ones((n_days, 3, 3))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


# ---------------------------------------------------------------------------
# build_file_path
# ---------------------------------------------------------------------------


class TestBuildFilePath:
    def test_default_uses_var_key(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert "sst" in path.name

    def test_default_includes_source(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert "cmems" in path.name

    def test_default_ends_with_zarr(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert path.suffix == ".zarr"

    def test_name_key_replaces_var_key(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(
            _make_ds(), "year", name_key="cmems_mod_glo_phy_my"
        )
        assert "cmems_mod_glo_phy_my" in path.name

    def test_name_key_excludes_var_key(self, tmp_path):
        """When name_key is provided, var_key (sst) must not appear in the stem."""
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year", name_key="override-id")
        # stem: "cmems_override-id_<label>" — sst should not be in it
        assert "sst" not in path.stem

    def test_store_root_override(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        other_root = tmp_path / "other"
        path = catalog.build_file_path(_make_ds(), "year", store_root=other_root)
        assert path.parent == other_root

    def test_year_format_contains_year(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds("2021-03-01"), "year")
        assert "2021" in path.name

    def test_yearmonth_format_contains_month(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds("2021-03-01"), "yearmonth")
        assert "2021" in path.name
        assert "03" in path.name


# ---------------------------------------------------------------------------
# Helpers for dataset / provenance tests
# ---------------------------------------------------------------------------

_ENTRY_WITH_NRT = {
    **_ENTRY,
    "dataset_id_nrt": "cmems_obs-sl_glo_phy-ssh_nrt",
}


def _make_app_config_with_nrt() -> AppConfig:
    return msgspec.convert(
        {"variables": {"sst": _ENTRY_WITH_NRT}, "secrets": {}},
        AppConfig,
    )


def _make_catalog_with_nrt(tmp_path) -> ZarrCatalog:
    return ZarrCatalog(
        "sst",
        app_config=_make_app_config_with_nrt(),
        store_root=tmp_path,
        auto_refresh=False,
    )


def _write_zarr(store_root, ds, name="test.zarr"):
    """Write a consolidated zarr so xr.open_zarr(consolidated=True) works in tests."""
    path = store_root / name
    ds.to_zarr(path, consolidated=True)
    return path


def _two_row_df(zarr_path) -> pd.DataFrame:
    """Minimal DataFrame with rep and nrt rows for the same zarr path."""
    p = str(zarr_path)
    return pd.DataFrame(
        [
            {
                "path": p,
                "filename": zarr_path.name,
                "start_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2023-06-30"),
            },
            {
                "path": p,
                "filename": zarr_path.name,
                "start_date": pd.Timestamp("2023-07-01"),
                "end_date": pd.Timestamp("2023-12-31"),
            },
        ]
    )


# ---------------------------------------------------------------------------
# dataset column and provenance sidecar tests
# ---------------------------------------------------------------------------


class TestDatasetColumn:
    def test_no_sidecar_returns_single_row_with_rep_id(self, tmp_path):
        ds = _make_ds("2023-01-01", n_days=10)
        zarr_path = _write_zarr(tmp_path, ds)
        catalog = _make_catalog(tmp_path)

        rows = catalog._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 1
        assert rows[0]["dataset"] == _ENTRY["dataset_id_rep"]

    def test_zarr_attrs_two_sources_returns_two_rows(self, tmp_path):
        import zarr

        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        sources = [
            {
                "dataset_id": "REP_ID",
                "dataset_type": "rep",
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
            },
            {
                "dataset_id": "NRT_ID",
                "dataset_type": "nrt",
                "start_date": "2023-07-01",
                "end_date": "2023-12-31",
            },
        ]
        root = zarr.open_group(str(zarr_path), mode="r+")
        root.attrs["source_datasets"] = json.dumps(sources)
        zarr.consolidate_metadata(str(zarr_path))
        catalog = _make_catalog(tmp_path)

        rows = catalog._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 2
        assert rows[0]["dataset"] == "REP_ID"
        assert rows[1]["dataset"] == "NRT_ID"
        assert rows[0]["end_date"] < rows[1]["start_date"]

    def test_sidecar_fallback_still_works_for_old_files(self, tmp_path):
        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        sidecar = zarr_path.parent / (zarr_path.stem + "_prov.json")
        sidecar.write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "REP_ID",
                        "dataset_type": "rep",
                        "start_date": "2023-01-01",
                        "end_date": "2023-06-30",
                    },
                    {
                        "dataset_id": "NRT_ID",
                        "dataset_type": "nrt",
                        "start_date": "2023-07-01",
                        "end_date": "2023-12-31",
                    },
                ]
            )
        )
        catalog = _make_catalog(tmp_path)

        rows = catalog._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 2
        assert rows[0]["dataset"] == "REP_ID"
        assert rows[1]["dataset"] == "NRT_ID"

    def test_get_paths_in_range_deduplicates(self, tmp_path):
        zarr_path = tmp_path / "dummy.zarr"
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _two_row_df(zarr_path)

        result = catalog.get_paths_in_range("2023-01-01", "2023-12-31")

        assert result == [str(zarr_path)]

    def test_map_dates_to_paths_with_split_rows(self, tmp_path):
        zarr_path = tmp_path / "dummy.zarr"
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _two_row_df(zarr_path)

        result = catalog.map_dates_to_paths(["2023-03-15", "2023-09-20"])

        assert set(result.keys()) == {str(zarr_path)}
        assert len(result[str(zarr_path)]) == 2

    def test_load_from_disk_adds_dataset_column_for_old_parquet(self, tmp_path):
        old_df = pd.DataFrame(
            [
                {
                    "path": "/p/a.zarr",
                    "filename": "a.zarr",
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2020-12-31"),
                }
            ]
        )
        catalog = _make_catalog(tmp_path)
        catalog.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        old_df.to_parquet(catalog.catalog_path, index=False)

        df = catalog._load_from_disk()

        assert "dataset" in df.columns
        assert (df["dataset"] == _ENTRY["dataset_id_rep"]).all()

    def test_backfill_provenance_splits_boundary_file(self, tmp_path):
        import zarr

        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        catalog = _make_catalog_with_nrt(tmp_path)

        n = catalog.backfill_provenance("2023-06-30")

        assert n == 1
        # Provenance must be in zarr attrs, not a sidecar file
        root = zarr.open_group(str(zarr_path), mode="r")
        sources = json.loads(root.attrs["source_datasets"])
        assert len(sources) == 2
        assert sources[0]["dataset_type"] == "rep"
        assert sources[1]["dataset_type"] == "nrt"
        prov_file = zarr_path.parent / (zarr_path.stem + "_prov.json")
        assert not prov_file.exists()
        df = catalog.df
        assert df["path"].nunique() == 1
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Variable coverage: non-null end + ndarray-cell handling
# ---------------------------------------------------------------------------


def _make_padded_ds(start: str, n_days: int, n_valid: int, var: str) -> xr.Dataset:
    """Dataset where *var* has real data for the first *n_valid* days, NaN after."""
    times = pd.date_range(start, periods=n_days, freq="D")
    data = np.ones((n_days, 3, 3))
    data[n_valid:] = np.nan
    return xr.Dataset(
        {var: (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
    )


def _df_for_zarr(zarr_path, variables, end_date) -> pd.DataFrame:
    """Single-row catalog df. ``variables`` may be a list or ndarray on purpose."""
    return pd.DataFrame(
        [
            {
                "path": str(zarr_path),
                "filename": zarr_path.name,
                "variables": variables,
                "start_date": pd.Timestamp("2026-01-01"),
                "end_date": pd.Timestamp(end_date),
            }
        ]
    )


class TestVarsNonnullEnd:
    def test_nonnull_end_ignores_nan_tail(self, tmp_path):
        # ac_amp has data for 10 days then NaN-padded to 15 (mimics a lagging
        # variable padded out to the global end by the compiler's outer merge).
        ds = _make_padded_ds("2026-05-01", n_days=15, n_valid=10, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _df_for_zarr(zarr_path, ["ac_amp"], "2026-05-15")

        result = catalog.get_vars_nonnull_end(["ac_amp"])

        # Real extent is day 10 (2026-05-10), not the file end (2026-05-15).
        assert result["ac_amp"] == pd.Timestamp("2026-05-10")

    def test_nonnull_end_handles_ndarray_variables_cell(self, tmp_path):
        # Catalogs loaded from Parquet hold the variables cell as an ndarray.
        ds = _make_padded_ds("2026-05-01", n_days=15, n_valid=12, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _df_for_zarr(zarr_path, np.array(["ac_amp"]), "2026-05-15")

        result = catalog.get_vars_nonnull_end(["ac_amp"])

        assert result["ac_amp"] == pd.Timestamp("2026-05-12")

    def test_absent_variable_omitted(self, tmp_path):
        ds = _make_padded_ds("2026-05-01", n_days=5, n_valid=5, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _df_for_zarr(zarr_path, ["ac_amp"], "2026-05-05")

        assert catalog.get_vars_nonnull_end(["not_a_var"]) == {}

    def test_get_var_time_coverage_handles_ndarray_cell(self, tmp_path):
        # Regression: ndarray cells previously made membership checks always
        # false, so this silently returned None for disk-loaded catalogs.
        catalog = _make_catalog(tmp_path)
        catalog._df_cache = _df_for_zarr(
            tmp_path / "h2ds_2026.zarr", np.array(["ac_amp", "c_amp"]), "2026-05-15"
        )

        cov = catalog.get_var_time_coverage("ac_amp")

        assert cov is not None
        assert cov.end == pd.Timestamp("2026-05-15")
