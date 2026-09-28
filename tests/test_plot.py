"""Tests for plot_maps — data validation and time_col auto-derive logic."""

from datetime import date

# Use non-interactive backend before any matplotlib/cartopy import
import matplotlib
import polars as pl
import pytest

matplotlib.use("Agg")

cartopy = pytest.importorskip("cartopy", reason="cartopy not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monthly_df(n_months: int = 3) -> pl.DataFrame:
    """Gridded df with a pre-computed 'month' column (no 'time' column)."""
    rows = [
        {"month": m, "lon": lon, "lat": lat, "sst": float(20 + m)}
        for m in range(1, n_months + 1)
        for lon in [-10.0, -5.0, 0.0]
        for lat in [30.0, 35.0, 40.0]
    ]
    return pl.DataFrame(rows)


def _timed_df(months: list[int] | None = None) -> pl.DataFrame:
    """Gridded df with a Date 'time' column but no 'month' or 'season' column."""
    if months is None:
        months = [1, 2, 3]
    rows = [
        {"time": date(2020, m, 15), "lon": lon, "lat": lat, "sst": float(20 + m)}
        for m in months
        for lon in [-10.0, -5.0, 0.0]
        for lat in [30.0, 35.0, 40.0]
    ]
    return pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Date))


# ---------------------------------------------------------------------------
# plot_maps — error handling
# ---------------------------------------------------------------------------


class TestPlotMapsErrors:
    def test_raises_on_empty_df(self):
        from h2mare.utils.plot import plot_maps

        with pytest.raises(ValueError, match="No data"):
            plot_maps(pl.DataFrame(), "sst", agg_by="month")

    def test_raises_on_missing_var(self):
        from h2mare.utils.plot import plot_maps

        df = _monthly_df()
        with pytest.raises(Exception):
            plot_maps(df, "chl", agg_by="month")

    def test_raises_when_group_col_absent_and_time_col_missing(self):
        """Neither 'month' nor the specified time_col present → ValueError."""
        from h2mare.utils.plot import plot_maps

        # df has no 'month' and no 'time' column
        df = _monthly_df()  # has 'month', drop it; result has no 'time' either
        df_no_group = df.drop("month")
        with pytest.raises(Exception):
            plot_maps(df_no_group, "sst", agg_by="month", time_col="time")


# ---------------------------------------------------------------------------
# plot_maps — time_col auto-derive
# ---------------------------------------------------------------------------


class TestPlotMapsAutoDerive:
    def test_month_derived_from_time_col(self, tmp_path):
        """month column is derived from time_col when absent."""
        from h2mare.utils.plot import plot_maps

        df = _timed_df(months=[1, 2, 3])
        save = tmp_path / "month.png"
        plot_maps(df, "sst", agg_by="month", time_col="time", save_path=save)
        assert save.exists()

    def test_season_derived_from_time_col(self, tmp_path):
        """season column is derived with correct meteorological labels."""
        from h2mare.utils.plot import plot_maps

        # One month per season: Feb(winter), May(spring), Aug(summer), Nov(autumn)
        df = _timed_df(months=[2, 5, 8, 11])
        save = tmp_path / "season.png"
        plot_maps(df, "sst", agg_by="season", time_col="time", save_path=save)
        assert save.exists()

    def test_precomputed_group_col_used_directly(self, tmp_path):
        """When agg_by column already present, time_col is not needed."""
        from h2mare.utils.plot import plot_maps

        df = _monthly_df(n_months=3)  # has 'month', no 'time'
        save = tmp_path / "precomputed.png"
        # time_col default is 'time', but 'time' absent — should NOT raise
        # because 'month' is already present
        plot_maps(df, "sst", agg_by="month", save_path=save)
        assert save.exists()

    def test_season_labels_correct(self):
        """Derived season values match meteorological convention."""
        from h2mare.utils.plot import split_by_group

        month_to_season = {
            12: "winter",
            1: "winter",
            2: "winter",
            3: "spring",
            4: "spring",
            5: "spring",
            6: "summer",
            7: "summer",
            8: "summer",
            9: "autumn",
            10: "autumn",
            11: "autumn",
        }
        for month, expected_season in month_to_season.items():
            rows = [
                {
                    "time": date(2020, month if month != 12 else 12, 1),
                    "lon": lon,
                    "lat": lat,
                    "sst": 20.0,
                }
                for lon in [-10.0, -5.0, 0.0]
                for lat in [30.0, 35.0, 40.0]
            ]
            df = pl.DataFrame(rows).with_columns(pl.col("time").cast(pl.Date))
            df = df.with_columns(
                pl.when(pl.col("time").dt.month().is_in([12, 1, 2]))
                .then(pl.lit("winter"))
                .when(pl.col("time").dt.month().is_in([3, 4, 5]))
                .then(pl.lit("spring"))
                .when(pl.col("time").dt.month().is_in([6, 7, 8]))
                .then(pl.lit("summer"))
                .otherwise(pl.lit("autumn"))
                .alias("season")
            )
            groups = split_by_group(df, "season")
            assert expected_season in groups, (
                f"Month {month} should map to '{expected_season}', got {list(groups.keys())}"
            )

    def test_infers_bbox_from_data(self, tmp_path):
        """data_bbox=None infers extent from data without error."""
        from h2mare.utils.plot import plot_maps

        df = _timed_df()
        save = tmp_path / "bbox_inferred.png"
        plot_maps(
            df, "sst", agg_by="month", time_col="time", data_bbox=None, save_path=save
        )
        assert save.exists()

    def test_explicit_bbox_used(self, tmp_path):
        """Explicit data_bbox overrides data-derived extent."""
        from h2mare.utils.plot import plot_maps

        df = _timed_df()
        save = tmp_path / "bbox_explicit.png"
        plot_maps(
            df,
            "sst",
            agg_by="month",
            time_col="time",
            data_bbox=(-15.0, 25.0, 5.0, 45.0),
            save_path=save,
        )
        assert save.exists()

    def test_explicit_vminmax_used(self, tmp_path):
        """Explicit vminmax overrides data-derived min/max."""
        from h2mare.utils.plot import plot_maps

        df = _timed_df()
        save = tmp_path / "vminmax.png"
        plot_maps(
            df,
            "sst",
            agg_by="month",
            time_col="time",
            vminmax=(15.0, 30.0),
            save_path=save,
        )
        assert save.exists()


# ---------------------------------------------------------------------------
# field_for_plot — collapsing a date's field down to something plottable
# ---------------------------------------------------------------------------


def _field_ds(n_times: int) -> "xr.Dataset":  # noqa: F821
    """(time, lat, lon) dataset for one calendar day, with `n_times` steps."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    return xr.Dataset(
        {"u10": (("time", "lat", "lon"), np.ones((n_times, 2, 2)))},
        coords={
            "time": pd.date_range("2020-01-01", periods=n_times, freq="h"),
            "lat": [30.0, 31.0],
            "lon": [-10.0, -9.0],
        },
    )


class TestFieldForPlot:
    """
    open_dataset selects by calendar day, so an hourly store hands back all 24
    steps and the array reaching .plot() is 3-D. It is reduced here, and the
    note is what keeps the plot from quietly misreporting what it shows.
    """

    def test_hourly_day_is_averaged_and_says_so(self):
        from h2mare.utils.plot import field_for_plot

        field, note = field_for_plot(_field_ds(24), "u10")

        assert set(field.dims) == {"lat", "lon"}
        assert note == "u10: mean of 24 steps"

    def test_single_step_is_squeezed_silently(self):
        """A daily store needs no note — there was nothing to collapse."""
        from h2mare.utils.plot import field_for_plot

        field, note = field_for_plot(_field_ds(1), "u10")

        assert set(field.dims) == {"lat", "lon"}
        assert note == ""

    def test_missing_var_blames_the_file_not_the_request(self):
        """
        Reachable after routing: get_variables unions across a store's files, so
        a variable added later is 'in the store' but absent from an older file.
        """
        from h2mare.utils.plot import field_for_plot

        with pytest.raises(KeyError, match="added to the store"):
            field_for_plot(_field_ds(24), "wind_mean")


# ---------------------------------------------------------------------------
# plot_records_on_field — a record the store has no file for
# ---------------------------------------------------------------------------


class TestUncoveredRecordsAreSkipped:
    """
    The docstring has always promised these are skipped with a warning, but
    open_dataset raises rather than returning None, so the `is None` check that
    stood here never fired and the first uncovered record ended the loop with
    every later one undrawn.
    """

    def _patch(self, monkeypatch, missing_date: str) -> list:
        """Catalog answering every date but *missing_date*; returns figures drawn."""
        from unittest.mock import MagicMock

        import pandas as pd

        from h2mare.utils import plot as plot_module

        catalog = MagicMock()
        catalog.var_key = "atm-instante"

        def _open(dates=None, **_kw):
            if pd.Timestamp(dates) == pd.Timestamp(missing_date):
                raise FileNotFoundError(f"No zarr files contain dates: [{dates}]")
            return _field_ds(1)

        catalog.open_dataset.side_effect = _open
        monkeypatch.setattr(plot_module, "catalog_for_var", lambda *a, **k: catalog)

        drawn: list = []
        monkeypatch.setattr(plot_module.plt, "show", lambda *a, **k: drawn.append(1))
        return drawn

    def _records(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "lon": [-9.0, -8.0, -7.0],
                "lat": [31.0, 32.0, 33.0],
            }
        )

    def test_the_loop_survives_and_draws_the_rest(self, monkeypatch):
        from h2mare.utils.plot import plot_records_on_field

        drawn = self._patch(monkeypatch, missing_date="2020-01-02")

        plot_records_on_field(self._records(), "atm-instante", var="u10")

        assert len(drawn) == 2

    def test_every_record_missing_draws_nothing_and_does_not_raise(self, monkeypatch):
        from h2mare.utils.plot import plot_records_on_field

        drawn = self._patch(monkeypatch, missing_date="2020-01-01")
        one = self._records().head(1)

        plot_records_on_field(one, "atm-instante", var="u10")

        assert drawn == []


class TestDepthSliceReadFromNativeStore:
    """
    ``thetao_0`` is a name only extraction and compile produce; the native store
    holds ``thetao`` on a depth axis. Routed as-is it went to the compiled store,
    so a project that never compiles could not plot any depth-sliced column.
    """

    def _patch(self, monkeypatch) -> tuple[list, list]:
        """Config with dyn_rep.thetao at [0, 110]; returns (routes, plotted ds)."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import numpy as np
        import pandas as pd
        import xarray as xr

        from h2mare.utils import plot as plot_module

        var_cfg = SimpleNamespace(depth_levels={"thetao": [0, 110]})
        settings = SimpleNamespace(
            app_config=SimpleNamespace(variables={"dyn_rep": var_cfg})
        )
        monkeypatch.setattr(plot_module, "get_settings", lambda: settings)

        # depth 0 holds 1s, depth 110 holds 2s, so the slice taken is visible.
        ds = xr.Dataset(
            {
                "thetao": (
                    ("time", "depth", "lat", "lon"),
                    np.stack([np.ones((2, 2)), np.full((2, 2), 2.0)])[None],
                )
            },
            coords={
                "time": pd.to_datetime(["2020-01-01"]),
                "depth": [0.49, 109.7],
                "lat": [30.0, 31.0],
                "lon": [-10.0, -9.0],
            },
        )
        catalog = MagicMock()
        catalog.open_dataset.return_value = ds

        routes: list = []

        def _route(var, var_key, read_from="auto"):
            routes.append((var, var_key, read_from))
            return catalog

        monkeypatch.setattr(plot_module, "catalog_for_var", _route)

        plotted: list = []
        real_field_for_plot = plot_module.field_for_plot

        def _field(ds, var):
            plotted.append(ds)
            return real_field_for_plot(ds, var)

        monkeypatch.setattr(plot_module, "field_for_plot", _field)
        monkeypatch.setattr(plot_module.plt, "show", lambda *a, **k: None)
        return routes, plotted

    def _records(self):
        import pandas as pd

        return pd.DataFrame(
            {"date": pd.to_datetime(["2020-01-01"]), "lon": [-9.5], "lat": [30.5]}
        )

    def test_slice_is_cut_from_the_native_variable(self, monkeypatch):
        from h2mare.utils.plot import plot_records_on_field

        routes, plotted = self._patch(monkeypatch)

        plot_records_on_field(self._records(), "dyn_rep", var="thetao_110")

        assert routes == [("thetao", "dyn_rep", "native")]
        assert list(plotted[0].data_vars) == ["thetao_110"]
        assert float(plotted[0]["thetao_110"].max()) == 2.0

    def test_derived_slice_is_routed_unchanged(self, monkeypatch):
        from h2mare.utils.plot import plot_records_on_field

        routes, _ = self._patch(monkeypatch)

        # ke has no native variable, so it cannot be cut here.
        with pytest.raises(KeyError):
            plot_records_on_field(self._records(), "dyn_rep", var="ke_0")

        assert routes == [("ke_0", "dyn_rep", "auto")]

    def test_compiled_read_from_still_asks_for_the_h2ds_column(self, monkeypatch):
        from h2mare.utils.plot import plot_records_on_field

        routes, _ = self._patch(monkeypatch)

        with pytest.raises(KeyError):
            plot_records_on_field(
                self._records(), "dyn_rep", var="thetao_0", read_from="compiled"
            )

        assert routes == [("thetao_0", "dyn_rep", "compiled")]


class TestNoNotebookOnlyImportsAtModuleScope:
    """
    IPython is needed by one function (``animate_vars``, which renders through
    ``IPython.display`` and only works in a notebook). At module scope it made
    a heavyweight notebook-only dependency a hard requirement of importing any
    plotting helper, including in headless pipeline runs.
    """

    def test_ipython_is_not_imported_at_module_scope(self):
        import ast
        import pathlib

        import h2mare.utils.plot as plot_mod

        tree = ast.parse(pathlib.Path(plot_mod.__file__).read_text(encoding="utf-8"))
        top_level = [
            n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        names = {
            (n.module or "") if isinstance(n, ast.ImportFrom) else n.names[0].name
            for n in top_level
        }
        offenders = {n for n in names if n.split(".")[0] == "IPython"}
        assert not offenders, f"notebook-only import at module scope: {offenders}"

    def test_animate_vars_still_reaches_ipython(self):
        """The import moved, it did not disappear — the function still needs it."""
        import inspect

        from h2mare.utils.plot import animate_vars

        src = inspect.getsource(animate_vars)
        assert "from IPython.display import" in src
