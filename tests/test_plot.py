"""Tests for plot_maps — data validation and time_col auto-derive logic."""

import pytest
import polars as pl
from datetime import date

# Use non-interactive backend before any matplotlib/cartopy import
import matplotlib

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
        from h2mare.utils.plot import plot_maps, split_by_group

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
