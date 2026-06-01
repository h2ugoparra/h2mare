"""Tests for storage/parquet_plotter.py — ParquetPlotter methods."""

from unittest.mock import MagicMock, patch

import plotly.graph_objects as go
import polars as pl
import pytest

from h2mare.storage.parquet_plotter import ParquetPlotter

# ---------------------------------------------------------------------------
# _agg_key
# ---------------------------------------------------------------------------


class TestAggKey:
    def test_returns_tuple(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        key = plotter._agg_key("sst", "month", None, None)
        assert isinstance(key, tuple)

    def test_different_inputs_produce_different_keys(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        k1 = plotter._agg_key("sst", "month", None, None)
        k2 = plotter._agg_key("sst", "year", None, None)
        assert k1 != k2

    def test_list_dates_converted_to_tuple(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        k = plotter._agg_key("sst", "month", ["2020-01-01", "2020-06-01"], None)
        # dates_key should be a tuple so the whole key is hashable
        assert isinstance(k, tuple)


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_empties_cache(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        plotter._cache["dummy_key"] = "dummy_value"
        plotter.clear_cache()
        assert plotter._cache == {}

    def test_clear_on_empty_cache_is_no_op(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        plotter.clear_cache()
        assert plotter._cache == {}


# ---------------------------------------------------------------------------
# _snap_to_grid
# ---------------------------------------------------------------------------


class TestSnapToGrid:
    def test_returns_nearest_cell(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        plotter._grid_coords = pl.DataFrame(
            {
                "lon": [-10.0, -5.0, 0.0],
                "lat": [30.0, 35.0, 40.0],
            }
        )
        # lon=-7 → nearest is -5; lat=32 → nearest is 30
        result = plotter._snap_to_grid((-7.0, 32.0))
        assert result == (-5.0, 30.0, -5.0, 30.0)

    def test_returns_four_element_tuple(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        plotter._grid_coords = pl.DataFrame(
            {
                "lon": [-10.0, -5.0],
                "lat": [30.0, 35.0],
            }
        )
        result = plotter._snap_to_grid((-10.0, 35.0))
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _get_agg_df
# ---------------------------------------------------------------------------


class TestGetAggDf:
    def test_caches_result_on_second_call(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        fake_df = pl.DataFrame({"time_agg": [1], "sst": [20.0]})
        mock_lf = MagicMock()
        mock_lf.collect.return_value = fake_df

        with patch(
            "h2mare.storage.parquet_plotter.aggregate_by_space_time",
            return_value=mock_lf,
        ) as mock_agg:
            plotter._get_agg_df("sst", "month", None, None)
            plotter._get_agg_df("sst", "month", None, None)

        assert mock_agg.call_count == 1

    def test_returns_dataframe(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        result = plotter._get_agg_df("sst", "month", None, None)
        assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# time_series
# ---------------------------------------------------------------------------


class TestTimeSeries:
    def test_returns_plotly_figure(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {}
            fig = plotter.time_series("sst", "month")
        assert isinstance(fig, go.Figure)

    def test_unknown_variable_raises_value_error(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with pytest.raises(ValueError, match="not in parquet"):
            plotter.time_series("bad_var", "month")

    def test_figure_has_one_trace(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {
                "long_name": "SST"
            }
            fig = plotter.time_series("sst", "month")
        assert len(fig.data) == 1

    def test_single_var_uses_full_domain_and_no_second_axis(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {}
            fig = plotter.time_series("sst", "month")
        assert tuple(fig.layout.xaxis.domain) == (0.0, 1.0)
        # No additional axes configured for a single variable
        assert "yaxis2" not in fig.layout.to_plotly_json()


class TestTimeSeriesMultiVar:
    """Multi-variable / multi-y-axis behavior of time_series."""

    @staticmethod
    def _plot(plotter, var_name, **kwargs):
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {}
            return plotter.time_series(var_name, "month", **kwargs)

    def test_one_trace_per_variable(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl", "mld"])
        assert len(fig.data) == 3

    def test_traces_bound_to_distinct_axes(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl", "mld", "adt"])
        assert [t.yaxis for t in fig.data] == ["y", "y2", "y3", "y4"]

    def test_second_axis_on_right_overlaying_first(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl"])
        assert fig.layout.yaxis2.side == "right"
        assert fig.layout.yaxis2.overlaying == "y"

    def test_third_axis_floated_left(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl", "mld"])
        assert fig.layout.yaxis3.side == "left"
        assert fig.layout.yaxis3.anchor == "free"
        assert fig.layout.yaxis3.position == 0.0
        # Plot area shrinks on the left to make room for the floated axis
        assert tuple(fig.layout.xaxis.domain) == (0.08, 1.0)

    def test_fourth_axis_floated_right_and_domain_shrinks_both_sides(
        self, multivar_indexer
    ):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl", "mld", "adt"])
        assert fig.layout.yaxis4.side == "right"
        assert fig.layout.yaxis4.anchor == "free"
        assert fig.layout.yaxis4.position == 1.0
        assert tuple(fig.layout.xaxis.domain) == (0.08, 0.92)

    def test_more_than_four_variables_raises(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        with pytest.raises(ValueError, match="at most 4 variables"):
            self._plot(plotter, ["sst", "chl", "mld", "adt", "extra"])

    def test_empty_list_raises(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        with pytest.raises(ValueError, match="at least one variable"):
            self._plot(plotter, [])

    def test_unknown_variable_in_list_raises(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        with pytest.raises(ValueError, match="not in parquet"):
            self._plot(plotter, ["sst", "bad_var"])

    def test_custom_title_overrides_default(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {}
            fig = plotter.time_series(["sst", "chl"], "month", title="My Comparison")
        assert fig.layout.title.text == "My Comparison"

    def test_default_title_for_multiple_vars(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl"])
        assert fig.layout.title.text == "Time series"

    def test_background_is_white(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        fig = self._plot(plotter, ["sst", "chl"])
        assert fig.layout.plot_bgcolor == "white"
        assert fig.layout.paper_bgcolor == "white"

    def test_season_aggregation_builds_figure(self, multivar_indexer):
        plotter = ParquetPlotter(multivar_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {}
            fig = plotter.time_series(["sst", "chl"], "season")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


# ---------------------------------------------------------------------------
# stats_summary
# ---------------------------------------------------------------------------


class TestStatsSummary:
    def test_returns_plotly_figure(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {
                "long_name": "SST"
            }
            fig = plotter.stats_summary("sst", "day")
        assert isinstance(fig, go.Figure)

    def test_custom_title_overrides_default(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {
                "long_name": "SST"
            }
            fig = plotter.stats_summary("sst", "day", title="Custom Stats")
        assert fig.layout.title.text == "Custom Stats"

    def test_default_title_mentions_statistics_summary(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.get_settings") as mock_get_settings:
            mock_get_settings.return_value.get_var_info.return_value = {
                "long_name": "SST"
            }
            fig = plotter.stats_summary("sst", "day")
        assert "Statistics Summary" in fig.layout.title.text


# ---------------------------------------------------------------------------
# show / PLOT_CONFIG
# ---------------------------------------------------------------------------


class TestShow:
    def test_plot_config_pins_modebar(self):
        assert ParquetPlotter.PLOT_CONFIG["displayModeBar"] is True
        assert ParquetPlotter.PLOT_CONFIG["displaylogo"] is False

    def test_show_passes_plot_config(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        fig = MagicMock()
        plotter.show(fig)
        fig.show.assert_called_once_with(
            renderer=None, config=ParquetPlotter.PLOT_CONFIG
        )

    def test_show_allows_config_overrides(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        fig = MagicMock()
        plotter.show(fig, displaylogo=True)
        _, kwargs = fig.show.call_args
        assert kwargs["config"]["displaylogo"] is True
        assert kwargs["config"]["displayModeBar"] is True

    def test_show_passes_renderer(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        fig = MagicMock()
        plotter.show(fig, renderer="browser")
        _, kwargs = fig.show.call_args
        assert kwargs["renderer"] == "browser"
        assert kwargs["config"] == ParquetPlotter.PLOT_CONFIG


# ---------------------------------------------------------------------------
# spatial_maps
# ---------------------------------------------------------------------------


class TestSpatialMaps:
    def test_unknown_variable_raises_value_error(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with pytest.raises(ValueError, match="not in parquet"):
            plotter.spatial_maps("bad_var")

    def test_calls_plot_maps_with_correct_var(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.plot_maps") as mock_plot_maps:
            plotter.spatial_maps("sst")
        mock_plot_maps.assert_called_once()
        call_kwargs = mock_plot_maps.call_args
        assert call_kwargs.args[1] == "sst"

    def test_passes_kwargs_to_plot_maps(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.plot_maps") as mock_plot_maps:
            plotter.spatial_maps("sst", agg_by="season", cmap="viridis")
        _, kwargs = mock_plot_maps.call_args
        assert kwargs["cmap"] == "viridis"
        assert kwargs["agg_by"] == "season"

    def test_title_forwarded_as_main_title(self, loaded_indexer):
        plotter = ParquetPlotter(loaded_indexer)
        with patch("h2mare.storage.parquet_plotter.plot_maps") as mock_plot_maps:
            plotter.spatial_maps("sst", title="My Maps")
        _, kwargs = mock_plot_maps.call_args
        assert kwargs["main_title"] == "My Maps"
