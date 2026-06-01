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
