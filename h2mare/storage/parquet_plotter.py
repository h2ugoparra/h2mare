"""Visualization layer for ParquetIndexer data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import plotly.graph_objects as go
import polars as pl
from loguru import logger

from h2mare import settings
from h2mare.storage.parquet_helpers import aggregate_by_space_time, aggregate_by_time
from h2mare.utils.plot import plot_maps

if TYPE_CHECKING:
    from h2mare.storage.parquet_indexer import ParquetIndexer


class ParquetPlotter:
    """
    Visualization accessor for ParquetIndexer.

    Accessed via ``indexer.plot``. Do not instantiate directly.

    Example:
        >>> indexer.plot.time_series("sst", agg_by="month")
        >>> indexer.plot.monthly_map("sst")
    """

    def __init__(self, indexer: ParquetIndexer) -> None:
        self._idx = indexer
        self._cache: dict = {}

    def _snap_to_grid(self, point: tuple[float, float]) -> tuple[float, float, float, float]:
        lon, lat = point
        lon_col = self._idx.lon_col
        lat_col = self._idx.lat_col
        coords = (
            self._idx.scan(columns=[lon_col, lat_col])
            .select([lon_col, lat_col])
            .unique()
            .collect()
        )
        lons = coords[lon_col].unique()
        lats = coords[lat_col].unique()
        nearest_lon = float(lons[(lons - lon).abs().arg_min()])
        nearest_lat = float(lats[(lats - lat).abs().arg_min()])
        logger.debug(f"Point ({lon}, {lat}) snapped to grid cell ({nearest_lon}, {nearest_lat})")
        return (nearest_lon, nearest_lat, nearest_lon, nearest_lat)

    def _agg_key(self, var_name, agg_by, dates, bbox) -> tuple:
        dates_key = tuple(dates) if isinstance(dates, list) else dates
        return (var_name, agg_by, dates_key, bbox)

    def _get_agg_df(self, var_name, agg_by, dates, bbox) -> "pl.DataFrame":
        key = self._agg_key(var_name, agg_by, dates, bbox)
        if key not in self._cache:
            lon_col = self._idx.lon_col
            lat_col = self._idx.lat_col
            lf = self._idx.scan(
                dates=dates,
                bbox=bbox,
                columns=[self._idx.time_col, lon_col, lat_col, var_name],
            )
            self._cache[key] = aggregate_by_space_time(
                lf,
                var_name,
                agg_by=agg_by,
                time_col=self._idx.time_col,
                lon_col=lon_col,
                lat_col=lat_col,
            ).collect()
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the aggregation cache (e.g. after new data is added)."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    #  Time series                                                         #
    # ------------------------------------------------------------------ #

    def time_series(
        self,
        var_name: str,
        agg_by: Literal["day", "week", "month", "season", "year"],
        *,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float] | tuple[float, float, float, float]] = None,
    ) -> go.Figure:
        """
        Interactive time series line plot aggregated over space and time.

        Args:
            var_name: Variable name to plot.
            agg_by: Temporal aggregation granularity.
            dates: Discrete list of dates or (start, end) range. Defaults to full dataset.
            bbox: Spatial filter. Either a 4-tuple (xmin, ymin, xmax, ymax) for an extent
                or a 2-tuple (lon, lat) to select the nearest grid cell. Defaults to full extent.

        Note:
            Seasonal values are assigned to the first month of the respective season
            (e.g. spring → March 1st).

        Returns:
            go.Figure
        """
        if var_name not in self._idx.get_schema():
            raise ValueError(f"'{var_name}' not in parquet column names.")

        if bbox is not None and len(bbox) == 2:
            bbox = self._snap_to_grid(bbox)  # type: ignore[arg-type]

        lf = self._idx.scan(
            dates=dates, bbox=bbox, columns=[self._idx.time_col, var_name]
        )
        result = aggregate_by_time(lf, var_name, agg_by=agg_by)

        # Season encoding is YYYYS — convert to plottable dates
        if agg_by == "season":
            result = result.with_columns(
                time_plot=pl.datetime(
                    pl.col("time_agg") // 10,
                    pl.when(pl.col("time_agg") % 10 == 1)
                    .then(3)
                    .when(pl.col("time_agg") % 10 == 2)
                    .then(6)
                    .when(pl.col("time_agg") % 10 == 3)
                    .then(9)
                    .otherwise(12),
                    1,
                )
            )

        result = result.collect(engine="streaming")

        time_col = "time_plot" if "time_plot" in result.columns else "time_agg"
        long_name = settings.get_var_info(var_name).get("long_name", var_name)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=result[time_col].to_numpy(),
                y=result[var_name].to_numpy(),
                mode="lines",
                name=var_name,
            )
        )
        fig.update_layout(
            title=long_name,
            xaxis_title="Time",
            yaxis_title="Value",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Spatial maps                                                        #
    # ------------------------------------------------------------------ #

    def spatial_maps(
        self,
        var_name: str,
        *,
        agg_by: Literal["month", "season"] = "month",
        dates: Optional[Union[list, tuple]] = None,
        data_bbox: Optional[tuple[float, float, float, float]] = None,
        map_bbox: Optional[tuple[float, float, float, float]] = None,
        vminmax: Optional[tuple[float, float]] = None,
        main_title: Optional[str] = None,
        legend_title: Optional[str] = None,
        save_path=None,
    ) -> None:
        """
        Climatological spatial maps — one panel per month (12) or season (4).

        Each panel shows the long-term mean of ``var_name`` at every grid cell,
        averaged across all years present in the selected data.

        Args:
            var_name: Variable name to plot.
            agg_by: 'month' for 12 panels, 'season' for 4 panels. Defaults to 'month'.
            dates: Date range or list for filtering. Defaults to full dataset.
            data_bbox: Spatial data filter (xmin, ymin, xmax, ymax). Subsets the parquet data
                before aggregation. Defaults to full dataset extent.
            map_bbox: Map display bounds (xmin, ymin, xmax, ymax). Controls the visible
                region on each panel. Defaults to the extent of the loaded data.
            vminmax: Fixed (vmin, vmax) for the colorbar. Defaults to data range.
            main_title: Figure title. Defaults to None.
            legend_title: Colorbar label. Defaults to the variable short name from config.
            save_path: Path to save the figure. If None, the plot is shown interactively.
        """
        if var_name not in self._idx.get_schema():
            raise ValueError(f"'{var_name}' not in parquet column names.")

        lon_col = self._idx.lon_col
        lat_col = self._idx.lat_col

        df = self._get_agg_df(var_name, agg_by, dates, data_bbox)

        plot_maps(
            df,
            var_name,
            agg_by=agg_by,
            lon_col=lon_col,
            lat_col=lat_col,
            vminmax=vminmax,
            data_bbox=data_bbox,
            map_bbox=map_bbox,
            main_title=main_title,
            legend_title=legend_title,
            save_path=save_path,
        )
