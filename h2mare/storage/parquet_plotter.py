"""Visualization layer for ParquetIndexer data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
import plotly.graph_objects as go
import polars as pl
from loguru import logger
from plotly.colors import DEFAULT_PLOTLY_COLORS

from h2mare import get_settings
from h2mare.storage.parquet_helpers import (
    aggregate_by_space_time,
    aggregate_by_time,
    aggregate_by_time_stats,
)
from h2mare.utils.plot import plot_maps

if TYPE_CHECKING:
    from h2mare.storage.parquet_catalog import ParquetCatalog
    from h2mare.storage.parquet_indexer import ParquetIndexer


class ParquetPlotter:
    """
    Visualization accessor for ParquetIndexer.

    Accessed via ``indexer.plot``. Do not instantiate directly.

    Example:
        >>> indexer.plot.time_series("sst", agg_by="month")
        >>> indexer.plot.monthly_map("sst")
    """

    # Render config for the interactive (Plotly) figures. ``displayModeBar``
    # defaults to "hover" in Plotly, so the toolbar only shows while the cursor
    # is over the plot and fades as you move toward it; pinning it True keeps it
    # always visible. ``displaylogo`` drops the Plotly logo. Applied via ``show``
    # (and available to callers for ``fig.show``/``write_html``).
    PLOT_CONFIG = {"displayModeBar": True, "displaylogo": False}

    def __init__(self, indexer: "ParquetCatalog | ParquetIndexer") -> None:
        self._idx = indexer
        self._cache: dict = {}
        self._grid_coords: pl.DataFrame | None = None

    def show(
        self, fig: go.Figure, renderer: Optional[str] = None, **config_overrides
    ) -> None:
        """Display an interactive figure with the toolbar always visible.

        Convenience wrapper around ``fig.show`` that applies :attr:`PLOT_CONFIG`
        (modebar pinned, no Plotly logo). The plot functions return a
        ``go.Figure`` so they stay composable (e.g. ``write_html``); use this to
        display one without the toolbar fading on hover::

            idx.plot.show(idx.plot.time_series(["sst", "chl"], agg_by="month"))

        Args:
            fig: The figure to display.
            renderer: Plotly renderer override (e.g. ``"browser"``,
                ``"vscode"``, ``"notebook"``). Defaults to the active Plotly
                default. Use ``"browser"`` if the inline toolbar buttons are
                unresponsive in your IDE/notebook.
            **config_overrides: Plotly config keys to override, e.g.
                ``displaylogo=True``.
        """
        fig.show(renderer=renderer, config={**self.PLOT_CONFIG, **config_overrides})

    def _snap_to_grid(
        self, point: tuple[float, float]
    ) -> tuple[float, float, float, float]:
        lon, lat = point
        lon_col = self._idx.lon_col
        lat_col = self._idx.lat_col
        if self._grid_coords is None:
            # Every partition shares the same spatial grid — one file is enough
            first_file = self._idx._resolve_files(None)[0]
            self._grid_coords = (
                pl.scan_parquet(first_file)
                .select([lon_col, lat_col])
                .unique()
                .collect()
            )
        lons = self._grid_coords[lon_col].unique()
        lats = self._grid_coords[lat_col].unique()
        nearest_lon = float(lons[(lons - lon).abs().arg_min()])
        nearest_lat = float(lats[(lats - lat).abs().arg_min()])
        logger.debug(
            f"Point ({lon}, {lat}) snapped to grid cell ({nearest_lon}, {nearest_lat})"
        )
        return (nearest_lon, nearest_lat, nearest_lon, nearest_lat)

    @staticmethod
    def _add_season_plot_date(result: pl.LazyFrame) -> pl.LazyFrame:
        """Add a plottable ``time_plot`` date from the ``YYYYS`` season encoding.

        Seasons are anchored to the first month of each meteorological season
        (spring → March, summer → June, autumn → September, winter → December).
        """
        return result.with_columns(
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
            ).collect(engine="streaming")
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear the aggregation cache (e.g. after new data is added)."""
        self._cache.clear()

    # ------------------------------------------------------------------ #
    #  Time series                                                         #
    # ------------------------------------------------------------------ #

    def time_series(
        self,
        var_name: Union[str, list[str]],
        agg_by: Literal["day", "week", "month", "season", "year"],
        *,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float] | tuple[float, float, float, float]] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive time series line plot aggregated over space and time.

        Compares one or more variables as mean lines over time. With multiple
        variables each gets its own y-axis (left/right, then floated outward),
        so fields with different units/scales remain comparable. For a single
        variable's full distribution (±1σ, min/max, trends) use
        :meth:`stats_summary` instead.

        Args:
            var_name: Variable name, or a list of up to 4 variable names. Each
                variable is drawn as its own line on its own y-axis.
            agg_by: Temporal aggregation granularity.
            dates: Temporal filter. Pass a ``(start, end)`` tuple for a contiguous
                range (e.g. ``("2010-01-01", "2020-12-31")``) or a ``list`` of
                discrete dates (e.g. ``["2010-06-01", "2015-06-01"]``). A 2-element
                tuple is always treated as a range, not two discrete dates.
                Defaults to the full dataset.
            bbox: Spatial filter. Either a 4-tuple (xmin, ymin, xmax, ymax) for an extent
                or a 2-tuple (lon, lat) to select the nearest grid cell. Defaults to full extent.
            title: Figure title. Defaults to the variable's long name for a
                single variable, or ``"Time series"`` for multiple.

        Note:
            Seasonal values are assigned to the first month of the respective season
            (e.g. spring → March 1st).

        Raises:
            ValueError: If no variables are given, more than 4 are given, or any
                variable is absent from the store.

        Returns:
            go.Figure. Display it with :meth:`show` to keep the toolbar pinned
            (Plotly otherwise only reveals it on hover).
        """
        vars_list = [var_name] if isinstance(var_name, str) else list(var_name)

        if not vars_list:
            raise ValueError("var_name must contain at least one variable.")
        if len(vars_list) > 4:
            raise ValueError(
                f"time_series supports at most 4 variables (got {len(vars_list)}); "
                "plot fewer variables or split into multiple figures."
            )
        missing = [v for v in vars_list if v not in self._idx.get_schema()]
        if missing:
            raise ValueError(f"{missing} not in parquet column names.")

        if bbox is not None and len(bbox) == 2:
            bbox = self._snap_to_grid(bbox)  # type: ignore[arg-type]

        lf = self._idx.scan(
            dates=dates, bbox=bbox, columns=[self._idx.time_col, *vars_list]
        )
        result = aggregate_by_time(lf, vars_list, agg_by=agg_by)

        # Season encoding is YYYYS — convert to plottable dates
        if agg_by == "season":
            result = self._add_season_plot_date(result)

        result = result.collect(engine="streaming")

        time_col = "time_plot" if "time_plot" in result.columns else "time_agg"
        x = result[time_col].to_numpy()

        # Shrink the plot area to make room for any floated (3rd/4th) axes.
        x_left = 0.08 if len(vars_list) >= 3 else 0.0
        x_right = 1.0 - (0.08 if len(vars_list) >= 4 else 0.0)

        fig = go.Figure()
        layout_axes: dict = {}
        for i, var in enumerate(vars_list):
            color = DEFAULT_PLOTLY_COLORS[i % len(DEFAULT_PLOTLY_COLORS)]
            info = get_settings().get_var_info(var)
            axis_title = info.get("units") or info.get("long_name", var)

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=result[var].to_numpy(),
                    mode="lines",
                    name=info.get("short_name", var),
                    line=dict(color=color),
                    yaxis="y" if i == 0 else f"y{i + 1}",
                )
            )

            axis_cfg: dict = {
                "title": dict(text=axis_title, font=dict(color=color)),
                "tickfont": dict(color=color),
            }
            if i == 1:
                axis_cfg.update(overlaying="y", side="right")
            elif i == 2:
                axis_cfg.update(
                    overlaying="y", side="left", anchor="free", position=0.0
                )
            elif i == 3:
                axis_cfg.update(
                    overlaying="y", side="right", anchor="free", position=1.0
                )
            layout_axes["yaxis" if i == 0 else f"yaxis{i + 1}"] = axis_cfg

        if title is None:
            title = (
                get_settings().get_var_info(vars_list[0]).get("long_name", vars_list[0])
                if len(vars_list) == 1
                else "Time series"
            )
        fig.update_layout(
            title=title,
            xaxis=dict(
                title="Time",
                domain=[x_left, x_right],
                rangeslider=dict(visible=True),
                type="date",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            **layout_axes,
        )
        return fig

    # ------------------------------------------------------------------ #
    #  Statistics summary                                                  #
    # ------------------------------------------------------------------ #

    def stats_summary(
        self,
        var_name: str,
        agg_by: Literal["day", "week", "month", "season", "year"],
        *,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        lowess_frac: float = 0.3,
        title: Optional[str] = None,
    ) -> go.Figure:
        """
        Interactive composite plot of mean, ±1 std, min, and max over time.

        Each statistic is computed per time bucket after spatially aggregating all
        grid cells. LOWESS trend lines are overlaid for mean, min, and max.

        Args:
            var_name: Variable name to plot.
            agg_by: Temporal aggregation granularity (day, week, month, season, year).
            dates: Temporal filter. Pass a ``(start, end)`` tuple for a contiguous
                range or a ``list`` of discrete dates. A 2-element tuple is always
                treated as a range. Defaults to the full dataset.
            bbox: Spatial filter as a 4-tuple ``(xmin, ymin, xmax, ymax)``.
                Defaults to full extent.
            lowess_frac: Fraction of data used for each local LOWESS fit (0 < frac ≤ 1).
                Lower values follow the data more closely; higher values produce a
                smoother curve. Defaults to 0.3.
            title: Figure title. Defaults to ``"{long_name} — Statistics Summary"``.

        Note:
            Seasonal values are assigned to the first month of the respective season
            (e.g. spring → March 1st). ``std`` is ``null`` for buckets with a single
            observation; Plotly renders those as gaps in the shaded band.

        Returns:
            go.Figure with mean (solid), ±1 std band (shaded), min/max (dashed),
            and LOWESS trend lines for mean, min, and max (dotted). Display it
            with :meth:`show` to keep the toolbar pinned.
        """
        if var_name not in self._idx.get_schema():
            raise ValueError(f"'{var_name}' not in parquet column names.")

        lf = self._idx.scan(
            dates=dates, bbox=bbox, columns=[self._idx.time_col, var_name]
        )
        result = aggregate_by_time_stats(lf, var_name, agg_by=agg_by)

        if agg_by == "season":
            result = self._add_season_plot_date(result)

        result = result.collect(engine="streaming")

        time_col = "time_plot" if "time_plot" in result.columns else "time_agg"
        long_name = get_settings().get_var_info(var_name).get("long_name", var_name)

        x = result[time_col].to_numpy()
        mean_arr = result[f"{var_name}_mean"].to_numpy()
        std_arr = result[f"{var_name}_std"].to_numpy()

        x_numeric = x.astype("float64")
        min_arr = result[f"{var_name}_min"].to_numpy()
        max_arr = result[f"{var_name}_max"].to_numpy()

        from statsmodels.nonparametric.smoothers_lowess import lowess

        def _trend(y: np.ndarray) -> np.ndarray:
            mask = ~np.isnan(y)
            return lowess(
                y[mask], x_numeric[mask], frac=lowess_frac, return_sorted=False
            )

        mean_trend = _trend(mean_arr)
        min_trend = _trend(min_arr)
        max_trend = _trend(max_arr)

        fig = go.Figure()
        # Shaded ±1 std band — two invisible boundary traces with fill between them
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_arr - std_arr,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                legendgroup="std",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_arr + std_arr,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(99, 110, 250, 0.2)",
                name="±1 std",
                legendgroup="std",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_arr,
                mode="lines",
                name="mean",
                line=dict(color="rgb(99, 110, 250)", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_trend,
                mode="lines",
                name="mean_trend",
                line=dict(dash="dot", color="rgb(99, 110, 250)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=min_arr,
                mode="lines",
                name="min",
                line=dict(dash="dash", color="rgb(239, 85, 59)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=min_trend,
                mode="lines",
                name="min_trend",
                line=dict(dash="dot", color="rgb(239, 85, 59)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=max_arr,
                mode="lines",
                name="max",
                line=dict(dash="dash", color="rgb(0, 204, 150)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=max_trend,
                mode="lines",
                name="max_trend",
                line=dict(dash="dot", color="rgb(0, 204, 150)", width=1),
            )
        )
        fig.update_layout(
            title=title or f"{long_name} — Statistics Summary",
            xaxis_title="Time",
            yaxis_title="Value",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            plot_bgcolor="white",
            paper_bgcolor="white",
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
        grid_shape: Optional[tuple[int, int]] = None,
        vminmax: Optional[tuple[float, float]] = None,
        cmap: str = "turbo",
        title: Optional[str] = None,
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
            dates: Temporal filter. Pass a ``(start, end)`` tuple for a contiguous
                range (e.g. ``("2010-01-01", "2020-12-31")``) or a ``list`` of
                discrete dates (e.g. ``["2010-06-01", "2015-06-01"]``). A 2-element
                tuple is always treated as a range, not two discrete dates.
                Defaults to the full dataset.
            data_bbox: Spatial data filter (xmin, ymin, xmax, ymax). Subsets the parquet data
                before aggregation. Defaults to full dataset extent.
            map_bbox: Map display bounds (xmin, ymin, xmax, ymax). Controls the visible
                region on each panel. Defaults to the extent of the loaded data.
            grid_shape: Grid layout as ``(nrows, ncols)``. Defaults to ``(6, 2)`` for
                monthly and ``(2, 2)`` for seasonal maps. Figsize is derived automatically
                from the map extent so there are no blank spaces between rows.
            vminmax: Fixed (vmin, vmax) for the colorbar. Defaults to data range.
            cmap: Matplotlib colormap name. Defaults to 'turbo'.
            title: Figure title. Defaults to None.
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
            cmap=cmap,
            data_bbox=data_bbox,
            map_bbox=map_bbox,
            grid_shape=grid_shape,
            main_title=title,
            legend_title=legend_title,
            save_path=save_path,
        )
