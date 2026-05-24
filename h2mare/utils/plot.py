"""
plot functions
"""

import calendar
import math
from pathlib import Path
from typing import Literal, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr
from IPython.display import clear_output, display

from h2mare.config import settings
from h2mare.storage.parquet_helpers import _required_columns
from h2mare.types import BBox


_PANEL_WIDTH = 3.0  # inches per panel column
_WSPACE = -0.15     # fractional horizontal gap between panels
_HSPACE = 0.20      # fractional vertical gap between panels (must fit panel titles)

# --------------------------------
#       PARQUET
# --------------------------------
def plot_maps(
    df: pl.DataFrame,
    var_name: str,
    *,
    agg_by: Literal["month", "season"],
    time_col: str = "time",
    lon_col: str = "lon",
    lat_col: str = "lat",
    vminmax: Optional[tuple[int | float, int | float]] = None,
    data_bbox: Optional[tuple[float, float, float, float]] = None,
    map_bbox: Optional[tuple[float, float, float, float]] = None,
    grid_shape: Optional[tuple[int, int]] = None,
    cmap: str = "turbo",
    main_title: Optional[str] = None,
    legend_title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plots monthly or seasonal maps.

    Args:
        df (pl.DataFrame): Data input. Must contain var_name, lon_col, lat_col, and
            either a pre-computed group column (``agg_by`` value) or ``time_col`` so the
            group column can be derived automatically.
        var_name (str): Variable name.
        agg_by (Literal['month', 'season']): Time aggregation.
        time_col (str): Name of the datetime column used to derive the group
            column when it is not already present in *df*. Defaults to None.
        vminmax (tuple[int | float, int | float], optional): Variable min and max.
            Defaults to None, inferring from data.
        data_bbox (tuple[float, float, float, float], optional): Data geographic extent
            (xmin, ymin, xmax, ymax). Used to derive the map extent when *map_bbox* is
            not provided. Defaults to None, inferring from data.
        map_bbox (tuple[float, float, float, float], optional): Map display extent
            (xmin, ymin, xmax, ymax). Controls the visible region on each panel.
            Defaults to None, falling back to *data_bbox* or the inferred data extent.
        grid_shape (tuple[int, int], optional): ``(nrows, ncols)`` grid layout passed to
            ``make_axes``. Defaults to None (auto).
        main_title (str, optional): Plot main title. Defaults to None.
        legend_title (str, optional): Legend title. Defaults to None; falls back to
            ``short_name`` in config.yaml, then to ``var_name``.
        save_path (Path, optional): Path to save plot. Defaults to None (show plot).

    Raises:
        ValueError: if df is empty, var_name is missing, or the group column cannot
            be derived.
    """
    if df.is_empty():
        raise ValueError("No data after aggregation.")

    _required_columns(df, var_name)

    # Derive group column from time_col when not already present
    if agg_by not in df.columns:
        if time_col is None:
            raise ValueError(
                f"Column '{agg_by}' not found in df. "
                f"Pass time_col so it can be derived automatically."
            )
        _required_columns(df, time_col)
        if agg_by == "month":
            df = df.with_columns(pl.col(time_col).dt.month().alias("month"))
        elif agg_by == "season":
            df = df.with_columns(
                pl.when(pl.col(time_col).dt.month().is_in([12, 1, 2]))
                .then(pl.lit("winter"))
                .when(pl.col(time_col).dt.month().is_in([3, 4, 5]))
                .then(pl.lit("spring"))
                .when(pl.col(time_col).dt.month().is_in([6, 7, 8]))
                .then(pl.lit("summer"))
                .otherwise(pl.lit("autumn"))
                .alias("season")
            )

    if data_bbox is not None:
        xmin, ymin, xmax, ymax = data_bbox
    else:
        metadata = BBox.from_dataframe(df, lon_col=lon_col, lat_col=lat_col)
        xmin = float(metadata.xmin)
        xmax = float(metadata.xmax)
        ymin = float(metadata.ymin)
        ymax = float(metadata.ymax)

    if map_bbox is not None:
        map_xmin, map_ymin, map_xmax, map_ymax = map_bbox
    else:
        map_xmin, map_ymin, map_xmax, map_ymax = xmin, ymin, xmax, ymax

    if vminmax is not None:
        vmin, vmax = vminmax
    else:
        metadata = df.select(
            [pl.col(var_name).min().alias("vmin"), pl.col(var_name).max().alias("vmax")]
        )

        vmin = float(metadata["vmin"][0])
        vmax = float(metadata["vmax"][0])

    groups = split_by_group(df, agg_by)

    fig, axes = make_axes(
        len(groups),
        grid_shape=grid_shape,
        map_extent=(map_xmin, map_xmax, map_ymin, map_ymax),
    )

    if main_title:
        fig.suptitle(main_title, fontsize=12, fontweight="bold")

    meshes = []

    for ax, (group, subdf) in zip(axes, groups.items()):
        lon, lat, grid = df_to_grid(subdf, var_name, lon_col=lon_col, lat_col=lat_col)

        title = calendar.month_abbr[group] if isinstance(group, int) else str(group)

        mesh = plot_panel(
            ax,
            lon,
            lat,
            grid,
            title=title,
            extent=(map_xmin, map_xmax, map_ymin, map_ymax),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        meshes.append(mesh)

    for ax in axes[len(groups) :]:
        fig.delaxes(ax)

    cbar = fig.colorbar(
        meshes[-1],
        ax=axes,
        location="bottom",
        pad=0.03,  # space between subplots and colorbar
        shrink=0.6,
    )

    legend_label = legend_title or settings.variable_attrs.get(var_name, {}).get(
        "short_name", var_name
    )
    cbar.set_label(legend_label)

    if save_path:
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_panel(
    ax,
    lon,
    lat,
    grid,
    *,
    title: str,
    extent: tuple[float, float, float, float],
    vmin: float,
    vmax: float,
    cmap: str = "turbo",
):
    ax.set_extent(extent)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")

    mesh = ax.pcolormesh(
        lon,
        lat,
        grid,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(title, fontsize=8)
    return mesh


def make_axes(
    n_panels: int,
    grid_shape: Optional[tuple[int, int]] = None,
    map_extent: Optional[tuple[float, float, float, float]] = None,
):
    """
    Define subplots layout according to n_panels.

    Default layouts: (nrows=6, ncols=2) for 12 panels, (nrows=2, ncols=2) for 4 panels.
    When map_extent is provided the figsize is derived from the panel aspect ratio so
    there are no blank spaces between rows.

    Args:
        n_panels (int): Number of panels.
        grid_shape (tuple[int, int], optional): ``(nrows, ncols)`` override. Defaults to None (auto).
        map_extent (tuple[float, float, float, float], optional): ``(xmin, xmax, ymin, ymax)``
            used to compute the panel aspect ratio. Defaults to None (formula fallback).
    """
    if grid_shape is not None:
        nrows, ncols = grid_shape
    elif n_panels == 12:
        nrows, ncols = 6, 2
    elif n_panels == 4:
        nrows, ncols = 2, 2
    else:
        ncols = math.ceil(math.sqrt(n_panels))
        nrows = math.ceil(n_panels / ncols)

    if map_extent is not None:
        xmin, xmax, ymin, ymax = map_extent
        panel_aspect = (xmax - xmin) / (ymax - ymin)
        panel_h = _PANEL_WIDTH / panel_aspect
        # Inflate figsize so that after wspace/hspace are removed each cell is
        # exactly _PANEL_WIDTH × panel_h, keeping GeoAxes aspect ratio filled.
        figsize = (
            _PANEL_WIDTH * (ncols + _WSPACE * (ncols - 1)),
            panel_h * (nrows + _HSPACE * (nrows - 1)),
        )
    else:
        figsize = (ncols * 3.5, nrows * 2.5)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"wspace": _WSPACE, "hspace": _HSPACE},
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    return fig, axes.flatten()


def df_to_grid(
    df: pl.DataFrame, var_name: str, *, lon_col: str = "lon", lat_col: str = "lat"
):
    _required_columns(df, [var_name, lon_col, lat_col])
    lon = df[lon_col].to_numpy()
    lat = df[lat_col].to_numpy()
    val = df[var_name].to_numpy()

    unique_lon = np.sort(np.unique(lon))
    unique_lat = np.sort(np.unique(lat))

    lon_idx = np.searchsorted(unique_lon, lon)
    lat_idx = np.searchsorted(unique_lat, lat)

    grid = np.full((unique_lat.size, unique_lon.size), np.nan)
    grid[lat_idx, lon_idx] = val

    return unique_lon, unique_lat, grid


def split_by_group(
    df: pl.DataFrame,
    group_col: str,
) -> dict[int | str, pl.DataFrame]:
    _required_columns(df, group_col)

    if group_col == "month":
        df = df.sort("month")
    elif group_col == "season":
        season_order = ["spring", "summer", "autumn", "winter"]
        df = df.with_columns(
            pl.col("season").cast(pl.Enum(season_order)).alias("_season_ord")
        ).sort("_season_ord")

    return {g[0]: subdf for g, subdf in df.group_by(group_col, maintain_order=True)}


# --------------------------------
#       XARRAY
# --------------------------------


def animate_vars(
    data: xr.Dataset | xr.DataArray,
    var_name: str | None = None,
    nsteps: int = 30,
    dim: Literal["time", "depth"] = "time",
    time_idx: int = 0,
    depth_idx: int = 0,
) -> None:
    """
    Animate plots over a selected dimension (time or depth).

    When dim='time' (default): animates over time steps; if a depth dimension is
    also present, depth_idx fixes the depth level shown in every frame.
    When dim='depth': animates over depth levels; time_idx fixes the time step.

    Args:
        data: Input data.
        var_name: Name of variable to plot (only needed if input is a Dataset).
        nsteps: Maximum number of frames along the animated dimension.
        dim: Dimension to animate over. Defaults to 'time'.
        time_idx: Time index used as the fixed time step when dim='depth'. Defaults to 0.
        depth_idx: Depth index used as the fixed depth level when dim='time'. Defaults to 0.
    """
    if isinstance(data, xr.Dataset):
        if var_name is None:
            raise ValueError("var_name must be provided when input is a Dataset")
        da = data[var_name]
    else:
        da = data

    if dim not in da.dims:
        raise ValueError(f"Input does not have a '{dim}' dimension")

    if dim == "time":
        if "depth" in da.dims:
            da = da.isel(depth=depth_idx)
            depth_label = (
                f" | depth: {float(da['depth'].values):.0f} m"
                if "depth" in da.coords
                else ""
            )
        else:
            depth_label = ""
        nframes = min(nsteps, da.sizes["time"])
        for i in range(nframes):
            fig, ax = plt.subplots()
            da.isel(time=i).plot(ax=ax)  # type: ignore
            plt.title(f"Time step: {i} | {str(da['time'].values[i])}{depth_label}")
            display(fig)
            clear_output(wait=True)
            plt.close()

    else:  # dim == "depth"
        if "time" in da.dims:
            da = da.isel(time=time_idx)
            time_label = (
                f" | time: {str(da['time'].values)}" if "time" in da.coords else ""
            )
        else:
            time_label = ""
        nframes = min(nsteps, da.sizes["depth"])
        for i in range(nframes):
            frame = da.isel(depth=i)
            depth_val = (
                f"{float(frame['depth'].values):.0f} m"
                if "depth" in frame.coords
                else str(i)
            )
            fig, ax = plt.subplots()
            frame.plot(ax=ax)  # type: ignore
            plt.title(f"Depth: {depth_val}{time_label}")
            display(fig)
            clear_output(wait=True)
            plt.close()


def plot_snapshot(
    ds: xr.Dataset,
    time_idx: int = 0,
    depth_idx: int | None = None,
) -> None:
    """
    Plot all variables at a given time index, optionally selecting a depth level.

    For variables that have a depth dimension, depth_idx (or 0 if not provided)
    is used to select a single level before plotting.

    Args:
        ds: Input dataset.
        time_idx: Integer index along the time dimension. Defaults to 0.
        depth_idx: Integer index along the depth dimension. Defaults to 0 for
            variables that have a depth dimension when not provided.
    """
    vars_with_time = [v for v in ds.data_vars if "time" in ds[v].dims]
    for var in vars_with_time:
        da = ds[var].isel(time=time_idx)
        if "depth" in da.dims:
            da = da.isel(depth=depth_idx if depth_idx is not None else 0)
        da.plot()  # type: ignore
        plt.show()
