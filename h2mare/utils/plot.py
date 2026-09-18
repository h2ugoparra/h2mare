"""
plot functions
"""

import calendar
import math
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.storage.var_routing import catalog_for_var
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, ReadFrom
from h2mare.validators import validate_columns, validate_var_key

_PANEL_WIDTH = 3.0  # inches per panel column
_WSPACE = -0.15  # fractional horizontal gap between panels
_HSPACE = 0.20  # fractional vertical gap between panels (must fit panel titles)


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

    validate_columns(df, var_name)

    # Derive group column from time_col when not already present
    if agg_by not in df.columns:
        if time_col is None:
            raise ValueError(
                f"Column '{agg_by}' not found in df. "
                f"Pass time_col so it can be derived automatically."
            )
        validate_columns(df, time_col)
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

    legend_label = legend_title or get_settings().variable_attrs.get(var_name, {}).get(
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
    validate_columns(df, [var_name, lon_col, lat_col])
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
    validate_columns(df, group_col)

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

    Note:
        Renders through ``IPython.display``, so it only animates inside a
        notebook. The import is function-local: this is the only thing in the
        package that needs IPython, and at module scope it made a heavyweight
        notebook-only dependency a hard requirement of importing any plotting
        helper.
    """
    from IPython.display import clear_output, display

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


# --------------------------------
#       INTERACTIVE
# --------------------------------


def _to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to WGS84 (EPSG:4326) if it carries a different CRS.

    Plotly and the zarr grid both expect lon/lat in WGS84; a GeoDataFrame in another CRS
    would otherwise plot in the wrong place. A GeoDataFrame with no CRS is left untouched.
    """
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        return gdf.to_crs(4326)
    return gdf


def _points_to_latlon(gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, str, str]:
    """Reproject a point GeoDataFrame to WGS84 and extract lon/lat columns.

    MultiPoint geometries are exploded into one Point per row. Non-point geometries
    are rejected because they need a different Plotly trace type (choropleth/line map).

    Args:
        gdf: Source GeoDataFrame with point (or multipoint) geometry.

    Returns:
        A tuple ``(df, lat_col, lon_col)`` where *df* is a plain DataFrame carrying the
        original attribute columns plus synthesized ``"_lat"`` / ``"_lon"`` columns.

    Raises:
        ValueError: if the GeoDataFrame contains non-point geometry.
    """
    geom_types = set(gdf.geom_type.dropna().unique())
    allowed = {"Point", "MultiPoint"}
    if not geom_types <= allowed:
        raise ValueError(
            f"plot_interactive_map supports point geometry only; got {sorted(geom_types)}. "
            "Polygons/lines need a choropleth or line map instead."
        )

    gdf = _to_wgs84(gdf)

    if "MultiPoint" in geom_types:
        gdf = gdf.explode(index_parts=False)

    geom = gdf.geometry
    df = pd.DataFrame(gdf.drop(columns=geom.name))
    df["_lon"] = geom.x.to_numpy()
    df["_lat"] = geom.y.to_numpy()
    return df, "_lat", "_lon"


def plot_interactive_map(
    df: pd.DataFrame | gpd.GeoDataFrame,
    lat: str | None = None,
    lon: str | None = None,
    hover_cols: str | list[str] | None = None,
    export_path: str | Path | None = None,
    map_style: str = "open-street-map",
    zoom: int = 4,
) -> None:
    """Show an interactive scatter map from a DataFrame or point GeoDataFrame.

    Args:
        df: Source data. A plain DataFrame with explicit lat/lon columns, or a
            GeoDataFrame with point (or multipoint) geometry. For a GeoDataFrame the
            geometry is reprojected to WGS84 and lat/lon are derived from it, so the
            *lat* / *lon* arguments are ignored.
        lat: Name of the latitude column. Required for a plain DataFrame; ignored for a
            GeoDataFrame.
        lon: Name of the longitude column. Required for a plain DataFrame; ignored for a
            GeoDataFrame.
        hover_cols: Column name(s) to display on hover. Pass None to show no extra data.
        export_path: If provided, saves the map as an HTML file at this path.
        map_style: Plotly basemap style. Defaults to "open-street-map". If exported to HTML,
            "carto-positron" or "carto-darkmatter" are recommended because OpenStreetMap's
            tile servers reject requests with no Referer header (e.g. an exported HTML
            opened from a file:// path), which shows up as "403 Access blocked" tiles.
            The mentioned Carto basemaps need no token and no Referer, so they render in
            exported files.
        zoom: Initial map zoom level. Defaults to 4.

    Raises:
        ValueError: if *df* is empty, if lat/lon are missing for a plain DataFrame, or
            if a GeoDataFrame carries non-point geometry.
    """
    if isinstance(hover_cols, str):
        hover_cols = [hover_cols]

    # Normalize point GeoDataFrame input to a plain DataFrame with lat/lon columns.
    if isinstance(df, gpd.GeoDataFrame):
        df, lat, lon = _points_to_latlon(df)
    elif lat is None or lon is None:
        raise ValueError(
            "lat and lon column names are required for non-GeoDataFrame input."
        )

    if len(df) == 0:
        raise ValueError("No data to plot.")

    # Suppress lat/lon from the tooltip; show requested hover columns on top.
    hover_data: dict[str, bool] = {lat: False, lon: False}
    if hover_cols:
        hover_data.update({col: True for col in hover_cols})

    center = {"lat": df[lat].mean(), "lon": df[lon].mean()}

    fig = px.scatter_map(
        df,
        lat=lat,
        lon=lon,
        hover_data=hover_data,
        center=center,
        zoom=zoom,
        map_style=map_style,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    if export_path is not None:
        fig.write_html(str(export_path))
        logger.info(f"Map saved to {export_path}")

    fig.show()


def _default_var_for(var_key: str) -> str:
    """The variable to plot when the caller names none.

    ``compiled_vars[0]`` by config convention: the first name the var_key
    publishes, and one :func:`catalog_for_var` can route. A var_key publishing
    nothing (the compiled store itself) has no such list, so its own store
    answers instead.
    """
    app_config = get_settings().app_config
    var_key = validate_var_key(var_key, app_config)
    declared = list(getattr(app_config.variables[var_key], "compiled_vars", None) or [])
    if declared:
        return declared[0]

    stored = sorted(ZarrCatalog(var_key).get_variables())
    if not stored:
        raise ValueError(
            f"'{var_key}' declares no compiled_vars and its store holds nothing, "
            f"so there is no variable to default to — pass var explicitly."
        )
    return stored[0]


def field_for_plot(ds: xr.Dataset, var: str) -> tuple[xr.DataArray, str]:
    """One 2-D field for *var*, plus a note on how its time axis was collapsed.

    ``open_dataset(dates=...)`` selects by calendar day, so an hourly store
    returns all 24 steps for a date and the array reaching ``.plot()`` is 3-D.
    They are averaged into the day's field here rather than left to raise, and
    the note says so — silently plotting one arbitrary hour, or averaging
    without saying it, is how a diagnostic starts lying about what it shows.

    Args:
        ds: Dataset for a single date.
        var: Data variable to draw.

    Returns:
        ``(field, note)``. *note* is empty when nothing was collapsed.

    Raises:
        KeyError: if *var* is absent from *ds* — reachable even after routing,
            since a store's early files can predate a variable added later.
    """
    if var not in ds.data_vars:
        raise KeyError(
            f"'{var}' is not in this file: it holds {sorted(map(str, ds.data_vars))}. "
            f"The store as a whole lists it, so it was likely added to the store "
            f"after the file covering this date was written."
        )

    da = ds[var]
    if "time" in da.dims and da.sizes["time"] > 1:
        n = da.sizes["time"]
        return da.mean("time", keep_attrs=True), f"{var}: mean of {n} steps"
    if "time" in da.dims:
        da = da.isel(time=0)
    return da, ""


def plot_records_on_field(
    data: pd.DataFrame | gpd.GeoDataFrame,
    var_key: str,
    var: str | None = None,
    time_col: str = "date",
    lon_col: str = "lon",
    lat_col: str = "lat",
    offset: float = 1.0,
    max_plots: int = 12,
    title_fn: Callable[[pd.Series], str] | None = None,
    read_from: ReadFrom = "auto",
) -> None:
    """Plot the variable field around each record, with the location overlaid.

    For each of the first ``max_plots`` records, opens the gridded ``var`` field in a bbox
    around the record's location and overlays that location in red. Records with no data
    for their date/bbox are skipped with a warning.

    Works off either spatial source, no index required:
    - a GeoDataFrame: reprojected to WGS84; the bbox is each row's geometry bounds
      (+/- ``offset``) and the geometry is overlaid in red.
    - a plain DataFrame: the bbox is the (``lon_col``, ``lat_col``) point (+/- ``offset``)
      and the point is overlaid in red.

    Args:
        data: Records to plot. If a GeoDataFrame, its active geometry is used (and
            reprojected to WGS84); otherwise ``lon_col``/``lat_col`` are used.
        var_key: Variable key the variable belongs to.
        var: Data variable name. Defaults to the first name ``var_key`` publishes.
            Need not live in ``var_key``'s own store — see ``read_from``.
        time_col: Name of the date column.
        lon_col, lat_col: Coordinate columns, used only when ``data`` has no geometry.
        offset: Half-width (in degrees) of the bbox drawn around each location.
        max_plots: Maximum number of records to plot.
        title_fn: Optional callable mapping a row to a plot title. Defaults to the date.
        read_from: Which store to read ``var`` from. ``"auto"`` routes to whichever
            one holds the name (see :func:`~h2mare.storage.var_routing.catalog_for_var`),
            so a daily h2ds column of an hourly var_key plots the daily field that
            extraction returned. ``"native"`` pins the var_key's own store, which
            is how you look at the raw hourly field behind such a column.
    """
    if var is None:
        var = _default_var_for(var_key)
        logger.info(f"var not set; plotting '{var}' for '{var_key}'.")

    cat = catalog_for_var(var, var_key, read_from=read_from)
    is_geo = isinstance(data, gpd.GeoDataFrame)
    if is_geo:
        data = _to_wgs84(data)

    for _, row in data.head(max_plots).iterrows():
        date = row[time_col]

        if is_geo:
            minx, miny, maxx, maxy = row.geometry.bounds
        else:
            minx = maxx = row[lon_col]
            miny = maxy = row[lat_col]
        bbox = (minx - offset, miny - offset, maxx + offset, maxy + offset)

        try:
            ds = cat.open_dataset(dates=date, bbox=bbox, variables=var)  # type: ignore[arg-type]
        except FileNotFoundError:
            # open_dataset raises for a date the store has no file for; it never
            # returns None, so catching is the only way to skip an uncovered
            # record rather than end the loop on it. Not re-raised with the
            # original message: it lists every date the file does hold, which is
            # a year of them.
            logger.warning(
                f"[{cat.var_key}] no data for {pd.to_datetime(date).date()}; "  # type: ignore[arg-type]
                f"skipping this record."
            )
            continue

        field, note = field_for_plot(ds, var)

        fig, ax = plt.subplots()
        field.plot(ax=ax)  # type: ignore
        if is_geo:
            gpd.GeoSeries([row.geometry], crs=data.crs).plot(
                ax=ax, color="red", markersize=10, zorder=5
            )
        else:
            ax.scatter(row[lon_col], row[lat_col], color="red", s=10, zorder=5)

        title = title_fn(row) if title_fn else str(pd.to_datetime(date).date())  # type: ignore
        ax.set_title(f"{title}\n{note}" if note else title)
        plt.show()
        plt.close(fig)
