# Plotting utilities

Standalone plotting helpers in `h2mare.utils.plot`:

```python
from h2mare.utils.plot import (
    plot_maps,             # climatological panel maps from a Parquet DataFrame
    plot_snapshot,         # all variables of a Dataset at one time step
    animate_vars,          # animate a variable over time or depth
    plot_interactive_map,  # interactive Plotly scatter from points
    plot_records_on_field, # the variable field around each record, location overlaid
)
```

Import from the module path, not from `h2mare.utils`. The package deliberately
does not re-export these: `plot_records_on_field` needs `ZarrCatalog`, and
`storage.zarr_catalog` imports `utils.labels`, so re-exporting plot would make
importing anything from `h2mare.utils` circular — and would pull
matplotlib/cartopy/geopandas into every `h2mare.utils.*` import.

These are free functions for ad-hoc and notebook use. For the Parquet store's
bundled accessor (`indexer.plot.time_series(...)`, etc.) see
[`ParquetPlotter`](parquet_plotter.md) — its `spatial_maps()` wraps `plot_maps()`.

---

## `plot_maps()`

```python
plot_maps(
    df,                 # pl.DataFrame
    var_name,
    *,
    agg_by,             # "month" | "season"
    time_col="time",
    lon_col="lon",
    lat_col="lat",
    vminmax=None,
    data_bbox=None,
    map_bbox=None,
    grid_shape=None,
    cmap="turbo",
    main_title=None,
    legend_title=None,
    save_path=None,
)
```

Climatological panel maps from a Polars DataFrame — 12 panels for
`agg_by="month"`, 4 for `agg_by="season"`. Each panel is the mean field over all
rows in that group. The group column is derived from `time_col` when not already
present.

| Parameter | Description |
|---|---|
| `df` | Polars DataFrame with `var_name`, `lon_col`, `lat_col`, and either the `agg_by` group column or `time_col` |
| `var_name` | Variable column to plot |
| `agg_by` | `"month"` (12 panels) or `"season"` (4 panels) |
| `time_col` | Datetime column used to derive the group column when absent. Defaults to `"time"` |
| `lon_col` / `lat_col` | Coordinate columns. Default `"lon"` / `"lat"` |
| `vminmax` | Fixed `(vmin, vmax)` for the colorbar. Defaults to the data range |
| `data_bbox` | `(xmin, ymin, xmax, ymax)` data extent. Defaults to inferred from data |
| `map_bbox` | Visible region per panel. Defaults to `data_bbox` or the inferred extent |
| `grid_shape` | `(nrows, ncols)` layout override. Defaults to auto |
| `cmap` | Matplotlib colormap. Defaults to `"turbo"` |
| `main_title` | Figure title |
| `legend_title` | Colorbar label. Defaults to the variable `short_name` from config, then `var_name` |
| `save_path` | Path to save the figure. If `None`, shown interactively |

Raises `ValueError` if `df` is empty, `var_name` is missing, or the group column
cannot be derived.

---

## `plot_snapshot()`

```python
plot_snapshot(
    ds,                 # xr.Dataset
    time_idx=0,
    depth_idx=None,
)
```

Plot every variable of a Dataset at a single time index, one figure per variable.
Variables with a `depth` dimension are sliced at `depth_idx` (or level 0 when not
given) before plotting.

| Parameter | Description |
|---|---|
| `ds` | Source Dataset |
| `time_idx` | Integer index along the `time` dimension. Defaults to `0` |
| `depth_idx` | Integer index along the `depth` dimension for variables that have one. Defaults to level `0` |

---

## `animate_vars()`

```python
animate_vars(
    data,               # xr.Dataset | xr.DataArray
    var_name=None,
    nsteps=30,
    dim="time",         # "time" | "depth"
    time_idx=0,
    depth_idx=0,
)
```

Animate a variable across one dimension by redrawing frames in place (for
notebooks). With `dim="time"` it steps over time and `depth_idx` fixes the level;
with `dim="depth"` it steps over depth and `time_idx` fixes the time step.

| Parameter | Description |
|---|---|
| `data` | A DataArray, or a Dataset (then `var_name` is required) |
| `var_name` | Variable to animate; only needed when `data` is a Dataset |
| `nsteps` | Maximum number of frames along the animated dimension. Defaults to `30` |
| `dim` | Dimension to animate: `"time"` (default) or `"depth"` |
| `time_idx` | Fixed time step used when `dim="depth"`. Defaults to `0` |
| `depth_idx` | Fixed depth level used when `dim="time"`. Defaults to `0` |

Raises `ValueError` if `data` is a Dataset and `var_name` is omitted, or if `dim`
is not a dimension of the data.

---

## `plot_interactive_map()`

```python
plot_interactive_map(
    df,                 # pd.DataFrame | gpd.GeoDataFrame
    lat=None,
    lon=None,
    hover_cols=None,
    export_path=None,
    map_style="open-street-map",
    zoom=4,
)
```

Interactive Plotly scatter map of point locations. Accepts a plain DataFrame with
explicit lat/lon columns, or a point GeoDataFrame — the latter is reprojected to
WGS84 and its coordinates are derived from the geometry, so `lat`/`lon` are
ignored. MultiPoint geometry is exploded to one point per row; non-point geometry
(polygons/lines) raises.

| Parameter | Description |
|---|---|
| `df` | A DataFrame with lat/lon columns, or a point/multipoint GeoDataFrame (reprojected to WGS84) |
| `lat` / `lon` | Coordinate column names. Required for a plain DataFrame; ignored for a GeoDataFrame |
| `hover_cols` | Column name(s) to show on hover. `None` shows no extra data |
| `export_path` | If given, also saves the map as standalone HTML at this path |
| `map_style` | Plotly basemap. Defaults to `"open-street-map"`. For exported HTML prefer `"carto-positron"` / `"carto-darkmatter"` — OSM tiles 403 without a Referer (e.g. a `file://` HTML), while the Carto basemaps need no token or Referer |
| `zoom` | Initial zoom level. Defaults to `4` |

Raises `ValueError` if `df` is empty, if lat/lon are missing for a plain
DataFrame, or if a GeoDataFrame carries non-point geometry.

```python
import geopandas as gpd
gdf = gpd.read_file("stations.gpkg")          # any CRS; point geometry
plot_interactive_map(gdf, hover_cols=["station_id", "depth"],
                     export_path="stations.html", map_style="carto-positron")
```

---

## `plot_records_on_field()`

```python
plot_records_on_field(
    data,               # pd.DataFrame | gpd.GeoDataFrame
    var_key,
    var=None,
    time_col="date",
    lon_col="lon",
    lat_col="lat",
    offset=1.0,
    max_plots=12,
    title_fn=None,
)
```

For each of the first `max_plots` records, open the gridded variable field
([`ZarrCatalog`](zarr_catalog.md)) in a small bbox around the record's location
and overlay that location in red. Records with no data for their date/bbox are
skipped with a warning. A GeoDataFrame is reprojected to WGS84; its geometry
bounds (± `offset`) set the bbox and the geometry is drawn in red. A plain
DataFrame uses its point (± `offset`) and a red marker.

| Parameter | Description |
|---|---|
| `data` | Records to plot. GeoDataFrame (active geometry, reprojected to WGS84) or DataFrame (`lon_col`/`lat_col`) |
| `var_key` | Variable key store passed to `ZarrCatalog` |
| `var` | Data variable name within the dataset. Defaults to the first data variable in the store (names need not match the key); the available names are logged so you can pick one |
| `time_col` | Date column. Defaults to `"date"` |
| `lon_col` / `lat_col` | Coordinate columns, used only when `data` has no geometry. Default `"lon"` / `"lat"` |
| `offset` | Half-width (degrees) of the bbox drawn around each location. Defaults to `1.0` |
| `max_plots` | Maximum number of records to plot. Defaults to `12` |
| `title_fn` | Callable mapping a row to a plot title. Defaults to the record date |

```python
# Inspect the SST field around the first 12 observation records
plot_records_on_field(observations_gdf, var_key="sst", offset=2.0,
                      title_fn=lambda r: f"{r['id_row']} — {r['date']:%Y-%m-%d}")
```
