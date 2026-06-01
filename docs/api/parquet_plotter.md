# ParquetPlotter

`ParquetPlotter` is the visualization accessor for [`ParquetIndexer`](parquet_indexer.md). Access it via `indexer.plot` — do not instantiate it directly.

```python
idx.plot.time_series("sst", agg_by="month")
idx.plot.spatial_maps("sst", agg_by="season")
```

---

## `time_series()`

```python
idx.plot.time_series(
    var_name,         # str, or list[str] of up to 4 variables
    agg_by,           # "day" | "week" | "month" | "season" | "year"
    dates=None,
    bbox=None,
)
```

Returns an interactive Plotly line chart comparing one or more variables, each
aggregated (mean) over space and time. Pass a list to overlay up to 4 variables;
each gets its own y-axis (1 = left, 2 = left/right, 3–4 = floated outward) so
fields with different units/scales stay comparable. For a single variable's full
distribution (±1σ, min/max, trend lines) use [`stats_summary()`](#stats_summary)
instead.

| Parameter | Description |
|---|---|
| `var_name` | Variable column to plot, or a `list[str]` of up to 4 columns (one line + y-axis each) |
| `agg_by` | Temporal aggregation: `"day"`, `"week"`, `"month"`, `"season"`, or `"year"` |
| `dates` | `(start, end)` tuple or `list[str]` of dates. Defaults to full dataset |
| `bbox` | `(xmin, ymin, xmax, ymax)` for an area, or `(lon, lat)` to select the nearest grid cell. Defaults to full extent |
| `title` | Figure title. Defaults to the variable long name (single) or `"Time series"` (multiple) |

Raises `ValueError` if no variables are given, more than 4 are given, or any
variable is absent from the store.

Seasonal values are assigned to the first month of the season (e.g. spring → March 1st) for plotting purposes.

```python
# Compare SST and chlorophyll at a single grid point
idx.plot.time_series(["sst", "chl"], agg_by="month", bbox=(-30, 40))
```

---

## `stats_summary()`

```python
idx.plot.stats_summary(
    var_name,
    agg_by,           # "day" | "week" | "month" | "season" | "year"
    dates=None,
    bbox=None,
    lowess_frac=0.3,
    title=None,
)
```

Single-variable distribution over time: mean line, ±1σ shaded band, min/max
lines, and LOWESS trend lines for mean/min/max. Use this for one variable in
depth; use [`time_series()`](#time_series) to compare several variables.

| Parameter | Description |
|---|---|
| `var_name` | Single variable column to plot |
| `agg_by` | Temporal aggregation: `"day"`, `"week"`, `"month"`, `"season"`, or `"year"` |
| `dates` | `(start, end)` tuple or `list[str]` of dates. Defaults to full dataset |
| `bbox` | `(xmin, ymin, xmax, ymax)` area filter. Defaults to full extent |
| `lowess_frac` | Fraction of data per local LOWESS fit (0 < frac ≤ 1). Lower = follows data more closely; higher = smoother. Defaults to `0.3` |
| `title` | Figure title. Defaults to `"{long_name} — Statistics Summary"` |

`std` is `null` for buckets with a single observation, rendered as gaps in the band.

---

## `spatial_maps()`

```python
idx.plot.spatial_maps(
    var_name,
    agg_by="month",   # "month" | "season"
    dates=None,
    data_bbox=None,
    map_bbox=None,
    vminmax=None,
    title=None,
    legend_title=None,
    save_path=None,
)
```

Climatological panel maps — 12 panels for `agg_by="month"`, 4 for `agg_by="season"`. Each panel shows the long-term mean at every grid cell across all years in the selected data.

| Parameter | Description |
|---|---|
| `var_name` | Variable column to plot |
| `agg_by` | `"month"` (12 panels) or `"season"` (4 panels) |
| `dates` | Date range or list for filtering. Defaults to full dataset |
| `data_bbox` | Spatial filter applied before aggregation |
| `map_bbox` | Visible region on each panel. Defaults to extent of loaded data |
| `vminmax` | Fixed `(vmin, vmax)` for the colorbar. Defaults to data range |
| `title` | Figure title |
| `legend_title` | Colorbar label. Defaults to the variable short name from config |
| `save_path` | Path to save the figure. If `None`, shown interactively |

---

## Displaying interactive figures

`time_series()` and `stats_summary()` return a `go.Figure` so they stay composable (e.g. `fig.write_html(...)`). Plotly's toolbar (`displayModeBar`) defaults to `"hover"`, so it only appears while the cursor is over the plot and fades as you move toward it. Use `show()` to pin it:

```python
fig = idx.plot.time_series(["sst", "chl"], agg_by="month")
idx.plot.show(fig)                       # toolbar always visible, no Plotly logo
idx.plot.show(fig, displaylogo=True)     # override individual config keys
```

`show()` applies `ParquetPlotter.PLOT_CONFIG` (`{"displayModeBar": True, "displaylogo": False}`). To keep the toolbar when exporting or showing yourself, pass it directly:

```python
fig.show(config=ParquetPlotter.PLOT_CONFIG)
fig.write_html("ts.html", config=ParquetPlotter.PLOT_CONFIG)
```

(`spatial_maps()` is Matplotlib-based and unaffected.)

---

## Caching

Aggregation results are cached internally by `(var_name, agg_by, dates, bbox)`. Call `idx.plot.clear_cache()` to invalidate manually, or it is cleared automatically after each `add_data()` call on the parent indexer.
