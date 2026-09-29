"""
Reading Zarr data for one variable's store.

Resolves which files to open through a :class:`~h2mare.storage.zarr_index.ZarrIndex`,
then opens them with xarray, applying variable selection, spatial subsetting and
time normalisation. Holds no state of its own — the catalog bookkeeping and its
cache live entirely in the index.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.models import StoreDtype, TimeStep
from h2mare.storage.zarr_index import ZarrIndex
from h2mare.types import BBox, DateLike, DateRange, DatesLike
from h2mare.utils.datetime_utils import end_of_day, normalize_dates
from h2mare.utils.spatial import sel_padded_bbox

# Aliased rather than re-implemented: an inclusive `slice` here needs the same
# whole-day upper bound the compile and download paths do.
_end_of_day = end_of_day

#: Coordinates whose float drift between files is worth snapping away, each
#: mapped to the largest per-point disagreement still treated as one axis
#: written twice rather than two different axes. Public because
#: ``scripts/repair_axis_drift.py`` repairs exactly what this reader snaps, and
#: the two drifting apart is the failure mode that leaves a store unreadable.
#:
#: lat/lon are degrees. Observed drift is ~5e-12 degrees — a sub-micrometre
#: relabel — while the coarsest grid step in the pipeline is 0.25, so the gap
#: between "noise" and "genuinely different" is nine orders of magnitude wide.
#:
#: depth is metres, and drifts by a float32 ULP rather than a float64 one: the
#: o2 store's 2023+ files label the same 46 model levels up to 6e-5 m away from
#: the older files. A millimetre keeps a 16x margin over that while sitting
#: ~500x below the tightest level spacing (~1 m near the surface), so the gap
#: here is wide too and neither threshold is delicate.
AXIS_SNAP_TOL: dict[str, float] = {"lat": 1e-9, "lon": 1e-9, "depth": 1e-3}


def snap_axes_to_reference(
    ds: xr.Dataset,
    reference: dict[str, np.ndarray],
    tol: float | Mapping[str, float] = AXIS_SNAP_TOL,
) -> tuple[xr.Dataset, dict[str, float]]:
    """
    Replace spatial axes that differ from *reference* only by float noise.

    ``open_mfdataset(combine="by_coords")`` compares coordinate arrays exactly.
    Two files written from the same grid at different times can disagree in the
    last floating-point bits, and xarray then stops treating the axis as shared
    and starts treating it as one to concatenate along — surfacing as
    "Resulting object does not have monotonic global indexes along dimension
    lon", which says nothing about the actual cause.

    Only axes of identical length and within *tol* everywhere are snapped, so a
    genuinely different grid — different extent, spacing or size — still reaches
    xarray and still raises. *tol* is per coordinate, since the axes are not in
    the same units; a bare float applies one limit to all of them.

    Returns:
        The dataset, and ``{coord: max_abs_delta}`` for whatever was snapped.
    """
    snapped: dict[str, float] = {}

    for name, ref in reference.items():
        if name not in ds.coords:
            continue
        current = ds.coords[name].values
        if current.shape != ref.shape or np.array_equal(current, ref):
            continue

        limit = tol if isinstance(tol, (int, float)) else tol.get(name)
        if limit is None:  # no limit configured — not an axis this snaps
            continue

        delta = float(np.abs(current - ref).max())
        if delta > limit:
            continue

        ds = ds.assign_coords({name: ref})
        snapped[name] = delta

    return ds, snapped


class ZarrReader:
    """Opens Zarr datasets for the store catalogued by *index*."""

    def __init__(self, index: ZarrIndex) -> None:
        self._index = index
        # One warning per coordinate per reader, not one per drifted file.
        self._drift_reported: set[str] = set()
        # Likewise one warning per reader for the padded variables, not one
        # per short file.
        self._padding_reported: set[str] = set()

    # ---- index queries the opening paths rely on -------------------------
    # Named to match the index so the opening logic reads the same either way.

    @property
    def var_key(self) -> str:
        return self._index.var_key

    @property
    def df(self) -> pd.DataFrame:
        return self._index.df

    def _log(self, level: str, msg: str) -> None:
        self._index._log(level, msg)

    def get_time_coverage(self) -> DateRange | None:
        return self._index.get_time_coverage()

    def get_paths_in_range(self, start_date: DateLike, end_date: DateLike) -> list[str]:
        return self._index.get_paths_in_range(start_date, end_date)

    def map_dates_to_paths(self, dates):
        return self._index.map_dates_to_paths(dates)

    # ---- opening paths ---------------------------------------------------

    def open_dataset(
        self,
        dates: DatesLike | None = None,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        bbox: BBox | Sequence[float] | None = None,
        variables: str | Sequence[str] | None = None,
        chunks: dict | str | None = "auto",
    ) -> xr.Dataset:
        """
        Open zarr dataset(s) with flexible date and spatial selection.

        Supports two modes:
        1. Sparse dates: Provide `dates` for specific dates
        2. Date range: Provide `start_date` and/or `end_date`

        Args:
            dates: Specific dates to select (sparse mode)
            start_date: Start of date range (range mode)
            end_date: End of date range (range mode)
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            variables: Variables to select (None = all)
            chunks: Chunking strategy for dask ('auto' or dict)

        Returns:
            Lazy xarray Dataset, or None if no data found

        Raises:
            ValueError: If neither dates nor start_date/end_date provided,
                       or if bbox format is invalid

        Example:
            >>> # Sparse dates
            >>> ds = catalog.open_dataset(dates=['2020-01-15', '2020-06-20'])
            >>>
            >>> # Date range
            >>> ds = catalog.open_dataset(
            ...     start_date='2020-01-01',
            ...     end_date='2020-12-31',
            ...     bbox=(-180, -90, 180, 90),
            ...     variables=['mld']
            ... )
        """
        # Validate inputs
        if dates is None and start_date is None and end_date is None:
            date_range = self.get_time_coverage()
            if date_range:
                start_date, end_date = date_range.start, date_range.end
            else:
                raise ValueError(
                    "Please provide sparse 'dates' or 'start_date/end_date' range"
                )

        if dates is not None and (start_date is not None or end_date is not None):
            raise ValueError(
                "Cannot use both 'dates' and 'start_date'/'end_date'. "
                "Use one mode or the other."
            )

        if bbox is not None:
            bbox = bbox if isinstance(bbox, BBox) else BBox.from_tuple(bbox)

        # Route to appropriate method
        if dates is not None:
            return self._open_sparse_dates(
                dates=dates,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )
        else:
            return self._open_date_range(
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )

    def _open_sparse_dates(
        self,
        dates: DatesLike,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for specific sparse dates."""
        date_list = normalize_dates(dates)
        if not date_list:
            raise ValueError("No valid dates provided")

        # Get file paths
        path_mapping = self.map_dates_to_paths(date_list)

        if not path_mapping:
            raise FileNotFoundError(f"No zarr files contain dates: {date_list}")

        paths = list(path_mapping.keys())
        reference = self._reference_axes(paths)
        union = self._reference_data_vars(paths)

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                # Explicit, for the reason storage.py pins it on concat: xarray's
                # default moves from "outer" to "exact". Here "exact" is also the
                # behaviour this reader is built to give. The snap above unifies
                # axes that differ by float noise, and anything coarser is a
                # genuinely different grid that must raise — but under "outer" it
                # does not: two grids 5 degrees apart union into a longer axis of
                # near-duplicate points and the read succeeds with a coordinate
                # axis that is not the grid. That is the silent corruption the
                # snap exists to prevent, arriving by the other door.
                join="exact",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(
                    d, bbox, variables, reference, union
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time coordinates
        ds = self._undo_decode_widening(self._normalize_time(ds))

        # Select only requested dates
        requested_dates = pd.DatetimeIndex(date_list).normalize()
        day_of_step = pd.DatetimeIndex(ds.time.values).normalize()
        available_dates = day_of_step.unique()

        valid_dates = requested_dates.intersection(available_dates)

        if len(valid_dates) == 0:
            raise FileNotFoundError(
                f"None of the requested dates found in dataset. "
                f"Requested: {requested_dates.tolist()}, "
                f"Available: {available_dates.tolist()}"
            )

        if len(valid_dates) < len(requested_dates):
            missing = requested_dates.difference(available_dates)
            self._log("warning", f"Missing dates: {missing.tolist()}")

        # Select by calendar day rather than exact timestamp: on a sub-daily
        # axis only the midnight step matches a normalized date, so exact
        # selection would return one hour per requested day. Equivalent to the
        # old timestamp selection for a daily store.
        return ds.isel(time=np.flatnonzero(day_of_step.isin(valid_dates)))

    def _open_date_range(
        self,
        start_date: DateLike | None,
        end_date: DateLike | None,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for continuous date range."""

        # Get catalog
        df = self.df

        if df.empty:
            raise FileNotFoundError("Catalog is empty")

        # Default to full range if not specified
        start: pd.Timestamp = (
            df["start_date"].min()
            if start_date is None
            else pd.to_datetime(start_date).normalize()
        )
        end: pd.Timestamp = (
            df["end_date"].max()
            if end_date is None
            else pd.to_datetime(end_date).normalize()
        )

        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                "Date range contains NaT — check catalog date columns for missing values"
            )

        # Warn if requested range extends beyond what the catalog covers
        available_start = pd.Timestamp(df["start_date"].min()).normalize()
        available_end = pd.Timestamp(df["end_date"].max()).normalize()

        if start_date is not None and start < available_start:
            self._log(
                "warning",
                f"[{self.var_key}] Requested start {start.date()} not available — "
                f"opening from {available_start.date()}",
            )
            start = available_start

        if end_date is not None and end > available_end:
            self._log(
                "warning",
                f"[{self.var_key}] Requested end {end.date()} not available — "
                f"opening until {available_end.date()}",
            )
            end = available_end

        # Get overlapping paths
        paths = self.get_paths_in_range(start, end)

        if not paths:
            raise FileNotFoundError(
                f"No zarr files found for range: {start.date()} to {end.date()}"
            )

        reference = self._reference_axes(paths)
        union = self._reference_data_vars(paths)

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                # Explicit, for the reason storage.py pins it on concat: xarray's
                # default moves from "outer" to "exact". Here "exact" is also the
                # behaviour this reader is built to give. The snap above unifies
                # axes that differ by float noise, and anything coarser is a
                # genuinely different grid that must raise — but under "outer" it
                # does not: two grids 5 degrees apart union into a longer axis of
                # near-duplicate points and the read succeeds with a coordinate
                # axis that is not the grid. That is the silent corruption the
                # snap exists to prevent, arriving by the other door.
                join="exact",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(
                    d, bbox, variables, reference, union
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time
        ds = self._undo_decode_widening(self._normalize_time(ds))

        # Select time range. `end` names a calendar day, so the window runs to
        # the *end* of that day, not to its midnight stamp: an inclusive slice
        # at midnight returns only the first step of the final day on a
        # sub-daily axis, silently dropping the other 23 hours. Identical to
        # slice(start, end) for a daily store, whose stamps are all midnight.
        return ds.sel(time=slice(start, _end_of_day(end)))

    def _normalize_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Normalize time coordinates to midnight (00:00:00), for daily stores only.

        Daily products are often published stamped at 12:00, and callers select
        by calendar day, so snapping to midnight is what makes ``sel`` and the
        date bookkeeping line up.

        Skipped for an ``HOURLY`` store: there the sub-daily stamps *are* the
        data, and snapping them would map all 24 steps of a day onto one
        timestamp — 8784 steps collapsing to 366 duplicated stamps for a year,
        silently and without error.

        Args:
            ds: Dataset with time coordinate

        Returns:
            Dataset with normalized time, or *ds* unchanged for an hourly store.
        """
        if "time" not in ds.coords:
            return ds

        if self._time_step is TimeStep.HOURLY:
            return ds

        normalized_time = pd.to_datetime(ds["time"].values).normalize()
        return ds.assign_coords(time=normalized_time)

    def _undo_decode_widening(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Bring an int16 store's variables back to float32 after CF decoding.

        scale_factor/add_offset decoding promotes to float64 regardless of the
        stored dtype — zarr attributes round-trip through JSON, so the scale
        loses its own dtype on the way out. Left alone, an int16 store would cost
        twice the memory of the float32 one it replaced. Only applied to stores
        declared int16, so nothing else is touched.
        """
        cfg = self._index.var_config
        if getattr(cfg, "store_dtype", None) is not StoreDtype.INT16:
            return ds
        casts = {
            name: da.astype("float32")
            for name, da in ds.data_vars.items()
            if da.dtype == "float64"
        }
        return ds.assign(casts) if casts else ds

    @property
    def _time_step(self) -> TimeStep:
        """
        Configured cadence for this store, defaulting to DAILY.

        Read from the index's ``var_config`` rather than global settings so an
        injected ``app_config`` is honoured. Defaults when the entry is a
        stand-in without the field, keeping config-free reads on the old path.
        """
        return getattr(self._index.var_config, "time_step", TimeStep.DAILY)

    @staticmethod
    def _reference_axes(paths: Sequence) -> dict[str, np.ndarray]:
        """
        The spatial axes every file in this open is snapped onto.

        Taken from the lowest-sorted path so one read is self-consistent and
        repeatable, and only when more than one file is involved — a single
        file is never combined against anything and defines its own axes.

        Failure here is not fatal: without a reference the snap is skipped and
        xarray behaves exactly as it did before.
        """
        if len(paths) < 2:
            return {}

        try:
            with xr.open_zarr(sorted(map(str, paths))[0], consolidated=False) as ds:
                return {
                    name: ds.coords[name].values
                    for name in AXIS_SNAP_TOL
                    if name in ds.coords
                }
        except Exception as e:  # noqa: BLE001 - advisory only
            logger.debug(f"Could not read reference axes for snapping: {e}")
            return {}

    @staticmethod
    def _reference_data_vars(paths: Sequence) -> dict[str, tuple[str, ...]]:
        """
        Every data variable any file in this open holds, with its dims.

        ``combine_by_coords`` groups datasets *by their set of data variables*
        before combining anything (xarray ``structure/combine.py``, ``vars_as_keys``),
        combines each group along time on its own, then merges the groups under
        ``join``. A store whose files do not all carry the same variables
        therefore arrives at that merge as two cubes covering different periods,
        and ``join="exact"`` refuses to align them — reported as "cannot align
        objects ... along these coordinates (dimensions): 'time'", which names
        the axis the cubes differ on rather than the variables that split them.

        h2ds is ragged by design: ``run -v X`` compiles only X's columns, and a
        source that does not reach into the current year yet (seapodym, the
        productivity trio) leaves that year's file short. Padding each file up
        to this union puts them all back in one group, so the merge never has to
        align anything.

        Failure here is not fatal: without a union the padding is skipped and
        xarray behaves exactly as it did before.
        """
        if len(paths) < 2:
            return {}

        union: dict[str, tuple[str, ...]] = {}
        try:
            for path in sorted(map(str, paths)):
                with xr.open_zarr(path, consolidated=False) as ds:
                    for name, da in ds.data_vars.items():
                        union.setdefault(str(name), tuple(map(str, da.dims)))
        except Exception as e:  # noqa: BLE001 - advisory only
            logger.debug(f"Could not read reference data_vars for padding: {e}")
            return {}

        return union

    def _preprocess_dataset(
        self,
        ds: xr.Dataset,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        reference: dict[str, np.ndarray] | None = None,
        data_vars_union: dict[str, tuple[str, ...]] | None = None,
    ) -> xr.Dataset:
        """
        Preprocess dataset: apply bbox and variable selection.

        This runs BEFORE datasets are combined in open_mfdataset.

        Args:
            ds: Input dataset
            bbox: Bounding box to apply
            variables: Variables to select
            reference: Spatial axes to snap float-drifted ones onto, from
                :meth:`_reference_axes`. Omitted for a single-file open, where
                there is nothing to combine against.
            data_vars_union: Variables every file is padded up to, from
                :meth:`_reference_data_vars`. Omitted for a single-file open,
                which is never grouped against anything.

        Returns:
            Preprocessed dataset
        """
        # Select variables
        if variables is not None:
            var_list = [variables] if isinstance(variables, str) else list(variables)
            # Only select variables that exist
            available = set(ds.data_vars.keys())
            to_select = [v for v in var_list if v in available]

            if not to_select:
                self._log(
                    "warning",
                    f"None of requested variables found. "
                    f"Requested: {var_list}, Available: {list(available)}",
                )
            else:
                ds = ds[to_select]

        # Ensure lat is monotonically increasing (ERA5 comes north→south)
        if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
            ds = ds.sortby("lat")

        # After the orientation fix so both sides are compared the same way
        # round, and before the bbox subset so every file is cut from an
        # identical axis and the subsets line up too.
        if reference:
            ds, snapped = snap_axes_to_reference(ds, reference)
            for coord, delta in snapped.items():
                if coord in self._drift_reported:
                    continue
                self._drift_reported.add(coord)
                self._log(
                    "warning",
                    f"[{self.var_key}] '{coord}' differs between store files by "
                    f"up to {delta:.2e} — float noise from the same axis being "
                    f"written more than once, snapped so the files can be "
                    f"combined. The store is inconsistent; re-converting the odd "
                    f"file out settles it.",
                )

        # Apply spatial subset
        if bbox is not None:
            ds = self._apply_bbox(ds, bbox)

        # Last, so the placeholders are cut to the subset the real variables
        # were cut to and cost nothing to build.
        if data_vars_union:
            ds = self._pad_to_union(ds, data_vars_union, variables)

        return ds

    def _pad_to_union(
        self,
        ds: xr.Dataset,
        union: dict[str, tuple[str, ...]],
        variables: str | Sequence[str] | None,
    ) -> xr.Dataset:
        """
        Add the variables this file lacks as all-NaN, so every file combines
        as one group. See :meth:`_reference_data_vars` for why that matters.

        Filtered by *variables* for the same reason the selection above exists:
        padding back a variable the caller asked not to have would re-split the
        group it was meant to keep whole.
        """
        wanted = set(union)
        if variables is not None:
            requested = [variables] if isinstance(variables, str) else list(variables)
            wanted &= set(requested)

        missing = [name for name in wanted if name not in ds.data_vars]
        if not missing:
            return ds

        placeholders = {}
        for name in missing:
            dims = union[name]
            # A dim this file does not have leaves nothing sensible to shape the
            # placeholder to. Skipped rather than reshaped, so the group splits
            # and xarray says so, instead of a quietly wrong-shaped variable.
            if any(d not in ds.sizes for d in dims):
                continue
            shape = tuple(ds.sizes[d] for d in dims)
            # float32 rather than the stored dtype: an int16 store decodes to
            # float on the way out anyway (see _undo_decode_widening), and NaN
            # has no integer spelling.
            placeholders[name] = (dims, np.full(shape, np.nan, dtype="float32"))

        if not placeholders:
            return ds

        if not self._padding_reported:
            self._padding_reported.update(placeholders)
            self._log(
                "warning",
                f"[{self.var_key}] {sorted(placeholders)} are absent from part of the "
                f"store and read back as NaN there. Expected while a source lags "
                f"the rest — a full `compile` fills them in once it has data.",
            )

        return ds.assign(placeholders)

    def _apply_bbox(
        self,
        ds: xr.Dataset,
        bbox: BBox,
    ) -> xr.Dataset:
        """
        Apply bounding box selection.

        Args:
            ds: Input dataset
            bbox: [xmin, ymin, xmax, ymax]

        Returns:
            Spatially subset dataset. A dataset without lat/lon (or y/x) cannot
            be subset and is returned whole, with a warning.

        Raises:
            ValueError: If the selection itself fails — an axis that is not
                monotonic, say. That used to be logged and the *unclipped*
                dataset returned, so a caller asking for a region got the whole
                domain as though it were the region.
        """
        # Determine coordinate names (support lat/lon or y/x)
        lat_coord = "lat" if "lat" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "x"

        if lat_coord not in ds.coords or lon_coord not in ds.coords:
            self._log(
                "warning",
                f"Cannot apply bbox: missing coordinates. "
                f"Available: {list(ds.coords.keys())}",
            )
            return ds

        try:
            return sel_padded_bbox(
                ds, bbox.to_tuple(), lat_coord=lat_coord, lon_coord=lon_coord
            )
        except Exception as e:
            raise ValueError(
                f"[{self.var_key}] could not subset to {bbox}: {type(e).__name__}: {e}"
            ) from e
