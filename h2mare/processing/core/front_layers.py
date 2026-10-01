"""
Front layers for species distribution models, declared per var_key in
``front_layers`` (:class:`h2mare.models.FrontLayerSpec`).

Replaces the BOA front distances (``processing/core/fronts.py``) with layers
built for L4 analyses; the design, and the evidence behind every parameter, is
``plans/front-layers.md``. Per day and per field:

1. transform (chl -> log10) and fill gaps from the nearest valid cell, so a
   coastline or ice edge is not read as a front;
2. smooth with a Gaussian of width ``sigma_km`` **in km on both axes**, at the
   scale the analysis actually resolves;
3. take the gradient per km, with the cos(lat) metric, so its magnitude means
   the same at every latitude, orientation and grid resolution;
4. thin to ridge lines (non-maximum suppression along the gradient);
5. keep ridges by hysteresis: pixels >= ``low_per_km`` connected to one
   >= ``high_per_km``;
6. mask what cannot be assessed: gaps, and where the field's own confidence
   variable exceeds its cut.

Outputs per layer name: the daily front mask (1 front, 0 no front, NaN not
assessed), the gradient magnitude, and the front frequency over the last N
days (fronts / days assessed).
"""

from __future__ import annotations

import shutil
import time
from collections import deque
from typing import Any, Callable, Optional, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter1d,
    label,
    maximum_filter1d,
)
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from h2mare.models import FrontLayerSpec
from h2mare.processing.core import fronts as _boa
from h2mare.utils.parallel import resolve_n_workers
from h2mare.utils.spatial import haversine_min_distance_kdtree

#: km per degree of latitude.
KM_PER_DEG = 111.2


def prepare_field(
    values: NDArray, transform: str
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    The field in detection units, with gaps filled from the nearest valid cell.

    Returns the filled field and the gap mask (True where the input had no
    value). ``log10`` treats non-positive values as gaps.
    """
    f = np.asarray(values, dtype="float64")
    if transform == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.log10(np.where(f > 0, f, np.nan))
    gaps = np.isnan(f)
    if gaps.any() and not gaps.all():
        nearest = distance_transform_edt(
            gaps, return_distances=False, return_indices=True
        )
        f = f[tuple(np.asarray(nearest))]
    return f, gaps


def cell_size_km(lat: NDArray, lon: NDArray) -> tuple[float, NDArray[np.float64]]:
    """North-south cell size (km) and the east-west size per row (km)."""
    dy = abs(float(lat[1] - lat[0])) * KM_PER_DEG
    dx = abs(float(lon[1] - lon[0])) * KM_PER_DEG * np.cos(np.deg2rad(lat))
    return dy, np.asarray(dx, dtype="float64")


def smooth_km(f: NDArray, lat: NDArray, lon: NDArray, sigma_km: float) -> NDArray:
    """
    Gaussian smoothing of width *sigma_km* in km along both axes.

    Separable: the north-south width is one number, the east-west width is set
    per row because a longitude cell narrows with latitude. A width in cells
    taken from the latitude spacing would smooth less, in km, east-west at high
    latitude.
    """
    dy, dx = cell_size_km(lat, lon)
    s = gaussian_filter1d(f, sigma_km / dy, axis=0, mode="nearest")
    return np.stack(
        [gaussian_filter1d(row, sigma_km / w, mode="nearest") for row, w in zip(s, dx)]
    )


def metric_gradient(
    f: NDArray, lat: NDArray, lon: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Gradient magnitude per km, plus the per-cell components for thinning.

    Centred differences divided by the true cell size, so an east-west and a
    north-south gradient of the same physical strength have the same
    magnitude at any latitude.
    """
    dy, dx = cell_size_km(lat, lon)
    gy_cell, gx_cell = np.gradient(f)
    mag = np.hypot(gx_cell / dx[:, None], gy_cell / dy)
    return mag, gx_cell, gy_cell


def thin(mag: NDArray, gx: NDArray, gy: NDArray) -> NDArray[np.bool_]:
    """
    Non-maximum suppression: keep pixels that peak across the front.

    The gradient direction is binned into four (0, 45, 90, 135 degrees in index
    space) and each pixel is compared with its two neighbours along it.
    """
    angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    p = np.pad(mag, 1, mode="edge")
    c = p[1:-1, 1:-1]
    east, west = p[1:-1, 2:], p[1:-1, :-2]
    north, south = p[2:, 1:-1], p[:-2, 1:-1]
    ne, sw = p[2:, 2:], p[:-2, :-2]
    nw, se = p[2:, :-2], p[:-2, 2:]
    b0 = (angle < 22.5) | (angle >= 157.5)
    b45 = (angle >= 22.5) & (angle < 67.5)
    b90 = (angle >= 67.5) & (angle < 112.5)
    b135 = (angle >= 112.5) & (angle < 157.5)
    return (
        (b0 & (c >= east) & (c >= west))
        | (b90 & (c >= north) & (c >= south))
        | (b45 & (c >= ne) & (c >= sw))
        | (b135 & (c >= nw) & (c >= se))
    )


def hysteresis(
    mag: NDArray, candidates: NDArray[np.bool_], low: float, high: float
) -> NDArray[np.bool_]:
    """
    Keep each 8-connected group of candidates >= *low* holding one >= *high*.

    A strong pixel seeds a front and a weak one extends it; a weak ridge that
    never touches a strong one is dropped, which is what removes the speckle a
    single per-pixel threshold leaves.
    """
    weak = candidates & (mag >= low)
    labels, n = label(weak, structure=np.ones((3, 3), bool))  # type: ignore[misc]
    if n == 0:
        return weak
    seeds = np.unique(labels[weak & (mag >= high)])
    return np.isin(labels, seeds[seeds > 0])


def detect_fronts(
    values: NDArray,
    lat: NDArray,
    lon: NDArray,
    *,
    transform: str,
    sigma_km: float,
    low_per_km: float,
    high_per_km: float,
    confidence: Optional[NDArray] = None,
    confidence_max: Optional[float] = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    One day's front mask and gradient magnitude.

    Args:
        values: The field, lat x lon, in its stored units (chl in mg m-3).
        lat, lon: Increasing, regular axes in degrees.
        transform: ``"none"`` or ``"log10"``.
        sigma_km: Smoothing width in km.
        low_per_km, high_per_km: Hysteresis thresholds, in the transformed
            field's units per km.
        confidence: The field's own uncertainty, same shape, or None.
        confidence_max: Pixels above this cannot be assessed.

    Returns:
        ``mask``: 1 front, 0 no front, NaN not assessed (a gap, or confidence
        above the cut); float32. ``grad``: the gradient magnitude per km, NaN
        on gaps; float32.
    """
    f, gaps = prepare_field(values, transform)
    shape = np.shape(values)
    if gaps.all():
        nan = np.full(shape, np.nan, "float32")
        return nan, nan.copy()

    mag, gx, gy = metric_gradient(smooth_km(f, lat, lon, sigma_km), lat, lon)
    assessed = ~gaps
    if confidence is not None and confidence_max is not None:
        with np.errstate(invalid="ignore"):
            assessed &= ~(np.asarray(confidence) > confidence_max)

    fronts = hysteresis(mag, thin(mag, gx, gy) & assessed, low_per_km, high_per_km)
    mask = np.where(assessed, fronts, np.nan).astype("float32")
    grad = np.where(gaps, np.nan, mag).astype("float32")
    return mask, grad


def front_frequency(masks: NDArray, min_assessed: int) -> NDArray[np.float32]:
    """
    Share of the assessed days on which each pixel was a front.

    Args:
        masks: ``(days, lat, lon)`` stack of daily masks (1, 0, NaN).
        min_assessed: Fewest assessed days for a value; fewer gives NaN.

    A day the pixel could not be assessed (masked, or a gap) drops out of the
    denominator. Counting it as "no front" lowered the frequency wherever the
    confidence mask applied, and the coarse OSTIA error field turned that into
    square holes (plans/front-layers.md, §4.3).
    """
    assessed = np.isfinite(masks)
    n = assessed.sum(axis=0)
    fronts = np.where(assessed, masks, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = fronts / n
    return np.where(n >= min_assessed, freq, np.nan).astype("float32")


# ============================================================
#   FREQUENCY
# ============================================================


def proximity_mask(
    mask: NDArray, lat: NDArray, lon: NDArray, radius_km: float
) -> NDArray[np.float32]:
    """
    1 where a front lies within *radius_km* (a box, in km on both axes), 0
    where none does, NaN where the pixel itself was not assessed.

    The analyses place fronts to within tens of km, so an exact pixel hit is
    mostly position noise; see ``FrontLayerSpec.frequency_radius_km``.
    """
    front = np.nan_to_num(mask) > 0
    if radius_km > 0 and front.any():
        dy, dx = cell_size_km(lat, lon)
        near = maximum_filter1d(front, 2 * int(round(radius_km / dy)) + 1, axis=0)
        near = np.stack(
            [
                maximum_filter1d(row, 2 * int(round(radius_km / w)) + 1)
                for row, w in zip(near, dx)
            ]
        )
    else:
        near = front
    return np.where(np.isfinite(mask), near, np.nan).astype("float32")


def persistent_distance(
    freq: NDArray, lat: NDArray, lon: NDArray, min_frequency: float
) -> NDArray[np.float32]:
    """
    Great-circle km from each pixel to the nearest persistent frontal zone.

    The zone is every pixel whose front frequency is at least *min_frequency*:
    a front recurred within ``frequency_radius_km`` on that share of the
    assessed days. Inside it the distance is 0. NaN where the frequency itself
    is (too few days assessed), and everywhere when no pixel reaches the cut —
    with nothing to measure to there is no distance.
    """
    assessed = np.isfinite(freq)
    zone = assessed & (np.nan_to_num(freq) >= min_frequency)
    out = np.full(freq.shape, np.nan, dtype="float32")
    if not zone.any():
        return out
    out[zone] = 0.0
    outside = assessed & ~zone
    if outside.any():
        # The nearest zone pixel to anything outside lies on the zone's edge,
        # so the tree only needs the edge: a fraction of the zone's size.
        edge = zone & ~np.asarray(binary_erosion(zone), dtype=bool)
        la, lo = np.meshgrid(lat, lon, indexing="ij")
        out[outside] = haversine_min_distance_kdtree(
            np.column_stack([la[outside], lo[outside]]),
            np.column_stack([la[edge], lo[edge]]),
        )
    return out


class RollingFrequency:
    """
    Front frequency over trailing windows of calendar days, fed one day at a
    time in date order.

    Calendar days, not entries: a day missing from the axis inside a window is
    simply not assessed, like a masked one. A value needs at least half the
    window's days assessed.
    """

    def __init__(self, windows: list[int], lat, lon, radius_km: float):
        self.windows = sorted(windows)
        self.lat, self.lon, self.radius_km = lat, lon, radius_km
        self._days: deque[tuple[pd.Timestamp, NDArray[np.float32]]] = deque()

    def push(self, day: pd.Timestamp, mask: NDArray) -> None:
        if self._days and day <= self._days[-1][0]:
            raise ValueError(f"days must be pushed in increasing order; got {day}")
        self._days.append(
            (day, proximity_mask(mask, self.lat, self.lon, self.radius_km))
        )
        oldest = day - pd.Timedelta(days=self.windows[-1] - 1)
        while self._days[0][0] < oldest:
            self._days.popleft()

    def frequency(self, window: int) -> NDArray[np.float32]:
        """Frequency over the *window* calendar days ending at the last push."""
        last = self._days[-1][0]
        start = last - pd.Timedelta(days=window - 1)
        stack = np.stack([m for d, m in self._days if d >= start])
        return front_frequency(stack, min_assessed=(window + 1) // 2)


# ============================================================
#   CONVERT STEP
# ============================================================

#: Pool size when an entry sets no ``n_workers``; capped by the host and by
#: ``H2MARE_MAX_WORKERS`` (``utils.parallel.resolve_n_workers``).
DEFAULT_N_WORKERS = 10

#: Reads a var_key's stored front masks: ``(mask variable, first day, last
#: day) -> DataArray (time, lat, lon)``, or None when there are none.
MaskReader = Callable[[str, pd.Timestamp, pd.Timestamp], Optional[xr.DataArray]]


#: Attempts at one store read before giving up.
#:
#: The store's drive has answered a burst of concurrent chunk reads with a
#: transient ``OSError: [Errno 22] Invalid argument`` that read back fine
#: moments later (plans/front-layers.md §6.6). Only OSError is retried, and not
#: FileNotFoundError: a missing file will not appear by waiting.
READ_ATTEMPTS = 4
READ_WAIT = wait_exponential(multiplier=1, min=2, max=30)

_T = TypeVar("_T")


def _transient(e: BaseException) -> bool:
    return isinstance(e, OSError) and not isinstance(e, FileNotFoundError)


def read_retrying(fn: Callable[[], _T], what: str) -> _T:
    """``fn()``, retried on a transient OSError; the last failure is re-raised."""

    def _log(state) -> None:
        exc = state.outcome.exception()
        logger.warning(
            f"reading {what}: attempt {state.attempt_number} failed "
            f"({type(exc).__name__}: {exc}); retrying in {state.next_action.sleep:.0f}s"
        )

    for attempt in Retrying(
        stop=stop_after_attempt(READ_ATTEMPTS),
        wait=READ_WAIT,
        retry=retry_if_exception(_transient),
        before_sleep=_log,
        reraise=True,
    ):
        with attempt:
            return fn()
    raise AssertionError("unreachable: Retrying either returns or raises")


def store_mask_reader(catalog: Any, owner: str) -> MaskReader:
    """
    A :data:`MaskReader` over a var_key's own store (a ``ZarrCatalog``).

    None when the window holds no stored mask: no file, or files written
    before the layer was declared. A window only partly covered reads as
    not assessed where it is missing, which is what those days are.
    """

    def read(
        var: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> Optional[xr.DataArray]:
        try:
            ds = catalog.open_dataset(start_date=start, end_date=end, variables=[var])
        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.debug(
                f"[{owner}] no stored {var} for {start.date()}..{end.date()}: {e}"
            )
            return None
        try:
            if var not in ds.data_vars or ds.sizes.get("time", 0) == 0:
                return None
            return read_retrying(ds[var].load, f"{owner} {var}")
        finally:
            ds.close()

    return read


def _detect_task(args) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    values, confidence, lat, lon, params = args
    return detect_fronts(values, lat, lon, confidence=confidence, **params)


class _SerialMap:
    """``pool.map`` in-process: the refresh covers a few weeks of days."""

    @staticmethod
    def map(fn, tasks):
        return [fn(t) for t in tasks]


def _pdist_task(args) -> NDArray[np.float32]:
    freq, lat, lon, min_frequency = args
    return persistent_distance(freq, lat, lon, min_frequency)


def _with_pdist(
    out: dict[str, list], spec: FrontLayerSpec, name: str, lat, lon, pool
) -> None:
    """Append the persistent distance of each day whose frequency *out* holds."""
    pd_ = spec.persistent_distance
    if pd_ is None:
        return
    freqs = out[f"{name}_ffreq{pd_.window}"]
    tasks = [(f, lat, lon, pd_.min_frequency) for f in freqs]
    out[spec.pdist_names(name)[0]] = list(pool.map(_pdist_task, tasks))


def _params(spec: FrontLayerSpec) -> dict:
    return dict(
        transform=spec.transform,
        sigma_km=spec.sigma_km,
        low_per_km=spec.low_per_km,
        high_per_km=spec.high_per_km,
        confidence_max=spec.confidence.max if spec.confidence else None,
    )


def _seed(
    rolling: RollingFrequency,
    read_masks: Optional[MaskReader],
    mask_var: str,
    first_day: pd.Timestamp,
    shape: tuple[int, int],
    owner: str,
) -> int:
    """Push the stored masks before *first_day* into *rolling*; return how many."""
    if read_masks is None:
        return 0
    span = rolling.windows[-1] - 1
    stored = read_masks(
        mask_var,
        first_day - pd.Timedelta(days=span),  # type: ignore[arg-type]
        first_day - pd.Timedelta(days=1),  # type: ignore[arg-type]
    )
    if stored is None or stored.sizes.get("time", 0) == 0:
        return 0
    if tuple(stored.shape[1:]) != tuple(shape):
        logger.warning(
            f"[{owner}] stored {mask_var} is on a {stored.shape[1:]} grid, not "
            f"{shape}; not seeding the frequency from it"
        )
        return 0
    stored = stored.sortby("time")
    for day, mask in zip(pd.DatetimeIndex(stored["time"].values), stored.values):
        rolling.push(day, mask)
    return int(stored.sizes["time"])


def _layer(
    ds: xr.Dataset,
    name: str,
    spec: FrontLayerSpec,
    owner: str,
    read_masks: Optional[MaskReader],
) -> xr.Dataset:
    """Detect one entry over *ds*'s days; return its outputs, lazily, from staging."""
    src = ds[spec.source]
    conf = ds[spec.confidence.var] if spec.confidence else None
    times = pd.DatetimeIndex(np.atleast_1d(src["time"].values))
    if not times.is_monotonic_increasing:
        raise ValueError(
            f"[{owner}] front_layers.{name}: '{spec.source}' has an unsorted time "
            f"axis; sort it before detecting fronts."
        )
    lat = np.asarray(src["lat"].values, dtype="float64")
    lon = np.asarray(src["lon"].values, dtype="float64")

    rolling = RollingFrequency(spec.frequency_days, lat, lon, spec.frequency_radius_km)
    seeded = _seed(
        rolling,
        read_masks,
        spec.mask_name(name),
        times[0],  # type: ignore[arg-type]
        (len(lat), len(lon)),
        owner,
    )
    stage = _boa.stage_path(owner, f"{name}_layers")
    shutil.rmtree(stage, ignore_errors=True)
    stage.parent.mkdir(parents=True, exist_ok=True)

    n_workers = resolve_n_workers(spec.n_workers, DEFAULT_N_WORKERS, f"{owner}/{name}")
    logger.info(
        f"[{owner}] front layers {name}: {len(times)} day(s), {n_workers} workers, "
        f"{seeded} stored day(s) of history"
    )
    params = _params(spec)
    t0 = time.perf_counter()
    staged = False
    # The BOA module's pool and staging, so its sweep and test fixtures cover
    # these layers too.
    with _boa._pool(n_workers) as pool:
        for batch in _boa._month_batches(times):
            what = f"{owner} {spec.source} {batch[0]:%Y-%m}"
            values = read_retrying(lambda: src.sel(time=batch).values, what)
            confs = (
                read_retrying(lambda: conf.sel(time=batch).values, what)
                if conf is not None
                else [None] * len(batch)
            )
            tasks = [(v, c, lat, lon, params) for v, c in zip(values, confs)]
            results = pool.map(_detect_task, tasks)

            out: dict[str, list] = {
                v: []
                for v in spec.output_names(name)
                if v not in spec.pdist_names(name)
            }
            for day, (mask, grad) in zip(batch, results):
                rolling.push(day, mask)
                out[spec.mask_name(name)].append(mask)
                out[spec.grad_name(name)].append(grad)
                for n, var in zip(spec.frequency_days, spec.freq_names(name)):
                    out[var].append(rolling.frequency(n))
            _with_pdist(out, spec, name, lat, lon, pool)

            block = xr.Dataset(
                {v: (("time", "lat", "lon"), np.stack(a)) for v, a in out.items()},
                coords={"time": batch, "lat": lat, "lon": lon},
            )
            if staged:
                block.to_zarr(stage, append_dim="time", consolidated=False)
            else:
                chunks = (1, min(256, len(lat)), min(256, len(lon)))
                block.to_zarr(
                    stage,
                    consolidated=False,
                    encoding={v: {"chunks": chunks} for v in block.data_vars},
                )
                staged = True
    logger.success(
        f"[{owner}] front layers {name}: {len(times)} day(s) in "
        f"{time.perf_counter() - t0:.1f}s"
    )

    layers = xr.open_zarr(stage, consolidated=False)
    for v in layers.variables:
        # The staging layout must not leak into the store's own chunking.
        layers[v].encoding = {}
    # The source's own coordinate objects, so the merge back aligns exactly.
    return layers.assign_coords(time=src["time"], lat=src["lat"], lon=src["lon"])


def apply_front_layers(
    ds: xr.Dataset,
    specs: Optional[dict[str, FrontLayerSpec]],
    owner: str,
    read_masks: Optional[MaskReader] = None,
) -> xr.Dataset:
    """
    Add each declared entry's outputs to *ds*, in declaration order.

    Detection is eager — a month at a time across a process pool, staged under
    ``INTERIM_DIR`` like the BOA layers, so the same ``clear_staging(owner)``
    sweeps it — and the outputs land on *ds* as lazy views of the staging.

    Args:
        ds: The var_key's dataset after its processor and renames.
        specs: ``front_layers`` from config; None or empty returns *ds*.
        owner: The var_key, for errors, staging and logs.
        read_masks: Reads the store's own front masks, to seed the frequency
            with the days before *ds* begins. None starts it cold: its first
            days are NaN until half a window has been assessed.
    """
    for name, spec in (specs or {}).items():
        needed = [spec.source] + ([spec.confidence.var] if spec.confidence else [])
        missing = [v for v in needed if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"[{owner}] front_layers.{name} reads {missing}, which the dataset "
                f"does not hold. Variables: {sorted(map(str, ds.data_vars))}."
            )
        dims = set(map(str, ds[spec.source].dims))
        if dims != {"time", "lat", "lon"}:
            raise ValueError(
                f"[{owner}] front_layers.{name} needs '{spec.source}' over exactly "
                f"time/lat/lon; it has {tuple(map(str, ds[spec.source].dims))}."
            )
        layers = _layer(ds, name, spec, owner, read_masks)
        for var in spec.output_names(name):
            ds[var] = layers[var]
    return ds


def recompute_following_frequency(
    specs: dict[str, FrontLayerSpec],
    read_masks: MaskReader,
    after: pd.Timestamp,
) -> Optional[xr.Dataset]:
    """
    Frequencies for the stored days just after a rewritten window.

    A day's frequency reads the masks of the window before it, so rewriting
    days up to *after* (REP replacing NRT, a re-conversion) leaves the
    frequency of the following ``max(frequency_days) - 1`` days computed from
    masks that no longer exist. This recomputes them from the store's masks,
    now including the rewritten ones.

    The persistent distance, when declared, is computed from the frequency,
    so it is recomputed with it.

    Returns a dataset holding only those variables for those days, to be
    written back over them, or None when no stored day follows *after*.
    """
    out = {}
    for name, spec in specs.items():
        span = max(spec.frequency_days) - 1
        stored = read_masks(
            spec.mask_name(name),
            after - pd.Timedelta(days=span - 1),  # type: ignore[arg-type]
            after + pd.Timedelta(days=span),  # type: ignore[arg-type]
        )
        if stored is None:
            continue
        stored = stored.sortby("time")
        days = pd.DatetimeIndex(stored["time"].values)
        if not (days > after).any():
            continue
        lat = np.asarray(stored["lat"].values, dtype="float64")
        lon = np.asarray(stored["lon"].values, dtype="float64")
        rolling = RollingFrequency(
            spec.frequency_days, lat, lon, spec.frequency_radius_km
        )
        freqs: dict[str, list] = {v: [] for v in spec.freq_names(name)}
        kept = []
        for day, mask in zip(days, stored.values):
            rolling.push(day, mask)
            # A day with no finite cell has no stored mask: the reader pads a
            # file written before the layer was declared, and such a file has
            # no frequency to bring up to date.
            if day > after and np.isfinite(mask).any():
                kept.append(day)
                for n, var in zip(spec.frequency_days, spec.freq_names(name)):
                    freqs[var].append(rolling.frequency(n))
        if not kept:
            continue
        _with_pdist(freqs, spec, name, lat, lon, _SerialMap)
        for var, arrays in freqs.items():
            out[var] = xr.DataArray(
                np.stack(arrays),
                dims=("time", "lat", "lon"),
                coords={"time": kept, "lat": stored["lat"], "lon": stored["lon"]},
            )
    return xr.Dataset(out) if out else None
