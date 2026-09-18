"""
Helpers for ZarrCatalog data coverage and store management.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from h2mare.storage.zarr_catalog import get_zarr_time_coverage
from h2mare.types import DateLike, DateRange, FilePeriod, to_datetime
from h2mare.utils.datetime_utils import normalize_date


def split_time_range(date_range: DateRange, split: FilePeriod) -> list[DateRange]:
    """
    Split date range into chunks.

    Args:
        date_range: Range to split
        split: Split strategy

    Returns:
        List of (start, end) tuples
    """
    chunks = []
    current = date_range.start

    while current <= date_range.end:
        if split == FilePeriod.MONTH:
            # End of current month
            chunk_end = current + pd.offsets.MonthEnd(0)
        elif split == FilePeriod.YEAR:
            # End of current year
            chunk_end = pd.Timestamp(year=current.year, month=12, day=31)
        else:
            raise ValueError("Invalid split value. Options are: 'month' or 'year'")

        # Don't exceed requested end
        chunk_end = min(chunk_end, date_range.end)

        chunks.append(DateRange(start=current, end=chunk_end))

        # Move to next period
        current = chunk_end + pd.Timedelta(days=1)

    return chunks


#: Why the last coverage lookup for a var_key failed, when it failed by raising
#: rather than by finding nothing. Read by :func:`resolve_date_range` to tell
#: "no store" apart from "store unreadable" in the error it raises.
#:
#: Every lookup either sets or clears its own key, so an entry always describes
#: the most recent attempt for that var_key and cannot go stale — a store that
#: fails once and reads cleanly later leaves nothing behind. Callers other than
#: get_store_coverage must not write to it.
_UNREADABLE_STORES: dict[str, Exception] = {}


def get_store_coverage(var_key: str) -> Optional[DateRange]:
    """
    Get time coverage of existing data in store.

    An unreadable store is not an empty one. Both return ``None`` here — the
    many callers that treat ``None`` as "nothing stored yet" depend on that —
    but the reason is recorded in :data:`_UNREADABLE_STORES` so
    :func:`resolve_date_range` can say which it was. Without that, a locked or
    damaged store presents as "no existing data found", sending the reader to
    look for missing files while the real cause sits on a warning line above.

    Returns:
        DateRange of stored data, or None if no data exists or it is unreadable.
    """
    try:
        coverage = get_zarr_time_coverage(var_key)
    except Exception as e:
        logger.warning(f"Could not read store coverage for {var_key}: {e}")
        _UNREADABLE_STORES[var_key] = e
        return None

    _UNREADABLE_STORES.pop(var_key, None)

    if coverage is None:
        logger.debug(f"No existing data found for {var_key}")
        return None

    return DateRange(start=coverage.start, end=coverage.end)


def resolve_date_range(
    var_key: str,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
) -> Optional[DateRange]:
    """
    Resolve a date range for download or processing.

    When *start* or *end* are omitted they are inferred from the store:
    start = store_end + 1 day, end = today.  If the store is already
    up to date (inferred start > end) ``None`` is returned so callers
    can skip gracefully without treating it as an error.

    When both dates are supplied explicitly and start > end a
    ``ValueError`` is raised (caller error).

    Lives beside :func:`get_store_coverage` rather than in ``utils`` because
    the store lookup is its whole purpose: importing it the other way round
    made a leaf package depend on ``storage`` and closed an import cycle.

    Args:
        var_key: Variable key used to look up the store.
        start: Explicit start date, or ``None`` to infer.
        end: Explicit end date, or ``None`` to infer.

    Returns:
        Resolved :class:`DateRange`, or ``None`` if the store is already
        up to date.

    Raises:
        ValueError: If no store exists and dates cannot be inferred, or
            if explicitly supplied dates produce start > end.
    """
    # Separate locals from the DateLike parameters: past this point the values
    # are always concrete datetimes, and reusing the parameter names would keep
    # the broader declared type (which allows str) for the comparisons below.
    start_ts = normalize_date(start) if start else None
    end_ts = normalize_date(end) if end else None
    inferred = start_ts is None or end_ts is None

    if inferred:
        store_coverage = get_store_coverage(var_key)

        if store_coverage is None:
            read_error = _UNREADABLE_STORES.get(var_key)
            if read_error is not None:
                # The store exists and could not be read. Saying "no existing
                # data" here sent the reader looking for missing files.
                raise ValueError(
                    f"Could not read the store for '{var_key}' to infer dates: "
                    f"{type(read_error).__name__}: {read_error}. Fix the store, "
                    f"or provide start and end dates explicitly to skip the "
                    f"lookup."
                ) from read_error
            raise ValueError(
                f"No existing data found for '{var_key}'. "
                f"Please provide start and end dates explicitly."
            )

        if start_ts is None:
            # to_datetime() is a pass-through for datetime inputs; it is here to
            # pin the result to datetime, which the pandas stubs otherwise widen
            # with NaT (unreachable — store_coverage.end is always a real date).
            start_ts = to_datetime(store_coverage.end + pd.Timedelta(days=1))
        if end_ts is None:
            end_ts = pd.Timestamp.now().normalize()

        logger.info(
            f"Date range in store: {store_coverage.start.date()} -> {store_coverage.end.date()}"
        )

    assert start_ts is not None and end_ts is not None
    if start_ts > end_ts:
        if inferred:
            return None
        raise ValueError(
            f"Invalid date range: start ({start_ts.date()}) > end ({end_ts.date()})"
        )

    return DateRange(start=start_ts, end=end_ts)
