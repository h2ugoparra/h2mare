"""Utility for resolving a download/processing date range from store coverage."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from h2mare.storage.coverage import get_store_coverage
from h2mare.types import DateLike, DateRange
from h2mare.utils.datetime_utils import normalize_date


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
    start = normalize_date(start) if start else None
    end = normalize_date(end) if end else None
    inferred = start is None or end is None

    if inferred:
        store_coverage = get_store_coverage(var_key)

        if store_coverage is None:
            raise ValueError(
                f"No existing data found for '{var_key}'. "
                f"Please provide start and end dates explicitly."
            )

        if start is None:
            start = store_coverage.end + pd.Timedelta(days=1)
        if end is None:
            end = pd.Timestamp.now().normalize()

        logger.info(
            f"Date range in store: {store_coverage.start.date()} -> {store_coverage.end.date()}"
        )

    if start > end:
        if inferred:
            logger.info(f"'{var_key}' is already up to date ({end.date()}) — skipping.")
            return None
        raise ValueError(
            f"Invalid date range: start ({start.date()}) > end ({end.date()})"
        )

    return DateRange(start=start, end=end)
