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
) -> DateRange:
    """
    Resolve storage date range for download.

    Priority:
        1. Explicit arguments
        2. Default dates from __init__
        3. Latest date from store + 1 day to today

    Args:
        start_date: Explicit start date
        end_date: Explicit end date

    Returns:
        Resolved DateRange

    Raises:
        ValueError: If dates cannot be resolved
    """
    start = normalize_date(start) if start else None
    end = normalize_date(end) if end else None

    # If still None, try to infer from store
    if start is None or end is None:
        store_coverage = get_store_coverage(var_key)

        if store_coverage is None:
            raise ValueError(
                f"No existing data found for '{var_key}'. "
                f"Please provide start and end dates explicitly."
            )

        # Default: download from day after latest stored data to today
        if start is None:
            start = store_coverage.end + pd.Timedelta(days=1)

        if end is None:
            end = pd.Timestamp.now().normalize()

        logger.info(
            f"Date range in store: {store_coverage.start.date()} -> {store_coverage.end.date()}\n"
        )

    # Validate
    if start > end:
        raise ValueError(
            f"Invalid date range: start ({start.date()}) > end ({end.date()})"
        )

    return DateRange(start=start, end=end)
