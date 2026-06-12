from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Sequence, cast

import pandas as pd

# Re-exported for callers that import it from here; the single definition
# lives in h2mare.types (this module is imported by h2mare.utils.__init__,
# which types.py cannot depend on without a cycle).
from h2mare.types import to_datetime as to_datetime

if TYPE_CHECKING:
    from h2mare.types import DateLike


def normalize_date(date: DateLike) -> pd.Timestamp:
    """Normalize a single date to a Timestamp at midnight."""
    return pd.Timestamp(date).normalize()


def normalize_dates(dates: DateLike | Sequence[DateLike]) -> list[pd.Timestamp]:
    """
    Normalize one date or a sequence of dates to a list of midnight Timestamps.

    Always returns a list, so callers accepting "date or dates" don't need to
    re-check what came back (the old scalar-or-list return forced isinstance
    guards at every such call site).
    """
    if isinstance(dates, (list, tuple)):
        return [pd.Timestamp(d).normalize() for d in dates]
    return [normalize_date(cast("DateLike", dates))]


def more_than_one_year(a: pd.Timestamp, b: pd.Timestamp) -> bool:
    """Check if the difference between two dates is more than one year."""
    earlier, later = sorted([a, b])
    return later > earlier + pd.DateOffset(years=1)


def date_to_standard_string(d: DateLike) -> str:
    """Convert str | datetime | date into a standardized 'YYYY-MM-DD' string."""
    if isinstance(d, str):
        return pd.to_datetime(d).date().isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return pd.to_datetime(d).date().isoformat()
