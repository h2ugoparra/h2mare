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

_LAST_INSTANT_OF_DAY = pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def normalize_date(date: DateLike) -> pd.Timestamp:
    """
    Normalize a single date to a Timestamp at midnight.

    Raises:
        ValueError: If *date* is None, NaT, or anything else pandas reads as
            missing. ``pd.Timestamp`` returns NaT for those, and NaT has no
            ``.normalize()``, so unchecked it surfaces as ``AttributeError:
            'NaTType' object has no attribute 'normalize'`` — which names
            neither the argument nor the caller's mistake.
    """
    ts = pd.Timestamp(date)
    if ts is pd.NaT:
        raise ValueError(f"Not a usable date: {date!r}")
    return cast(pd.Timestamp, ts).normalize()


def end_of_day(date: DateLike) -> pd.Timestamp:
    """
    Last representable instant of *date*'s calendar day.

    One nanosecond short of the next midnight — pandas' datetime64[ns]
    resolution, so nothing can fall between this and the following day.

    Turns a date-level upper bound into one that covers the whole day. Every
    date the pipeline passes around is a midnight-stamped ``Timestamp``, which
    on a sub-daily axis names that day's *first* step: used verbatim as an
    inclusive end bound it keeps one step of the final day and drops the other
    23, whether the bound is slicing a store or being sent to a provider.
    """
    return normalize_date(date) + _LAST_INSTANT_OF_DAY


def normalize_dates(dates: DateLike | Sequence[DateLike]) -> list[pd.Timestamp]:
    """
    Normalize one date or a sequence of dates to a list of midnight Timestamps.

    Always returns a list, so callers accepting "date or dates" don't need to
    re-check what came back (the old scalar-or-list return forced isinstance
    guards at every such call site).
    """
    # Each element goes through normalize_date rather than being normalized
    # inline, so one unusable entry in a list is reported the same way as a
    # lone one instead of raising AttributeError on NaT.
    if isinstance(dates, (list, tuple)):
        return [normalize_date(d) for d in dates]
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
