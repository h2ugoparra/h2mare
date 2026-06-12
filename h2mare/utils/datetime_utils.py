from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Sequence, cast, overload

import pandas as pd

# Re-exported for callers that import it from here; the single definition
# lives in h2mare.types (this module is imported by h2mare.utils.__init__,
# which types.py cannot depend on without a cycle).
from h2mare.types import to_datetime as to_datetime

if TYPE_CHECKING:
    from h2mare.types import DateLike


@overload
def normalize_date(date: DateLike) -> pd.Timestamp: ...
@overload
def normalize_date(date: Sequence[DateLike]) -> list[pd.Timestamp]: ...


def normalize_date(
    date: DateLike | Sequence[DateLike],
) -> pd.Timestamp | list[pd.Timestamp]:
    """Normalize date(s) to Timestamp(s) at midnight."""
    if isinstance(date, (list, tuple)):
        return [pd.to_datetime(d).normalize() for d in date]
    return pd.Timestamp(cast("DateLike", date)).normalize()


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
