"""WIG20 OHLCV validation against an explicit historical-session window.

The index's published dates supply the *observed* session reference; the
configured non-trading days and weekends are excluded. This is not a
replacement for a separately verified GPW exchange calendar.
SQL remains in marketsim.database.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable, Sequence

import pandas as pd

from marketsim.database.ikaros import ikaros_queries

PRICE_QUANTUM = Decimal("0.01")


def _dates(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "date" not in result:
        raise ValueError(f"{label}: missing date column")
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError(f"{label}: null session dates")
    # Normalize to date to detect duplicate sessions even when timestamps differ.
    result["date"] = result["date"].dt.date
    return result


def _preview(days: Iterable[date], *, limit: int = 5) -> str:
    values = sorted(set(days))
    listed = ", ".join(day.isoformat() for day in values[:limit])
    if len(values) > limit:
        listed += f" (+{len(values) - limit} more)"
    return listed


def index_session_history(
    frame: pd.DataFrame,
    *,
    cutoff: date,
    sessions: int,
    non_trading_dates: Iterable[date] = (),
) -> pd.DataFrame:
    """Return the exact recent index observations defining required sessions.

    Extra weekend/known-holiday observations are ignored. Duplicate *valid*
    sessions and insufficient real observations are fatal, never silently fixed.
    """
    if sessions < 2:
        raise ValueError("At least two lookback sessions required")
    if frame.empty:
        raise ValueError(f"WIG20: no index observations through {cutoff}")
    result = _dates(frame, label="WIG20 index")
    holidays = set(non_trading_dates)
    is_session = result["date"].map(lambda d: d <= cutoff and d.weekday() < 5 and d not in holidays)
    excluded = set(result.loc[~is_session & result["date"].map(lambda d: d <= cutoff), "date"])
    result = result.loc[is_session].copy()
    recent_dates = sorted(set(result["date"]))[-sessions:]
    if len(recent_dates) != sessions:
        raise ValueError(
            f"WIG20 index: {len(recent_dates)} valid sessions through {cutoff}; need {sessions}"
        )
    result = result.loc[result["date"].isin(recent_dates)].copy()
    duplicates = result.loc[result.duplicated("date", keep=False), "date"].tolist()
    if duplicates:
        raise ValueError(f"WIG20 index: duplicate valid sessions: {_preview(duplicates)}")
    result = result.sort_values("date").reset_index(drop=True)
    result.attrs["excluded_non_trading_dates"] = tuple(
        sorted(day for day in excluded if day >= result["date"].iloc[0])
    )
    if result["date"].iloc[-1] != cutoff:
        raise ValueError(
            f"WIG20 index: last valid session {result['date'].iloc[-1]} (target {cutoff})"
        )
    return result


def align_stock_history(
    frame: pd.DataFrame,
    *,
    symbol: str,
    required_sessions: Sequence[date],
    non_trading_dates: Iterable[date] = (),
) -> tuple[pd.DataFrame, tuple[date, ...]]:
    """Validate, sort and select every requested stock session, without filling.

    Returns excluded non-trading observations for reporting. A missing valid
    session, duplicate valid date or unexpected weekday not present in WIG20
    index history rejects the candidate cutoff.
    """
    if not required_sessions:
        raise ValueError("required_sessions cannot be empty")
    expected = set(required_sessions)
    if len(expected) != len(required_sessions):
        raise ValueError("Required index sessions contain duplicates")
    result = _dates(frame, label=symbol)
    holidays = set(non_trading_dates)
    nontrading = result["date"].map(lambda d: d.weekday() >= 5 or d in holidays)
    ignored = tuple(sorted(set(result.loc[nontrading, "date"])))
    result = result.loc[~nontrading].copy()

    observed = set(result["date"])
    duplicates = result.loc[result.duplicated("date", keep=False), "date"].tolist()
    missing = expected - observed
    extra = observed - expected
    issues = []
    if missing:
        issues.append(f"missing {len(missing)} required sessions: {_preview(missing)}")
    if duplicates:
        issues.append(f"duplicate valid sessions: {_preview(duplicates)}")
    if extra:
        issues.append(f"unexpected weekdays absent from WIG20 index: {_preview(extra)}")
    if issues:
        last = max(observed) if observed else None
        raise ValueError(f"{symbol}: {'; '.join(issues)}; last observed {last}")
    return result.sort_values("date").reset_index(drop=True), ignored


def futures_history(conn, contract_id, cutoff, sessions=250):
    frame = ikaros_queries.fetch_derivative_history(conn, contract_id, cutoff, sessions)

    if frame.empty:
        raise ValueError(f"No historical prices for futures contract {contract_id}")

    frame["date"] = pd.to_datetime(frame["date"])

    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].map(to_price)

    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")

    return frame.sort_values("date").reset_index(drop=True)


def to_price(value) -> Decimal:
    """Convert source values to Daedalus price precision."""
    if isinstance(value, Decimal):
        result = value
    else:
        result = Decimal(str(value))

    if not result.is_finite() or result <= 0:
        raise ValueError(f"Invalid price: {value!r}")

    return result.quantize(
        PRICE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )
