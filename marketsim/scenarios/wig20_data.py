"""WIG20 scenario OHLCV validation; SQL lives in database.ikaros_queries."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from marketsim.database import ikaros_queries

PRICE_QUANTUM = Decimal("0.01")


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
