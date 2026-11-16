"""Shared WIG20 scenario data utilities."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

PRICE_QUANTUM = Decimal("0.01")


def futures_history(conn, contract_id, cutoff, sessions=250):
    frame = query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE,
               "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
        FROM DERIVATIVE_PRICES
        WHERE CONTRACT_ID = %s
          AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC
        LIMIT %s
        """,
        (contract_id, cutoff, sessions),
    )

    if frame.empty:
        raise ValueError(f"No historical prices for futures contract {contract_id}")

    frame["date"] = pd.to_datetime(frame["date"])

    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].map(to_price)

    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")

    return frame.sort_values("date").reset_index(drop=True)


def query(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [item[0].lower() for item in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


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
