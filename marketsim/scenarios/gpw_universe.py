from __future__ import annotations

from datetime import date

import pandas as pd

from marketsim.scenarios.wig20 import query


def discover_stocks(conn, cutoff: date) -> pd.DataFrame:
    """Discover stock symbols with prices available at cutoff."""
    return query(
        conn,
        """
        SELECT DISTINCT "SYMBOL" AS SYMBOL
        FROM PRICES
        WHERE TO_DATE("DATE") = %s
        ORDER BY SYMBOL
        """,
        (cutoff,),
    )


def discover_derivatives(conn, cutoff: date) -> pd.DataFrame:
    """Discover registered contracts that have not expired."""
    return query(
        conn,
        """
        SELECT CONTRACT_ID, SYMBOL, INSTRUMENT_TYPE,
               UNDERLYING_SYMBOL, EXPIRY_DATE, MULTIPLIER
        FROM DERIVATIVE_CONTRACTS
        WHERE EXPIRY_DATE > %s
        ORDER BY INSTRUMENT_TYPE, UNDERLYING_SYMBOL,
                 EXPIRY_DATE, CONTRACT_ID
        """,
        (cutoff,),
    )


def universe_summary(
    stocks: pd.DataFrame,
    derivatives: pd.DataFrame,
) -> pd.DataFrame:
    rows = [{"instrument_type": "STOCK", "count": len(stocks)}]

    if not derivatives.empty:
        counts = derivatives.groupby("instrument_type").size()

        rows.extend(
            {"instrument_type": kind, "count": int(count)} for kind, count in counts.items()
        )

    return pd.DataFrame(rows)
