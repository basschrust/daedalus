from __future__ import annotations

from datetime import date

import pandas as pd

from marketsim.database import ikaros_queries


def discover_stocks(conn, cutoff: date) -> pd.DataFrame:
    """Discover stock symbols with prices available at cutoff."""
    return ikaros_queries.fetch_available_stock_symbols(conn, cutoff)


def discover_derivatives(conn, cutoff: date) -> pd.DataFrame:
    """Discover registered contracts that have not expired."""
    return ikaros_queries.fetch_active_derivatives(conn, cutoff)


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
