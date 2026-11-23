"""Date-window WIG20 history queries shared by Snowflake and cached Ikaros.

Keep SQL in the database layer. The existing CachedIkarosConnection adapter
translates the named table and Snowflake's date/placeholders for DuckDB.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from marketsim.database.ikaros_queries import _query


def fetch_stock_window(
    conn: Any,
    *,
    symbol: str,
    first_session: date,
    cutoff: date,
) -> pd.DataFrame:
    """Fetch *all* candidate candles, including duplicates, in a date interval.

    Unlike LIMIT N, a date interval never drops a real session merely because
    the source contains extra or duplicate observations.
    """
    return _query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE, "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
        FROM PRICES
        WHERE "SYMBOL" = %s
          AND TO_DATE("DATE") >= %s AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") ASC
        """,
        (symbol, first_session, cutoff),
    )
