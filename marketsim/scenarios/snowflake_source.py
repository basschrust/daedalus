
from __future__ import annotations

import os
from contextlib import closing
from datetime import date

import pandas as pd
import snowflake.connector


OHLCV = ("open", "high", "low", "close", "volume")


def connect():
    """Read-only data source for scenario preparation."""
    settings = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "warehouse": os.getenv(
            "SNOWFLAKE_WAREHOUSE", "DAEDALUS_WH"
        ),
        "database": os.getenv(
            "SNOWFLAKE_DATABASE", "DAEDALUS"
        ),
        "schema": os.getenv(
            "SNOWFLAKE_SCHEMA", "MARKET_DATA"
        ),
        "authenticator": os.getenv(
            "SNOWFLAKE_AUTHENTICATOR", "snowflake"
        ),
    }

    if settings["authenticator"] == "snowflake":
        settings["password"] = os.environ[
            "SNOWFLAKE_PASSWORD"
        ]

    return snowflake.connector.connect(**settings)


def query_frame(connection, sql, params=()):
    with closing(connection.cursor()) as cursor:
        cursor.execute(sql, params)
        columns = [col[0].lower() for col in cursor.description]
        return pd.DataFrame(
            cursor.fetchall(),
            columns=columns,
        )


def latest_complete_cutoff(
    connection,
    symbols: list[str],
    lookback_sessions: int = 250,
    max_date: date | None = None,
) -> date:
    """
    Find the latest common valid trading session.

    Requires a full historical lookback for every symbol.
    Excludes today's session by default, since it may
    still be incomplete.
    """
    if len(symbols) != 20 or len(set(symbols)) != 20:
        raise ValueError(
            "Expected exactly 20 distinct WIG20 stocks"
        )

    if lookback_sessions <= 0:
        raise ValueError("lookback_sessions must be positive")

    placeholders = ", ".join(["%s"] * len(symbols))

    sql = f"""
        WITH valid_sessions AS (
            SELECT
                "SYMBOL" AS symbol,
                TO_DATE("DATE") AS session_date
            FROM "PRICES"
            WHERE "SYMBOL" IN ({placeholders})
              AND TO_DATE("DATE") <=
                  COALESCE(%s::DATE, DATEADD(day, -1, CURRENT_DATE()))
              AND "OPEN" > 0
              AND "HIGH" > 0
              AND "LOW" > 0
              AND "CLOSE" > 0
              AND "VOLUME" >= 0
            GROUP BY "SYMBOL", TO_DATE("DATE")
            HAVING COUNT(*) = 1
               AND COUNT("OPEN") = 1
               AND COUNT("HIGH") = 1
               AND COUNT("LOW") = 1
               AND COUNT("CLOSE") = 1
               AND COUNT("VOLUME") = 1
        ),
        ranked AS (
            SELECT
                symbol,
                session_date,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY session_date
                ) AS session_number
            FROM valid_sessions
        ),
        eligible AS (
            SELECT
                symbol,
                session_date
            FROM ranked
            WHERE session_number >= %s
        )
        SELECT session_date
        FROM eligible
        GROUP BY session_date
        HAVING COUNT(DISTINCT symbol) = %s
        ORDER BY session_date DESC
        LIMIT 1
    """

    params = [
        *symbols,
        max_date,
        lookback_sessions,
        len(symbols),
    ]

    frame = query_frame(connection, sql, params)

    if frame.empty:
        raise ValueError(
            "No common complete WIG20 session with "
            f"{lookback_sessions} historical observations"
        )

    return frame.iloc[0]["session_date"]


def fetch_stock_history(
    connection,
    symbol: str,
    cutoff: date,
    lookback_sessions: int = 250,
) -> pd.DataFrame:
    sql = """
        SELECT
            TO_DATE("DATE") AS date,
            "OPEN" AS open,
            "HIGH" AS high,
            "LOW" AS low,
            "CLOSE" AS close,
            "VOLUME" AS volume
        FROM "PRICES"
        WHERE "SYMBOL" = %s
          AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC
        LIMIT %s
    """

    frame = query_frame(
        connection,
        sql,
        (symbol, cutoff, lookback_sessions),
    )

    if len(frame) != lookback_sessions:
        raise ValueError(
            f"{symbol}: expected {lookback_sessions} "
            f"sessions, got {len(frame)}"
        )

    frame["date"] = pd.to_datetime(frame["date"])

    if frame["date"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate sessions")

    for column in OHLCV:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        )

    if frame[list(OHLCV)].isna().any().any():
        raise ValueError(f"{symbol}: missing OHLCV values")

    if not (
        frame[["open", "high", "low", "close"]] > 0
    ).all().all():
        raise ValueError(f"{symbol}: invalid prices")

    if not (frame["volume"] >= 0).all():
        raise ValueError(f"{symbol}: negative volume")

    frame = frame.sort_values("date").set_index("date")

    if frame.index[-1].date() != cutoff:
        raise ValueError(
            f"{symbol}: history does not end at {cutoff}"
        )

    return frame


def fetch_index_constituents(
    connection,
    index_symbol: str,
    as_of: date,
) -> pd.DataFrame:
    """Load the index composition valid on a given date."""

    sql = """
        SELECT
            SECURITY_SYMBOL AS symbol,
            WEIGHT AS weight,
            SOURCE AS source
        FROM INDEX_CONSTITUENTS
        WHERE INDEX_SYMBOL = %s
          AND VALID_FROM <= %s
          AND (
              VALID_TO IS NULL
              OR VALID_TO >= %s
          )
        ORDER BY SECURITY_SYMBOL
    """

    frame = query_frame(
        connection,
        sql,
        (index_symbol, as_of, as_of),
    )

    if len(frame) != 20:
        raise ValueError(
            f"{index_symbol}: expected 20 constituents "
            f"on {as_of}, found {len(frame)}"
        )

    if frame["symbol"].isna().any():
        raise ValueError("Missing constituent symbol")

    if frame["symbol"].duplicated().any():
        raise ValueError(
            f"{index_symbol}: duplicate constituents"
        )

    frame["weight"] = pd.to_numeric(
        frame["weight"], errors="raise"
    )

    if (
        frame["weight"].isna().any()
        or not frame["weight"].map(math.isfinite).all()
        or (frame["weight"] <= 0).any()
    ):
        raise ValueError(
            f"{index_symbol}: invalid constituent weights"
        )

    return frame

