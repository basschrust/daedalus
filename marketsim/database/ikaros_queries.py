"""Ikaros historical market-data queries (Snowflake source).

Only this database module knows Ikaros's schema, SQL syntax or credentials.
Scenario code asks for named datasets, never arbitrary SQL strings.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import snowflake.connector


def connect() -> Any:
    """Create the existing Ikaros Snowflake source connection."""
    kwargs = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "DAEDALUS_WH"),
        database=os.getenv("SNOWFLAKE_DATABASE", "DAEDALUS"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "MARKET_DATA"),
        authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
        enable_server_session_keep_alive_auto_detection=False,
    )
    if kwargs["authenticator"] == "snowflake":
        kwargs["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    if os.getenv("SNOWFLAKE_ROLE"):
        kwargs["role"] = os.environ["SNOWFLAKE_ROLE"]
    return snowflake.connector.connect(**kwargs)


def _query(conn: Any, sql: str, params: tuple | list = ()) -> pd.DataFrame:
    """Only DB-level helpers accept SQL; consumers receive DataFrames."""
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [item[0].lower() for item in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def fetch_wig20_constituents(conn: Any, as_of: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT SECURITY_SYMBOL AS SYMBOL, WEIGHT, VALID_FROM, VALID_TO, SOURCE
        FROM INDEX_CONSTITUENTS
        WHERE INDEX_SYMBOL = 'WIG20' AND VALID_FROM <= %s
          AND (VALID_TO IS NULL OR VALID_TO >= %s)
        ORDER BY SECURITY_SYMBOL
    """,
        (as_of, as_of),
    )


def fetch_candidate_dates(conn: Any, max_date: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT DISTINCT TO_DATE("DATE") AS SESSION_DATE
        FROM PRICES
        WHERE TO_DATE("DATE") <= %s
        ORDER BY SESSION_DATE DESC LIMIT 90
    """,
        (max_date,),
    )


def fetch_wig20_stock_history(conn: Any, symbol: str, cutoff: date, sessions: int) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE, "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
        FROM PRICES
        WHERE "SYMBOL" = %s AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC LIMIT %s
    """,
        (symbol, cutoff, sessions),
    )


def fetch_wig20_index_close(conn: Any, cutoff: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT "CLOSE" AS INDEX_CLOSE FROM INDEX_PRICES
        WHERE "INDEX_SYMBOL" = %s AND TO_DATE("DATE") = %s
    """,
        ("WIG20", cutoff),
    )


def fetch_wig20_index_history(conn: Any, cutoff: date, sessions: int) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE, "OPEN", "HIGH", "LOW", "CLOSE"
        FROM INDEX_PRICES
        WHERE "INDEX_SYMBOL" = %s AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC LIMIT %s
    """,
        ("WIG20", cutoff, sessions),
    )


def fetch_wig20_futures_contracts(conn: Any, cutoff: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT CONTRACT_ID, SYMBOL, UNDERLYING_SYMBOL, EXPIRY_DATE, MULTIPLIER
        FROM DERIVATIVE_CONTRACTS
        WHERE INSTRUMENT_TYPE = 'FUTURE'
          AND UNDERLYING_SYMBOL = 'WIG20' AND EXPIRY_DATE > %s
        ORDER BY EXPIRY_DATE, CONTRACT_ID
    """,
        (cutoff,),
    )


def fetch_derivative_history(
    conn: Any, contract_id: int, cutoff: date, sessions: int
) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE, "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
        FROM DERIVATIVE_PRICES
        WHERE CONTRACT_ID = %s AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC LIMIT %s
    """,
        (contract_id, cutoff, sessions),
    )


def fetch_wig20_option_contracts(conn: Any, cutoff: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT * FROM DERIVATIVE_CONTRACTS
        WHERE UPPER(INSTRUMENT_TYPE) IN ('OPTION', 'INDEX_OPTION')
          AND UPPER(UNDERLYING_SYMBOL) = 'WIG20'
          AND EXPIRY_DATE > %s
        ORDER BY EXPIRY_DATE, CONTRACT_ID
    """,
        (cutoff,),
    )


def fetch_gpw_stock_histories(
    conn: Any, symbols: list[str], cutoff: date, sessions: int
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    placeholders = ", ".join(["%s"] * len(symbols))
    return _query(
        conn,
        f"""
        SELECT SYMBOL, DATE, OPEN, HIGH, LOW, CLOSE, VOLUME FROM (
            SELECT "SYMBOL" AS SYMBOL, TO_DATE("DATE") AS DATE,
                "OPEN" AS OPEN, "HIGH" AS HIGH, "LOW" AS LOW,
                "CLOSE" AS CLOSE, "VOLUME" AS VOLUME,
                ROW_NUMBER() OVER (
                    PARTITION BY "SYMBOL" ORDER BY TO_DATE("DATE") DESC
                ) AS RN
            FROM PRICES WHERE "SYMBOL" IN ({placeholders})
              AND TO_DATE("DATE") <= %s
        ) WHERE RN <= %s ORDER BY SYMBOL, DATE
    """,
        (*symbols, cutoff, sessions),
    )


def fetch_gpw_index_history(conn: Any, symbol: str, cutoff: date, sessions: int) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE, "OPEN", "HIGH", "LOW", "CLOSE"
        FROM INDEX_PRICES WHERE INDEX_SYMBOL = %s AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC LIMIT %s
    """,
        (symbol, cutoff, sessions),
    )


def fetch_option_metadata_columns(conn: Any) -> set[str]:
    """Look up real source columns without leaking a schema-probe query."""
    return set(_query(conn, "SELECT * FROM DERIVATIVE_CONTRACTS WHERE 1 = 0").columns)


def fetch_option_contract_details(
    conn: Any, contract_ids: list[int], strike_column: str, side_column: str
) -> pd.DataFrame:
    allowed = {"strike_price", "strike", "option_type", "option_side", "call_put"}
    if strike_column not in allowed or side_column not in allowed:
        raise ValueError("Unsupported Ikaros option metadata column")
    if not contract_ids:
        return pd.DataFrame(columns=["contract_id", "strike", "side"])
    placeholders = ", ".join(["%s"] * len(contract_ids))
    return _query(
        conn,
        f"""
        SELECT CONTRACT_ID, "{strike_column.upper()}" AS STRIKE,
               "{side_column.upper()}" AS SIDE
        FROM DERIVATIVE_CONTRACTS WHERE CONTRACT_ID IN ({placeholders})
    """,
        tuple(contract_ids),
    )


def fetch_available_stock_symbols(conn: Any, cutoff: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT DISTINCT "SYMBOL" AS SYMBOL FROM PRICES
        WHERE TO_DATE("DATE") = %s ORDER BY SYMBOL
    """,
        (cutoff,),
    )


def fetch_active_derivatives(conn: Any, cutoff: date) -> pd.DataFrame:
    return _query(
        conn,
        """
        SELECT CONTRACT_ID, SYMBOL, INSTRUMENT_TYPE,
               UNDERLYING_SYMBOL, EXPIRY_DATE, MULTIPLIER
        FROM DERIVATIVE_CONTRACTS WHERE EXPIRY_DATE > %s
        ORDER BY INSTRUMENT_TYPE, UNDERLYING_SYMBOL, EXPIRY_DATE, CONTRACT_ID
    """,
        (cutoff,),
    )


def fetch_source_regime_statistics(conn: Any, calibration_id: str) -> tuple[list, list]:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
            "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS "
            "WHERE CALIBRATION_ID = %s",
            (calibration_id,),
        )
        instruments = cursor.fetchall()
        cursor.execute(
            "SELECT SYMBOL_1, SYMBOL_2, COVARIANCE_DAILY "
            "FROM DAEDALUS.MART.L2_PAIR_STATISTICS WHERE CALIBRATION_ID = %s",
            (calibration_id,),
        )
        pairs = cursor.fetchall()
    return instruments, pairs


def fetch_legacy_historical_ohlcv(
    connection: Any, table: str, cutoff_date: str, lookback_days: int
) -> pd.DataFrame:
    """Legacy dynamic source table; retains the original caller contract."""
    return pd.read_sql(
        f"""
        SELECT SYMBOL, DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        FROM {table}
        WHERE DATE <= DATEADD(day, 0, %s)
          AND DATE > DATEADD(day, -%s, %s)
        ORDER BY SYMBOL, DATE
    """,
        connection,
        params=(cutoff_date, lookback_days, cutoff_date),
    )
