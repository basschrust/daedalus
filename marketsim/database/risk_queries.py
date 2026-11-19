"""Database operations for VaR calibrations and EoD Greeks.

No market/risk module should open a cursor or contain SQL.  Both supported
backends use the existing repository.connection and the same table layout.
This module does not create migrations or alter the simulation schema.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any


def _connection(repository: Any) -> Any:
    return repository.connection


def _snowflake(repository: Any) -> bool:
    module = type(_connection(repository)).__module__.lower()
    if "snowflake" in module:
        return True
    if "duckdb" in module:
        return False
    raise TypeError(f"Unsupported calibration backend: {type(_connection(repository))!r}")


def _placeholder(repository: Any) -> str:
    return "%s" if _snowflake(repository) else "?"


def ensure_daily_calibration_tables(repository: Any) -> None:
    """Ensure the three EXISTING calibration tables are present (no migration)."""
    cursor = _connection(repository).cursor()
    try:
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS risk_l2_instrument_statistics ("
            "calibration_id VARCHAR, symbol VARCHAR, "
            "mean_daily_return DOUBLE, daily_sigma DOUBLE)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS risk_l2_pair_statistics ("
            "calibration_id VARCHAR, symbol_1 VARCHAR, "
            "symbol_2 VARCHAR, covariance_daily DOUBLE)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS risk_calibration_history ("
            "day INTEGER, calibration_id VARCHAR, as_of_date DATE, "
            "first_session DATE, observations INTEGER, "
            "lookback_sessions INTEGER, source VARCHAR)"
        )
    finally:
        cursor.close()


def publish_daily_calibration(
    repository: Any,
    calibration_id: str,
    instruments: Sequence[tuple],
    pairs: Sequence[tuple],
    *,
    day: int,
    as_of_date: Any,
    first_session: Any,
    observations: int,
    lookback_sessions: int,
    source: str,
) -> None:
    """Publish immutable risk data in one backend-managed transaction."""
    mark = _placeholder(repository)
    cursor = _connection(repository).cursor()
    try:
        cursor.execute(
            f"SELECT day FROM risk_calibration_history WHERE calibration_id = {mark}",
            (calibration_id,),
        )
        existing = cursor.fetchall()
        if existing:
            if len(existing) != 1:
                raise ValueError(f"Duplicate calibration metadata: {calibration_id}")
            return
        cursor.execute("BEGIN TRANSACTION")
        try:
            cursor.executemany(
                "INSERT INTO risk_l2_instrument_statistics "
                "(calibration_id, symbol, mean_daily_return, daily_sigma) "
                f"VALUES ({', '.join([mark] * 4)})",
                list(instruments),
            )
            if pairs:
                cursor.executemany(
                    "INSERT INTO risk_l2_pair_statistics "
                    "(calibration_id, symbol_1, symbol_2, covariance_daily) "
                    f"VALUES ({', '.join([mark] * 4)})",
                    list(pairs),
                )
            cursor.execute(
                "INSERT INTO risk_calibration_history "
                "(day, calibration_id, as_of_date, first_session, "
                "observations, lookback_sessions, source) "
                f"VALUES ({', '.join([mark] * 7)})",
                (
                    day,
                    calibration_id,
                    as_of_date.isoformat() if isinstance(as_of_date, date) else as_of_date,
                    first_session.isoformat() if isinstance(first_session, date) else first_session,
                    observations,
                    lookback_sessions,
                    source,
                ),
            )
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    finally:
        cursor.close()


def fetch_source_risk_statistics(connection: Any, calibration_id: str) -> tuple[list, list]:
    """Read the historical L2 seed from Ikaros/Snowflake."""
    with connection.cursor() as cur:
        cur.execute(
            "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
            "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS "
            "WHERE CALIBRATION_ID = %s",
            (calibration_id,),
        )
        instruments = cur.fetchall()
        cur.execute(
            "SELECT SYMBOL_1, SYMBOL_2, COVARIANCE_DAILY "
            "FROM DAEDALUS.MART.L2_PAIR_STATISTICS WHERE CALIBRATION_ID = %s",
            (calibration_id,),
        )
        pairs = cur.fetchall()
    return instruments, pairs


def load_parametric_statistics(
    repository: Any,
    calibration_id: str,
    *,
    source_connection: Any = None,
) -> tuple[list, list]:
    """Replicate the existing parametric-calibration selection and caching."""
    conn = _connection(repository)
    if not _snowflake(repository):
        count = conn.execute(
            "SELECT count(*) FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
            [calibration_id],
        ).fetchone()[0]
        if count == 0:
            if source_connection is None:
                raise ValueError(
                    "DuckDB risk calibration absent; Snowflake init connection required"
                )
            instruments, pairs = fetch_source_risk_statistics(source_connection, calibration_id)
            if not instruments:
                raise ValueError(f"Missing calibration {calibration_id}")
            conn.execute("BEGIN TRANSACTION")
            try:
                conn.executemany(
                    "INSERT INTO risk_l2_instrument_statistics VALUES (?, ?, ?, ?)",
                    [(calibration_id, *row) for row in instruments],
                )
                if pairs:
                    conn.executemany(
                        "INSERT INTO risk_l2_pair_statistics VALUES (?, ?, ?, ?)",
                        [(calibration_id, *row) for row in pairs],
                    )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        instruments = conn.execute(
            "SELECT symbol, mean_daily_return, daily_sigma "
            "FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
            [calibration_id],
        ).fetchall()
        pairs = conn.execute(
            "SELECT symbol_1, symbol_2, covariance_daily "
            "FROM risk_l2_pair_statistics WHERE calibration_id = ?",
            [calibration_id],
        ).fetchall()
    else:
        instruments, pairs = fetch_source_risk_statistics(conn, calibration_id)
    return instruments, pairs


def fetch_source_greek_statistics(calibration_id: str) -> list[tuple]:
    """Fetch historical underlying volatilities without exposing connections to risk."""
    from marketsim.database.ikaros_queries import connect

    remote = connect()
    try:
        with remote.cursor() as cursor:
            cursor.execute(
                "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
                "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS WHERE CALIBRATION_ID = %s",
                (calibration_id,),
            )
            return list(cursor.fetchall())
    finally:
        remote.close()


def import_greek_calibration(
    repository: Any, calibration_id: str, data: Mapping[str, tuple]
) -> bool:
    """Return False for Snowflake; update the original local DuckDB snapshot."""
    if _snowflake(repository):
        return False
    conn = _connection(repository)
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute(
            "DELETE FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
            [calibration_id],
        )
        conn.executemany(
            "INSERT INTO risk_l2_instrument_statistics "
            "(calibration_id, symbol, mean_daily_return, daily_sigma) VALUES (?, ?, ?, ?)",
            [(calibration_id, sym, mean, sigma) for sym, (mean, sigma) in sorted(data.items())],
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return True


def uses_local_greek_calibration(repository: Any) -> bool:
    return not _snowflake(repository)


def load_greek_sigmas(repository: Any, calibration_id: str) -> list[tuple]:
    """Retrieve annualized volatility (from the same backend as before)."""
    snowflake = _snowflake(repository)
    sql = (
        "SELECT SYMBOL, ANNUALIZED_SIGMA FROM "
        "DAEDALUS.MART.L2_INSTRUMENT_STATISTICS WHERE CALIBRATION_ID = %s"
        if snowflake
        else "SELECT symbol, daily_sigma * SQRT(252.0) "
        "FROM risk_l2_instrument_statistics WHERE calibration_id = ?"
    )
    cur = _connection(repository).cursor()
    try:
        cur.execute(sql, (calibration_id,))
        return list(cur.fetchall())
    finally:
        cur.close()


def ensure_greek_tables(repository: Any) -> None:
    """Existing CREATE/ADD IF NOT EXISTS statements, relocated verbatim."""
    snowflake = _snowflake(repository)
    integer = "NUMBER(38,0)" if snowflake else "INTEGER"
    definitions = {
        "risk_greeks_short_rates": "calibration_id VARCHAR, currency VARCHAR, annual_rate DOUBLE",
        "risk_greeks_dividend_yields": "calibration_id VARCHAR, symbol VARCHAR, annual_yield DOUBLE",
        "risk_greeks_volatilities": "calibration_id VARCHAR, symbol VARCHAR, annual_sigma DOUBLE",
        "eod_option_greeks": (
            f"day {integer}, asset_id {integer}, underlying_id {integer}, "
            "model VARCHAR, delta DOUBLE, gamma DOUBLE, vega DOUBLE, theta DOUBLE, "
            "rho DOUBLE, multiplier DOUBLE, calibration_id VARCHAR"
        ),
        "eod_agent_option_greeks": (
            f"day {integer}, agent_id {integer}, asset_id {integer}, "
            f"underlying_id {integer}, position {integer}, delta DOUBLE, "
            "delta_equivalent DOUBLE, multiplier DOUBLE, calibration_id VARCHAR, "
            "gamma_equivalent DOUBLE, vega_equivalent DOUBLE, "
            "theta_equivalent DOUBLE, rho_equivalent DOUBLE"
        ),
    }
    cur = _connection(repository).cursor()
    try:
        for table, columns in definitions.items():
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns})")
        additions = {
            "eod_option_greeks": {"calibration_id": "VARCHAR"},
            "eod_agent_option_greeks": {
                "calibration_id": "VARCHAR",
                "gamma_equivalent": "DOUBLE",
                "vega_equivalent": "DOUBLE",
                "theta_equivalent": "DOUBLE",
                "rho_equivalent": "DOUBLE",
            },
        }
        for table, columns in additions.items():
            for name, dtype in columns.items():
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {dtype}")
    finally:
        cur.close()


def persist_greek_inputs(
    repository: Any,
    calibration_id: str,
    rates: Mapping[str, float],
    dividends: Mapping[str, float],
    sigmas: Mapping[str, float],
) -> None:
    mark = _placeholder(repository)
    values_by_table = (
        ("risk_greeks_short_rates", "currency", "annual_rate", rates),
        ("risk_greeks_dividend_yields", "symbol", "annual_yield", dividends),
        ("risk_greeks_volatilities", "symbol", "annual_sigma", sigmas),
    )
    cur = _connection(repository).cursor()
    try:
        for table, key_col, value_col, values in values_by_table:
            for key, value in sorted(values.items()):
                cur.execute(
                    f"SELECT {value_col} FROM {table} WHERE calibration_id={mark} "
                    f"AND {key_col}={mark}",
                    (calibration_id, key),
                )
                existing = cur.fetchall()
                if existing:
                    if len(existing) != 1 or not math.isclose(
                        float(existing[0][0]), value, rel_tol=1e-12, abs_tol=1e-12
                    ):
                        raise ValueError(f"Published calibration input changed: {table}/{key}")
                else:
                    cur.execute(
                        f"INSERT INTO {table} (calibration_id,{key_col},{value_col}) "
                        f"VALUES ({mark},{mark},{mark})",
                        (calibration_id, key, value),
                    )
    finally:
        cur.close()


def persist_eod_greek_results(
    repository: Any,
    day: int,
    calibration_id: str,
    options: Sequence[tuple],
    agents: Sequence[tuple],
) -> None:
    """Same delete/insert semantics as the original EoD Greeks calculation."""
    mark = _placeholder(repository)
    option_columns = (
        "day",
        "asset_id",
        "underlying_id",
        "model",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "multiplier",
        "calibration_id",
    )
    agent_columns = (
        "day",
        "agent_id",
        "asset_id",
        "underlying_id",
        "position",
        "delta",
        "delta_equivalent",
        "multiplier",
        "calibration_id",
        "gamma_equivalent",
        "vega_equivalent",
        "theta_equivalent",
        "rho_equivalent",
    )
    cur = _connection(repository).cursor()
    try:
        for row in options:
            cur.execute(
                f"DELETE FROM eod_option_greeks WHERE day={mark} "
                f"AND asset_id={mark} AND calibration_id={mark}",
                (day, row[1], calibration_id),
            )
            cur.execute(
                f"INSERT INTO eod_option_greeks ({','.join(option_columns)}) "
                f"VALUES ({','.join([mark] * len(option_columns))})",
                row,
            )
        for row in agents:
            cur.execute(
                f"DELETE FROM eod_agent_option_greeks WHERE day={mark} "
                f"AND agent_id={mark} AND asset_id={mark} AND calibration_id={mark}",
                (day, row[1], row[2], calibration_id),
            )
            cur.execute(
                f"INSERT INTO eod_agent_option_greeks ({','.join(agent_columns)}) "
                f"VALUES ({','.join([mark] * len(agent_columns))})",
                row,
            )
    finally:
        cur.close()
