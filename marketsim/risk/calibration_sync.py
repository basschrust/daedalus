"""Synchronize an immutable L2 calibration into the simulation's local DuckDB.

Snowflake remains the source of truth. No fallback or synthetic volatilities.
"""

from __future__ import annotations

import math


def _symbol(market):
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).upper()


def sync_greeks_calibration(simulator, config: dict) -> None:
    """Load the configured calibration before the first simulation tick."""
    greeks = config.get("risk_greeks") or {}
    if not greeks.get("enabled", False):
        return
    calibration_id = str(greeks.get("calibration_id") or "").strip()
    if not calibration_id:
        raise ValueError("risk_greeks.calibration_id is mandatory")

    required = {
        _symbol(m.underlying) for m in simulator.markets.values() if m.instrument_class == "option"
    }
    if not required:
        return
    if "" in required:
        raise ValueError("Option underlying has no symbol/short_name")

    local = simulator.repository.connection
    # Snowflake-backed simulations query L2 directly in eod_greeks; only
    # local DuckDB requires the synchronization.
    if "snowflake" in type(local).__module__.lower():
        return
    if "duckdb" not in type(local).__module__.lower():
        raise TypeError("Unsupported repository connection for calibration sync")

    from marketsim.scenarios.wig20 import connect

    remote = connect()
    try:
        with remote.cursor() as cursor:
            cursor.execute(
                "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
                "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS "
                "WHERE CALIBRATION_ID = %s",
                (calibration_id,),
            )
            rows = cursor.fetchall()
    finally:
        remote.close()

    clean = {}
    for symbol, mean, sigma in rows:
        symbol = str(symbol).upper()
        if symbol in clean:
            raise ValueError(f"Duplicate L2 calibration symbol: {symbol}")
        if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0:
            raise ValueError(f"Invalid daily sigma for {symbol} in {calibration_id}")
        clean[symbol] = (float(mean) if mean is not None else None, float(sigma))

    missing = required - clean.keys()
    if missing:
        raise ValueError(
            f"Calibration {calibration_id} missing underlying volatilities in Snowflake: "
            f"{sorted(missing)}"
        )
    if not clean:
        raise ValueError(f"Calibration {calibration_id} has no L2 statistics in Snowflake")

    local.execute("BEGIN TRANSACTION")
    try:
        # The run database is a local snapshot. Replace only the selected
        # calibration, never unrelated calibration IDs.
        local.execute(
            "DELETE FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
            [calibration_id],
        )
        local.executemany(
            "INSERT INTO risk_l2_instrument_statistics "
            "(calibration_id, symbol, mean_daily_return, daily_sigma) VALUES (?, ?, ?, ?)",
            [(calibration_id, sym, mean, sigma) for sym, (mean, sigma) in sorted(clean.items())],
        )
        local.execute("COMMIT")
    except Exception:
        local.execute("ROLLBACK")
        raise
    print(
        f"Synchronized L2 calibration {calibration_id}: "
        f"{len(clean)} instruments; required option underlyings: {sorted(required)}"
    )
