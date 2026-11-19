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

    if not simulator.repository.needs_local_greek_calibration():
        return
    rows = simulator.repository.fetch_source_greek_calibration(calibration_id)

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

    simulator.repository.import_greek_calibration(calibration_id, clean)
    print(
        f"Synchronized L2 calibration {calibration_id}: "
        f"{len(clean)} instruments; required option underlyings: {sorted(required)}"
    )
