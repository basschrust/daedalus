"""Resolve option-Greeks volatility calibration for the simulation repository.

A local simulation first uses its prepared immutable calibration cache, then
its own historical VaR calibration. Snowflake is only a fallback or an explicit
choice; missing optional Snowflake dependencies must not produce tracebacks.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb

from marketsim.scenarios.prepared_cache import PreparedDatasetCache


class GreeksCalibrationSetupError(RuntimeError):
    """An actionable missing or unusable Greeks calibration source."""


def _symbol(market: object) -> str:
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).upper()


def sync_greeks_calibration(simulator: object, config: dict) -> None:
    """Reuse a local, real historical calibration before attempting Snowflake.

    Never write local statistics under a different, externally supplied ID.
    Instead switch the run to the immutable calibration ID from its history.
    """
    greeks = config.get("risk_greeks") or {}
    if not greeks.get("enabled", False):
        return

    source = str(greeks.get("source", "auto")).lower()
    if source not in {"auto", "local", "snowflake"}:
        raise GreeksCalibrationSetupError(
            "risk_greeks.source must be 'auto', 'local', or 'snowflake'."
        )

    requested_id = str(greeks.get("calibration_id") or "").strip()
    required = {
        _symbol(m.underlying) for m in simulator.markets.values() if m.instrument_class == "option"
    }
    if not required:
        return
    if "" in required:
        raise GreeksCalibrationSetupError("An option underlying has no market symbol.")

    if not simulator.repository.needs_local_greek_calibration():
        return

    cache = PreparedDatasetCache(root=Path(__file__).resolve().parents[2])
    rows = None
    effective_id = requested_id
    provenance = ""
    should_cache_remote = False

    # An explicitly pinned calibration is preferred when already available
    # locally. Do not fetch it from Snowflake merely because the cache is empty.
    if requested_id and source != "snowflake":
        try:
            rows = cache.get_greek_calibration(calibration_id=requested_id)
        except (OSError, duckdb.Error):
            # A locked shared cache must not prevent use of calibration data
            # already available in this simulation's own DuckDB database.
            rows = None
        if rows and required.issubset({str(row[0]).upper() for row in rows}):
            provenance = "cached L2 calibration"
        else:
            rows = None

    # The daily VaR calibration is calculated from actual aligned historical
    # OHLCV and already published in the simulation's local DuckDB.
    local_problem = ""
    if rows is None and source != "snowflake":
        daily = getattr(simulator, "daily_risk_calibration", None)
        if daily is None and getattr(simulator, "historical_ohlcv_by_symbol", None):
            # Also allow historical Greeks-only scenarios (without VaR).
            from marketsim.risk.daily_calibration import DailyRiskCalibration

            try:
                daily = DailyRiskCalibration(
                    simulator,
                    lookback_sessions=int(greeks.get("lookback_sessions", 250)),
                )
            except (TypeError, ValueError, KeyError) as exc:
                local_problem = str(exc).splitlines()[0]

        if daily is not None:
            candidate_id = daily.latest_calibration_id
            local_rows, _ = simulator.repository.load_parametric_calibration(candidate_id)
            if required.issubset({str(row[0]).upper() for row in local_rows}):
                rows = local_rows
                effective_id = candidate_id
                provenance = "local historical VaR calibration"
            else:
                local_problem = (
                    f"Local calibration {candidate_id} is missing "
                    f"{sorted(required - {str(row[0]).upper() for row in local_rows})}."
                )

    # Remote access stays possible for explicitly configured Snowflake scenarios
    # or as a last resort when local histories cannot calibrate the option.
    if rows is None and source != "local" and requested_id:
        try:
            rows = simulator.repository.fetch_source_greek_calibration(requested_id)
        except Exception as exc:
            detail = str(exc).splitlines()[0] if str(exc).splitlines() else type(exc).__name__
            if "Snowflake support is not installed" in detail:
                detail = "optional Snowflake connector is not installed"
            elif isinstance(exc, KeyError):
                detail = "Snowflake credentials are incomplete"
            reason = (
                "Greeks calibration is unavailable: "
                + (
                    "local historical data could not be calibrated"
                    + (f" ({local_problem})" if local_problem else "")
                    + "; "
                    if source == "auto"
                    else ""
                )
                + f"Snowflake: {detail}. "
                "Provide complete historical data for a local run, "
                "or configure Snowflake if you intend to use it."
            )
            raise GreeksCalibrationSetupError(reason) from exc
        if rows:
            provenance = "Snowflake L2 calibration"
            should_cache_remote = True

    if not rows:
        reason = (
            "No usable local historical Greeks calibration"
            + (f" ({local_problem})" if local_problem else "")
            + ". Provide complete historical OHLCV data for the option underlyings"
        )
        if source == "snowflake" or requested_id:
            reason += " or a valid L2 calibration in Snowflake"
        raise GreeksCalibrationSetupError(reason + ".")

    clean: dict[str, tuple[float | None, float]] = {}
    for symbol, mean, sigma in rows:
        symbol = str(symbol).upper()
        if symbol not in required:
            continue
        if symbol in clean:
            raise GreeksCalibrationSetupError(f"Duplicate calibration symbol {symbol}.")
        try:
            volatility = float(sigma)
            if not math.isfinite(volatility) or volatility <= 0:
                raise ValueError("nonpositive or nonfinite")
            mean_value = float(mean) if mean is not None else None
        except (TypeError, ValueError) as exc:
            raise GreeksCalibrationSetupError(
                f"Calibration {effective_id} has invalid daily volatility for {symbol}."
            ) from exc
        clean[symbol] = (mean_value, volatility)

    missing = required - clean.keys()
    if missing:
        raise GreeksCalibrationSetupError(
            f"Calibration {effective_id} is missing volatilities for {sorted(missing)}."
        )

    # DailyRiskCalibration has already persisted these rows under effective_id.
    # Cached or remote rows must be imported under their exact original ID.
    if provenance != "local historical VaR calibration":
        simulator.repository.import_greek_calibration(effective_id, clean)
    if should_cache_remote:
        try:
            cache.store_greek_calibration(
                calibration_id=effective_id,
                rows=[(symbol, mean, sigma) for symbol, (mean, sigma) in sorted(clean.items())],
            )
        except (OSError, duckdb.Error):
            print("Greeks calibration: shared cache is busy; this run will still work.")

    greeks["calibration_id"] = effective_id
    simulator.risk_greeks["calibration_id"] = effective_id
    print(
        f"Greeks calibration: {provenance} ({effective_id}); "
        f"option underlyings: {', '.join(sorted(required))}"
    )
