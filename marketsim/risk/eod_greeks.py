"""Calculate EoD option Greeks; persistence belongs to the repository.

One short-term rate per currency; the current Black--Scholes implementation
requires explicit zero dividend yield.  No SQL, connection or backend logic here.
"""

from __future__ import annotations

import math
from typing import Any

from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut


def _valid(name: str, value: object, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise ValueError(f"Invalid {name}: {number}")
    return number


def _symbol(market: object) -> str:
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).upper()


def _load_sigmas(repository: Any, calibration_id: str, symbols: set[str]) -> dict[str, float]:
    """Validate the calibrated annualized volatilities returned by the repository."""
    rows = repository.load_greek_sigmas(calibration_id)
    values: dict[str, float] = {}
    for raw_symbol, sigma in rows:
        symbol = str(raw_symbol).upper()
        if symbol in symbols:
            if symbol in values or sigma is None:
                raise ValueError(f"Duplicate or null volatility for {symbol}")
            values[symbol] = _valid(f"sigma {symbol}", sigma, positive=True)
    missing = symbols - values.keys()
    if missing:
        raise ValueError(f"Calibration {calibration_id} missing volatility for {sorted(missing)}")
    return values


def write_eod_greeks(simulator: Any, day: int, config: dict) -> None:
    """Compute one EoD Greek snapshot, using only named repository operations."""
    calibration_id = str(config.get("calibration_id") or "").strip()
    if not calibration_id:
        raise ValueError("risk_greeks.calibration_id is mandatory")
    options = [m for m in simulator.markets.values() if m.instrument_class == "option"]
    if not options:
        return
    repository = simulator.repository
    symbols = {_symbol(m.underlying) for m in options}
    if not all(symbols):
        raise ValueError("Every option underlying must have a symbol/short_name")
    sigmas = _load_sigmas(repository, calibration_id, symbols)
    rates = {
        str(k).upper(): _valid(f"rate {k}", v) for k, v in config.get("short_rates", {}).items()
    }
    dividends = {
        str(k).upper(): _valid(f"dividend {k}", v)
        for k, v in config.get("dividend_yields", {}).items()
    }
    for option in options:
        currency = str(
            getattr(option, "currency", None) or getattr(option.underlying, "currency", "PLN")
        ).upper()
        if currency not in rates:
            raise ValueError(f"Missing short-term rate for {currency}")
        if _symbol(option.underlying) not in dividends:
            raise ValueError(f"Missing dividend assumption for {_symbol(option.underlying)}")
    if any(value != 0 for value in dividends.values()):
        raise ValueError(
            "Current BlackScholes implementation does not support nonzero dividend yields "
            "consistently; use explicit zero until model is corrected"
        )

    # These operations retain their current schemas and immutable inputs.
    repository.ensure_greek_tables()
    repository.save_greek_inputs(calibration_id, rates, dividends, sigmas)
    option_rows: list[tuple] = []
    agent_rows: list[tuple] = []
    for option in options:
        underlying = option.underlying
        symbol = _symbol(underlying)
        currency = str(
            getattr(option, "currency", None) or getattr(underlying, "currency", "PLN")
        ).upper()
        remaining = max(0, int(option.expiration) - int(day))
        spot = _valid(f"underlying close {symbol}", underlying.last_traded_price, positive=True)
        strike = _valid("strike", option.strike, positive=True)
        model = (
            BSCall
            if option.option_side.upper() == "CALL"
            else BSPut
            if option.option_side.upper() == "PUT"
            else None
        )
        if model is None:
            raise ValueError(f"Invalid option side {option.option_side}")
        values = model(
            S=spot,
            K=strike,
            r=rates[currency],
            volatility=sigmas[symbol],
            Time=remaining / 252.0,
            d=dividends[symbol],
        )
        greeks = {name: float(values[name]) for name in ("delta", "gamma", "vega", "theta", "rho")}
        if not all(math.isfinite(v) for v in greeks.values()):
            raise ValueError(f"Non-finite Greeks for option {option.asset_id}")
        multiplier = float(option.multiplier)
        option_rows.append(
            (
                day,
                option.asset_id,
                underlying.asset_id,
                "BLACK_SCHOLES",
                *(greeks[name] for name in ("delta", "gamma", "vega", "theta", "rho")),
                multiplier,
                calibration_id,
            )
        )
        for agent in simulator.agents.values():
            position = int(agent.position.get(option.asset_id, 0))
            if not position:
                continue
            agent_rows.append(
                (
                    day,
                    agent.agent_id,
                    option.asset_id,
                    underlying.asset_id,
                    position,
                    greeks["delta"],
                    position * multiplier * greeks["delta"],
                    multiplier,
                    calibration_id,
                    *(
                        position * multiplier * greeks[name]
                        for name in ("gamma", "vega", "theta", "rho")
                    ),
                )
            )
    repository.save_eod_greek_results(day, calibration_id, option_rows, agent_rows)
