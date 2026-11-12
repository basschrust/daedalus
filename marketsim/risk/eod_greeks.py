"""EOD option Greeks, with mandatory immutable calibration identifier.

Rates are one short-term annual rate per currency (tenor curves are future work).
Calibration sigma comes from L2 instrument statistics; no silent volatility defaults.
"""

from __future__ import annotations

import math

from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut


def _valid(name, value, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _symbol(market):
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).upper()


def _load_sigmas(conn, calibration_id, symbols):
    # Snowflake calibrated L2 statistics are authoritative. Local DuckDB L2
    # statistics may be used for offline tests and DuckDB simulations.
    is_snowflake = "snowflake" in type(conn).__module__.lower()
    if is_snowflake:
        sql = (
            "SELECT SYMBOL, ANNUALIZED_SIGMA FROM "
            "DAEDALUS.MART.L2_INSTRUMENT_STATISTICS WHERE CALIBRATION_ID = %s"
        )
    else:
        sql = (
            "SELECT symbol, daily_sigma * SQRT(252.0) "
            "FROM risk_l2_instrument_statistics WHERE calibration_id = ?"
        )
    cursor = conn.cursor()
    try:
        cursor.execute(sql, (calibration_id,))
        rows = cursor.fetchall()
    finally:
        cursor.close()
    values = {}
    for symbol, sigma in rows:
        symbol = str(symbol).upper()
        if symbol in symbols:
            if symbol in values or sigma is None:
                raise ValueError(f"Duplicate or null volatility for {symbol}")
            values[symbol] = _valid(f"sigma {symbol}", sigma, positive=True)
    missing = symbols - values.keys()
    if missing:
        raise ValueError(f"Calibration {calibration_id} missing volatility for {sorted(missing)}")
    return values


def _ensure_tables(conn):
    snowflake = "snowflake" in type(conn).__module__.lower()
    integer = "NUMBER(38,0)" if snowflake else "INTEGER"
    # Normalized, calibration-scoped assumptions. Never overwrite a published
    # calibration's values. The caller checks existing values for equality.
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
    cur = conn.cursor()
    try:
        for name, columns in definitions.items():
            cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({columns})")
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
                if snowflake:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {dtype}")
                else:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {dtype}")
    finally:
        cur.close()


def _persist_inputs(conn, table, calibration_id, values, column, *, placeholder):
    cur = conn.cursor()
    try:
        for key, value in sorted(values.items()):
            cur.execute(
                f"SELECT {column} FROM {table} WHERE calibration_id={placeholder} "
                f"AND {('currency' if table.endswith('rates') else 'symbol')}={placeholder}",
                (calibration_id, key),
            )
            existing = cur.fetchall()
            if existing:
                if len(existing) != 1 or not math.isclose(
                    float(existing[0][0]), value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"Published calibration input changed: {table}/{key}")
            else:
                key_col = "currency" if table.endswith("rates") else "symbol"
                cur.execute(
                    f"INSERT INTO {table} (calibration_id,{key_col},{column}) "
                    f"VALUES ({placeholder},{placeholder},{placeholder})",
                    (calibration_id, key, value),
                )
    finally:
        cur.close()


def write_eod_greeks(simulator, day: int, config: dict) -> None:
    """Calculate and persist one EOD snapshot, atomically for both result tables."""
    calibration_id = str(config.get("calibration_id") or "").strip()
    if not calibration_id:
        raise ValueError("risk_greeks.calibration_id is mandatory")
    options = [m for m in simulator.markets.values() if m.instrument_class == "option"]
    if not options:
        return
    conn = simulator.repository.connection
    symbols = {_symbol(m.underlying) for m in options}
    if not all(symbols):
        raise ValueError("Every option underlying must have a symbol/short_name")
    sigmas = _load_sigmas(conn, calibration_id, symbols)
    rates = {
        str(k).upper(): _valid(f"rate {k}", v) for k, v in config.get("short_rates", {}).items()
    }
    dividends = {
        str(k).upper(): _valid(f"dividend {k}", v)
        for k, v in config.get("dividend_yields", {}).items()
    }
    # Missing rate or dividend assumption is an error, not a silent zero.
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
            "Current BlackScholes implementation does not support nonzero dividend yields consistently; use explicit zero until model is corrected"
        )
    _ensure_tables(conn)
    snowflake = "snowflake" in type(conn).__module__.lower()
    placeholder = "%s" if snowflake else "?"
    _persist_inputs(
        conn,
        "risk_greeks_short_rates",
        calibration_id,
        rates,
        "annual_rate",
        placeholder=placeholder,
    )
    _persist_inputs(
        conn,
        "risk_greeks_dividend_yields",
        calibration_id,
        dividends,
        "annual_yield",
        placeholder=placeholder,
    )
    _persist_inputs(
        conn,
        "risk_greeks_volatilities",
        calibration_id,
        sigmas,
        "annual_sigma",
        placeholder=placeholder,
    )
    cur = conn.cursor()
    try:
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
            greeks = {
                name: float(values[name]) for name in ("delta", "gamma", "vega", "theta", "rho")
            }
            if not all(math.isfinite(v) for v in greeks.values()):
                raise ValueError(f"Non-finite Greeks for option {option.asset_id}")
            multiplier = float(option.multiplier)
            cur.execute(
                f"DELETE FROM eod_option_greeks WHERE day={placeholder} AND asset_id={placeholder} AND calibration_id={placeholder}",
                (day, option.asset_id, calibration_id),
            )
            columns = [
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
            ]
            record = (
                day,
                option.asset_id,
                underlying.asset_id,
                "BLACK_SCHOLES",
                *(greeks[n] for n in ("delta", "gamma", "vega", "theta", "rho")),
                multiplier,
                calibration_id,
            )
            cur.execute(
                f"INSERT INTO eod_option_greeks ({','.join(columns)}) VALUES ({','.join([placeholder] * len(columns))})",
                record,
            )
            for agent in simulator.agents.values():
                position = int(agent.position.get(option.asset_id, 0))
                if not position:
                    continue
                cur.execute(
                    f"DELETE FROM eod_agent_option_greeks WHERE day={placeholder} AND agent_id={placeholder} AND asset_id={placeholder} AND calibration_id={placeholder}",
                    (day, agent.agent_id, option.asset_id, calibration_id),
                )
                columns = [
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
                ]
                record = (
                    day,
                    agent.agent_id,
                    option.asset_id,
                    underlying.asset_id,
                    position,
                    greeks["delta"],
                    position * multiplier * greeks["delta"],
                    multiplier,
                    calibration_id,
                    *(position * multiplier * greeks[n] for n in ("gamma", "vega", "theta", "rho")),
                )
                cur.execute(
                    f"INSERT INTO eod_agent_option_greeks ({','.join(columns)}) VALUES ({','.join([placeholder] * len(columns))})",
                    record,
                )
    finally:
        cur.close()
