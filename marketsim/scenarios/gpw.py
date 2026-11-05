"""Prepare a full-GPW scenario from Ikaros Snowflake."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from marketsim.market.historical_prices import load_simulation_dates
from marketsim.scenarios.gpw_universe import discover_derivatives
from marketsim.scenarios.wig20 import (
    futures_expiration_day,
    query,
    to_price,
)

OHLCV = ("open", "high", "low", "close", "volume")


def wig_members(conn, cutoff: date) -> pd.DataFrame:
    """Official WIG constituents at the historical cutoff."""
    result = query(
        conn,
        """
        SELECT SECURITY_SYMBOL AS SYMBOL, WEIGHT
        FROM INDEX_CONSTITUENTS
        WHERE INDEX_SYMBOL = 'WIG'
          AND VALID_FROM <= %s
          AND (VALID_TO IS NULL OR VALID_TO >= %s)
        ORDER BY SECURITY_SYMBOL
        """,
        (cutoff, cutoff),
    )

    if result.empty:
        raise ValueError(
            f"No WIG constituents found for {cutoff}. Check INDEX_CONSTITUENTS in Ikaros."
        )

    if result.symbol.isna().any() or result.symbol.duplicated().any():
        raise ValueError("Invalid or duplicate WIG constituents")

    result["weight"] = pd.to_numeric(result.weight, errors="raise")

    if (result.weight <= 0).any():
        raise ValueError("WIG constituent weights must be positive")

    return result


def load_stock_history(
    conn,
    symbols: list[str],
    cutoff: date,
    sessions: int,
) -> dict[str, pd.DataFrame]:
    """Load all stock histories in one Snowflake query."""
    if not symbols:
        return {}

    # Snowflake parameter placeholders for the selected symbols.
    placeholders = ", ".join(["%s"] * len(symbols))

    frame = query(
        conn,
        f"""
        SELECT SYMBOL, DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        FROM (
            SELECT
                "SYMBOL" AS SYMBOL,
                TO_DATE("DATE") AS DATE,
                "OPEN" AS OPEN,
                "HIGH" AS HIGH,
                "LOW" AS LOW,
                "CLOSE" AS CLOSE,
                "VOLUME" AS VOLUME,
                ROW_NUMBER() OVER (
                    PARTITION BY "SYMBOL"
                    ORDER BY TO_DATE("DATE") DESC
                ) AS RN
            FROM PRICES
            WHERE "SYMBOL" IN ({placeholders})
              AND TO_DATE("DATE") <= %s
        )
        WHERE RN <= %s
        ORDER BY SYMBOL, DATE
        """,
        (*symbols, cutoff, sessions),
    )

    history = {}

    for symbol, group in frame.groupby("symbol"):
        data = group[["date", *OHLCV]].copy()
        data["date"] = pd.to_datetime(data["date"])

        for column in ("open", "high", "low", "close"):
            data[column] = data[column].map(to_price)

        data["volume"] = pd.to_numeric(data["volume"], errors="raise")

        data = data.sort_values("date").reset_index(drop=True)

        if data.date.duplicated().any():
            continue

        if (
            (data["volume"] < 0).any()
            or data[list(OHLCV[:4])].isna().any().any()
            or data["volume"].isna().any()
        ):
            continue

        if (
            (data["high"] < data[["open", "low", "close"]].max(axis=1))
            | (data["low"] > data[["open", "high", "close"]].min(axis=1))
        ).any():
            continue

        history[str(symbol)] = data

    return history


def load_index_history(conn, symbol: str, cutoff: date, sessions: int) -> pd.DataFrame:
    frame = query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE,
               "OPEN", "HIGH", "LOW", "CLOSE"
        FROM INDEX_PRICES
        WHERE INDEX_SYMBOL = %s
          AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC
        LIMIT %s
        """,
        (symbol, cutoff, sessions),
    )

    if frame.empty:
        raise ValueError(f"No historical index prices for {symbol}")

    frame["date"] = pd.to_datetime(frame["date"])

    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].map(to_price)

    # Index levels have no share-volume equivalent.
    frame["volume"] = 0

    return frame.sort_values("date").reset_index(drop=True)


def build_stock_markets(history: dict) -> dict:
    markets = {}

    for symbol, frame in sorted(history.items()):
        key = symbol.lower()

        markets[key] = {
            "name": symbol,
            "short_name": symbol,
            "symbol": symbol,
            "instrument_class": "stock",
            "market_type": "continuous",
            "reference_price": str(frame.close.iloc[-1]),
            "currency": "PLN",
        }

    return markets


def build_index_market(
    symbol: str,
    members: pd.DataFrame,
    history: dict,
    index_history: pd.DataFrame,
) -> dict:
    components = []

    for row in members.itertuples(index=False):
        if row.symbol not in history:
            continue

        components.append(
            {
                "security": row.symbol.lower(),
                "weight": float(row.weight),
                "base_price": str(history[row.symbol].close.iloc[-1]),
            }
        )

    if len(components) != len(members):
        raise ValueError(f"{symbol}: not all index constituents have eligible historical prices")

    return {
        "symbol": symbol,
        "name": symbol,
        "short_name": symbol,
        "instrument_class": "index",
        "components": components,
        "base_index_level": str(index_history.close.iloc[-1]),
    }


def build_futures(
    derivatives: pd.DataFrame,
    market_keys: set[str],
    simulation_dates: list[date],
    margin_rate: float,
) -> tuple[dict, list[dict]]:
    markets = {}
    rejected = []

    for row in derivatives.itertuples(index=False):
        if str(row.instrument_type).upper() not in {"FUTURE", "FUTURES"}:
            continue

        symbol = str(row.symbol)
        key = symbol.lower()
        underlying = str(row.underlying_symbol).lower()

        reason = None

        if underlying not in market_keys:
            reason = "missing_underlying"
        elif key in market_keys or key in markets:
            reason = "duplicate_symbol"
        elif pd.isna(row.multiplier) or float(row.multiplier) <= 0:
            reason = "invalid_multiplier"
        elif pd.isna(row.expiry_date):
            reason = "missing_expiration"

        if reason:
            rejected.append(
                {
                    "symbol": symbol,
                    "instrument_type": "FUTURE",
                    "reason": reason,
                }
            )
            continue

        markets[key] = {
            "symbol": symbol,
            "name": symbol,
            "short_name": symbol,
            "instrument_class": "future",
            "underlying": underlying,
            "expiration": futures_expiration_day(row.expiry_date, simulation_dates),
            "multiplier": float(row.multiplier),
            "margin_rate": margin_rate,
        }

    return markets, rejected


def build_options(
    conn,
    derivatives: pd.DataFrame,
    market_keys: set[str],
    simulation_dates: list[date],
) -> tuple[dict, list[dict]]:
    markets = {}
    rejected = []

    # Read the actual metadata columns rather than assuming
    # that the discovery query includes strike and option side.
    metadata = query(conn, "SELECT * FROM DERIVATIVE_CONTRACTS WHERE 1 = 0")
    columns = set(metadata.columns)

    strike_column = next(
        (c for c in ("strike_price", "strike") if c in columns),
        None,
    )
    side_column = next(
        (c for c in ("option_type", "option_side", "call_put") if c in columns),
        None,
    )

    option_rows = derivatives.loc[
        derivatives.instrument_type.astype(str).str.upper().isin(["OPTION", "OPTIONS"])
    ]

    if option_rows.empty:
        return markets, rejected

    if strike_column is None or side_column is None:
        for row in option_rows.itertuples(index=False):
            rejected.append(
                {
                    "symbol": row.symbol,
                    "instrument_type": "OPTION",
                    "reason": "option_metadata_schema_unresolved",
                }
            )
        return markets, rejected

    contract_ids = option_rows.contract_id.tolist()
    placeholders = ", ".join(["%s"] * len(contract_ids))

    details = (
        query(
            conn,
            f"""
        SELECT CONTRACT_ID,
               "{strike_column.upper()}" AS STRIKE,
               "{side_column.upper()}" AS SIDE
        FROM DERIVATIVE_CONTRACTS
        WHERE CONTRACT_ID IN ({placeholders})
        """,
            tuple(contract_ids),
        )
        .drop_duplicates("contract_id")
        .set_index("contract_id")
    )

    for row in option_rows.itertuples(index=False):
        symbol = str(row.symbol)
        key = symbol.lower()
        underlying = str(row.underlying_symbol).lower()
        reason = None

        if underlying not in market_keys:
            reason = "missing_underlying"
        elif key in market_keys or key in markets:
            reason = "duplicate_symbol"
        elif row.contract_id not in details.index:
            reason = "missing_option_metadata"
        elif pd.isna(row.multiplier) or float(row.multiplier) != 1:
            reason = "option_multiplier_not_supported"
        elif pd.isna(row.expiry_date):
            reason = "missing_expiration"

        if reason:
            rejected.append(
                {
                    "symbol": symbol,
                    "instrument_type": "OPTION",
                    "reason": reason,
                }
            )
            continue

        detail = details.loc[row.contract_id]
        side = str(detail["side"]).upper()

        side = {
            "C": "CALL",
            "P": "PUT",
        }.get(side, side)

        if side not in {"CALL", "PUT"}:
            reason = "invalid_option_side"
        elif pd.isna(detail["strike"]):
            reason = "missing_strike"
        elif float(detail["strike"]) <= 0:
            reason = "invalid_strike"

        if reason:
            rejected.append(
                {
                    "symbol": symbol,
                    "instrument_type": "OPTION",
                    "reason": reason,
                }
            )
            continue

        markets[key] = {
            "symbol": symbol,
            "name": symbol,
            "short_name": symbol,
            "instrument_class": "option",
            "market_type": "continuous",
            "derivatives_config": {
                "underlying": underlying,
                "strike": float(detail["strike"]),
                "option_side": side,
                "option_type": "European",
                "expiration": futures_expiration_day(row.expiry_date, simulation_dates),
            },
        }

    return markets, rejected


def load_derivative_history(
    conn,
    derivatives: pd.DataFrame,
    symbols: set[str],
    cutoff: date,
    sessions: int,
) -> list[pd.DataFrame]:
    frames = []

    selected = derivatives.loc[derivatives.symbol.astype(str).str.upper().isin(symbols)]

    for row in selected.itertuples(index=False):
        frame = query(
            conn,
            """
            SELECT TO_DATE("DATE") AS DATE,
                   "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
            FROM DERIVATIVE_PRICES
            WHERE CONTRACT_ID = %s
              AND TO_DATE("DATE") <= %s
            ORDER BY TO_DATE("DATE") DESC
            LIMIT %s
            """,
            (row.contract_id, cutoff, sessions),
        )

        if frame.empty:
            continue

        frame["date"] = pd.to_datetime(frame["date"])

        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column].map(to_price)

        frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")

        frame["symbol"] = row.symbol
        frames.append(frame.sort_values("date")[["symbol", "date", *OHLCV]])

    return frames


def prepare(conn, settings: dict, root: Path):
    lookback = int(settings.get("lookback_sessions", 252))
    min_history = int(settings.get("minimum_history_sessions", 30))

    if lookback < 4 or not 4 <= min_history <= lookback:
        raise ValueError("Invalid history requirements")

    # Use the latest available WIG index session before today.
    cutoff_frame = query(
        conn,
        """
        SELECT MAX(TO_DATE("DATE")) AS CUTOFF
        FROM INDEX_PRICES
        WHERE INDEX_SYMBOL = 'WIG'
          AND TO_DATE("DATE") < %s
        """,
        (date.today(),),
    )

    cutoff = cutoff_frame.iloc[0]["cutoff"]

    if pd.isna(cutoff):
        raise ValueError("No WIG cutoff available")

    cutoff = pd.Timestamp(cutoff).date()

    members = wig_members(conn, cutoff)

    stock_history = load_stock_history(
        conn,
        members.symbol.tolist(),
        cutoff,
        lookback,
    )

    # Keep a report of stocks that cannot be included.
    rejected = []

    for symbol in members.symbol:
        frame = stock_history.get(symbol)

        if frame is None or len(frame) < min_history:
            rejected.append(
                {
                    "symbol": symbol,
                    "instrument_type": "STOCK",
                    "reason": "insufficient_history",
                }
            )
            stock_history.pop(symbol, None)
        elif frame.date.iloc[-1].date() != cutoff:
            rejected.append(
                {
                    "symbol": symbol,
                    "instrument_type": "STOCK",
                    "reason": "stale_last_price",
                }
            )
            stock_history.pop(symbol, None)

    if not stock_history:
        raise ValueError("No eligible WIG stocks")

    index_prices = load_index_history(conn, "WIG", cutoff, lookback)

    output = root / "marketsim" / "input"
    data_dir = output / "market_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    with (output / settings.get("agents_profile", "gpw_50d.yaml")).open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)

    base["days"] = int(settings.get("days", 50))
    base["sim_time"] = int(settings.get("sim_time", 60))
    base["seed"] = int(settings.get("seed", 67))
    base["start_date"] = (cutoff + timedelta(days=1)).isoformat()

    simulation_dates = load_simulation_dates(base)

    markets = build_stock_markets(stock_history)

    # Build WIG only if all its constituents are present.
    if len(stock_history) == len(members):
        markets["wig"] = build_index_market("WIG", members, stock_history, index_prices)
    else:
        rejected.append(
            {
                "symbol": "WIG",
                "instrument_type": "INDEX",
                "reason": "missing_constituent",
            }
        )

    derivatives = discover_derivatives(conn, cutoff)

    futures, futures_rejected = build_futures(
        derivatives,
        set(markets),
        simulation_dates,
        float(settings.get("margin_rate", 0.10)),
    )

    markets.update(futures)
    rejected.extend(futures_rejected)

    # Options are handled by the separate validated builder below.
    options, options_rejected = build_options(
        conn,
        derivatives,
        set(markets),
        simulation_dates,
    )
    markets.update(options)
    rejected.extend(options_rejected)

    frames = []

    for symbol, frame in stock_history.items():
        data = frame.copy()
        data["symbol"] = symbol
        frames.append(data[["symbol", "date", *OHLCV]])

    index_data = index_prices.copy()
    index_data["symbol"] = "WIG"
    frames.append(index_data[["symbol", "date", *OHLCV]])

    # Historical derivative prices are loaded separately,
    # only for contracts included in the scenario.
    derivative_symbols = {
        key.upper()
        for key, market in markets.items()
        if market["instrument_class"] in {"future", "option"}
    }

    if derivative_symbols:
        derivative_history = load_derivative_history(
            conn, derivatives, derivative_symbols, cutoff, lookback
        )
        frames.extend(derivative_history)

    csv_name = "gpw_auto.ohlcv.csv"
    yaml_name = "gpw_auto.yaml"

    all_history = pd.concat(frames, ignore_index=True)
    all_history["date"] = pd.to_datetime(all_history["date"]).dt.strftime("%Y-%m-%d")

    all_history.to_csv(data_dir / csv_name, index=False)

    base["historical_prices"] = {
        "enabled": True,
        "path": f"market_data/{csv_name}",
        "lookback_sessions": lookback,
        "cutoff_date": cutoff.isoformat(),
        "allow_short_history": sorted(set(all_history.symbol)),
    }

    base["markets"] = markets

    with (output / yaml_name).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(base, stream, sort_keys=False)

    pd.DataFrame(
        rejected,
        columns=["symbol", "instrument_type", "reason"],
    ).to_csv(data_dir / "gpw_excluded.csv", index=False)

    return output / yaml_name, cutoff, len(markets)
