"""Prepare a WIG20 scenario from an Ikaros market-data source."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from marketsim.database.ikaros import ikaros_queries
from marketsim.database.ikaros.wig20_history_queries import fetch_stock_window
from marketsim.market.historical_prices import load_simulation_dates
from marketsim.scenarios.scenario_settings import build_scenario_settings
from marketsim.scenarios.wig20_data import (
    align_stock_history,
    futures_history,
    index_session_history,
    to_price,
)
from marketsim.scenarios.wig20_options import select_options

OHLCV = ("open", "high", "low", "close", "volume")


def futures_expiration_day(expiry_date, simulation_dates):
    expiry = pd.Timestamp(expiry_date).date()

    for day, session_date in enumerate(simulation_dates):
        if session_date >= expiry:
            return day

    # Contract expires after the simulation horizon.
    return len(simulation_dates)


def constituents(conn, as_of: date):
    frame = ikaros_queries.fetch_wig20_constituents(conn, as_of)

    if len(frame) != 20 or frame.symbol.isna().any() or frame.symbol.duplicated().any():
        raise ValueError(f"WIG20 on {as_of}: expected 20 unique constituents; got {len(frame)}")

    frame["weight"] = pd.to_numeric(frame.weight, errors="raise")

    if not frame.weight.map(math.isfinite).all() or (frame.weight <= 0).any():
        raise ValueError("Invalid WIG20 weights")

    return frame


def candidate_dates(conn, max_date: date):
    """Get possible cutoff dates from the stock price table."""
    frame = ikaros_queries.fetch_candidate_dates(conn, max_date)

    return [pd.Timestamp(value).date() for value in frame.session_date.tolist()]


def _non_trading_dates(values):
    return {date.fromisoformat(str(value)) for value in values}


def _index_window(conn, cutoff, sessions, holidays):
    # Overfetch before filtering source observations dated on weekends/holidays.
    raw = ikaros_queries.fetch_wig20_index_history(conn, cutoff, sessions * 3)
    return index_session_history(
        raw,
        cutoff=cutoff,
        sessions=sessions,
        non_trading_dates=holidays,
    )


def stock_history(
    conn,
    symbol: str,
    cutoff: date,
    sessions: int,
    *,
    expected_sessions,
    non_trading_dates=(),
):
    """Validate full date window; do not count a holiday as a trading session."""
    if len(expected_sessions) != sessions:
        raise ValueError(f"{symbol}: expected {sessions} index sessions")

    raw = fetch_stock_window(
        conn,
        symbol=symbol,
        first_session=expected_sessions[0],
        cutoff=cutoff,
    )
    frame, ignored = align_stock_history(
        raw,
        symbol=symbol,
        required_sessions=expected_sessions,
        non_trading_dates=non_trading_dates,
    )
    frame["date"] = pd.to_datetime(frame["date"])

    for col in ("open", "high", "low", "close"):
        frame[col] = frame[col].map(to_price)

    frame["volume"] = pd.to_numeric(frame["volume"], errors="raise")

    if not frame["volume"].map(math.isfinite).all() or (frame["volume"] < 0).any():
        raise ValueError(f"{symbol}: invalid volume")

    if (frame[list(OHLCV[:4])] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError(f"{symbol}: nonpositive prices or negative volume")

    if (
        (frame.high < frame[["open", "close", "low"]].max(axis=1))
        | (frame.low > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError(f"{symbol}: inconsistent OHLC")

    return frame.sort_values("date").reset_index(drop=True), ignored


def fetch_index_close(conn, cutoff: date) -> float:
    """Load the official WIG20 EoD close from Ikaros."""

    frame = ikaros_queries.fetch_wig20_index_close(conn, cutoff)

    if len(frame) != 1:
        raise ValueError(
            f"WIG20: expected exactly one EoD close "
            f"in INDEX_PRICES for {cutoff}, "
            f"found {len(frame)}"
        )

    value = to_price(frame.iloc[0]["index_close"])

    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"WIG20: invalid EoD close {value!r} on {cutoff}")

    return value


def resolve(conn, sessions=250, *, non_trading_dates=()):
    """Resolve newest complete candidate and report per-symbol calendar defects.

    A close on a weekend or a known exchange holiday cannot become a cutoff.
    The *observed* index sessions define the required window; a separately
    verified GPW calendar would provide stronger completeness guarantees.
    """
    if sessions < 2:
        raise ValueError("At least two lookback sessions required")

    holidays = _non_trading_dates(non_trading_dates)
    diagnostics = []

    # Exclude today, since today's ingest may be partial.
    max_date = date.today() - timedelta(days=1)

    for cutoff in candidate_dates(conn, max_date):
        if cutoff.weekday() >= 5 or cutoff in holidays:
            continue
        try:
            members = constituents(conn, cutoff)
            index_window = _index_window(conn, cutoff, sessions, holidays)
            expected = tuple(index_window["date"].tolist())
            excluded_index = index_window.attrs.get("excluded_non_trading_dates", ())

            history = {}
            ignored = {}
            failures = []
            for symbol in members.symbol:
                try:
                    frame, skipped = stock_history(
                        conn,
                        symbol,
                        cutoff,
                        sessions,
                        expected_sessions=expected,
                        non_trading_dates=holidays,
                    )
                    history[symbol] = frame
                    if skipped:
                        ignored[symbol] = skipped
                except ValueError as exc:
                    failures.append(str(exc))
            ignored_summary = []
            if excluded_index:
                ignored_summary.append(
                    "WIG20 index excluded non-trading dates: "
                    + ", ".join(day.isoformat() for day in excluded_index[:5])
                )
            if ignored:
                ignored_summary.append(
                    "excluded non-trading stock observations: " + ", ".join(sorted(ignored)[:8])
                )
            if failures:
                suffix = f"; (+{len(failures) - 8} other instruments)" if len(failures) > 8 else ""
                notice = " | " + " | ".join(ignored_summary) if ignored_summary else ""
                raise ValueError(" | ".join(failures[:8]) + suffix + notice)

            index_level = fetch_index_close(conn, cutoff)

            for notice in ignored_summary:
                print(f"WIG20 {cutoff}: {notice}")
            for symbol, days in sorted(ignored.items()):
                dates = ", ".join(value.isoformat() for value in days[:5])
                if len(days) > 5:
                    dates += f" (+{len(days) - 5} more)"
                print(f"WIG20 {cutoff}: excluded {symbol} non-trading observations: {dates}")

            return cutoff, members, history, index_level

        except ValueError as exc:
            diagnostics.append(f"{cutoff}: {exc}")

    raise ValueError(
        "No complete WIG20 cutoff in the last 90 observed dates.\n" + "\n".join(diagnostics[:12])
    )


def build_markets(
    members,
    history,
    index_level,
    contracts=None,
    simulation_dates=None,
    margin_rate=0.10,
):
    """Create stocks first, followed by the WIG20 index."""
    index_level = to_price(index_level)

    markets = {}
    components = []

    for row in members.itertuples(index=False):
        symbol = row.symbol
        key = symbol.lower()
        close = to_price(history[symbol].close.iloc[-1])

        markets[key] = dict(
            name=symbol,
            short_name=symbol,
            symbol=symbol,
            instrument_class="stock",
            market_type="continuous",
            reference_price=str(close),
            currency="PLN",
        )

        components.append(
            dict(
                security=key,
                weight=float(row.weight),
                base_price=str(close),
            )
        )

    markets["wig20"] = dict(
        symbol="WIG20",
        name="WIG20",
        short_name="WIG20",
        instrument_class="index",
        components=components,
        base_index_level=str(to_price(index_level)),
    )

    if contracts is not None and not contracts.empty:
        if simulation_dates is None:
            raise ValueError("simulation_dates required when futures are provided")

        for contract in contracts.itertuples(index=False):
            expiration_day = futures_expiration_day(
                contract.expiry_date,
                simulation_dates,
            )

            key = contract.symbol.lower()

            if key in markets:
                raise ValueError(f"Duplicate market: {key}")

            markets[key] = {
                "symbol": contract.symbol,
                "name": contract.symbol,
                "short_name": contract.symbol,
                "instrument_class": "future",
                "underlying": "wig20",
                "expiration": expiration_day,
                "multiplier": float(contract.multiplier),
                "margin_rate": margin_rate,
            }

    return markets


def prepare(conn, settings: dict, root: Path):
    """Build and save a reproducible Daedalus scenario."""

    lookback = int(settings.get("lookback_sessions", 250))
    holidays = _non_trading_dates(settings.get("non_trading_dates", []))

    cutoff, members, history, index_level = resolve(
        conn,
        lookback,
        non_trading_dates=holidays,
    )

    output = root / "input"
    data_dir = output / "market_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "wig20_auto.ohlcv.csv"
    yaml_name = "wig20_auto.yaml"

    frames = []

    # 20 WIG20 constituent stocks
    for symbol, frame in history.items():
        data = frame.copy()
        data["symbol"] = symbol
        data["date"] = data.date.dt.strftime("%Y-%m-%d")

        frames.append(data[["symbol", "date", *OHLCV]])

    # Official WIG20 index history, aligned to precisely the same session set.
    wig20_history = index_history(
        conn,
        cutoff,
        sessions=lookback,
        non_trading_dates=holidays,
    )

    index_data = wig20_history.reset_index()
    index_data["symbol"] = "WIG20"
    index_data["date"] = index_data["date"].dt.strftime("%Y-%m-%d")

    frames.append(index_data[["symbol", "date", *OHLCV]])

    contracts = futures_contracts(
        conn,
        cutoff,
        count=int(settings.get("futures_count", 1)),
    )

    for contract in contracts.itertuples(index=False):
        frame = futures_history(
            conn,
            contract.contract_id,
            cutoff,
            sessions=lookback,
        )

        frame["symbol"] = contract.symbol
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

        frames.append(frame[["symbol", "date", *OHLCV]])

    start_date = (cutoff + timedelta(days=1)).isoformat()
    simulation_dates = load_simulation_dates(
        {
            **settings,
            "start_date": start_date,
        }
    )

    allowed_symbols = ["WIG20", *contracts["symbol"].tolist()]

    options = select_options(
        conn,
        cutoff,
        simulation_dates,
        index_level,
        count=int(settings.get("options_count", 4)),
        lookback_sessions=lookback,
    )

    for option in options:
        data = option["history"].copy()
        data["symbol"] = option["symbol"]
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")

        frames.append(data[["symbol", "date", *OHLCV]])
        allowed_symbols.append(option["symbol"])

    all_history = pd.concat(frames, ignore_index=True)
    expected = 21 + len(contracts) + len(options)
    assert all_history["symbol"].nunique() == expected
    all_history.to_csv(data_dir / csv_name, index=False)

    markets = build_markets(
        members,
        history,
        index_level,
        contracts=contracts,
        simulation_dates=simulation_dates,
    )

    for option in options:
        symbol = option["symbol"]
        key = symbol.lower()

        if key in markets:
            raise ValueError(f"Duplicate market: {symbol}")

        markets[key] = {
            "symbol": symbol,
            "name": symbol,
            "short_name": symbol,
            "instrument_class": "option",
            "market_type": "continuous",
            "derivatives_config": {
                "underlying": "wig20",
                "strike": str(option["strike"]),
                "option_side": option["side"],
                "option_type": "European",
                "expiration": futures_expiration_day(
                    option["expiry_date"],
                    simulation_dates,
                ),
                "multiplier": option["multiplier"],
            },
        }

    market_data = {
        "markets": markets,
        "start_date": start_date,
        "historical_prices": {
            "enabled": True,
            "path": f"market_data/{csv_name}",
            "lookback_sessions": lookback,
            "cutoff_date": cutoff.isoformat(),
            "allow_short_history": allowed_symbols,
        },
    }

    base = build_scenario_settings(
        settings=settings,
        market_data=market_data,
    )

    with (output / yaml_name).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(base, stream, sort_keys=False)

    return output / yaml_name, cutoff, index_level


def index_history(conn, cutoff, sessions=250, *, non_trading_dates=()):
    frame = _index_window(conn, cutoff, sessions, _non_trading_dates(non_trading_dates))

    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].map(to_price)

    frame["date"] = pd.to_datetime(frame["date"])
    frame["volume"] = 0

    return frame.sort_values("date").set_index("date")


def futures_contracts(conn, cutoff, count=3):
    frame = ikaros_queries.fetch_wig20_futures_contracts(conn, cutoff)

    frame = frame.drop_duplicates(subset=["contract_id"]).head(count)

    if len(frame) != count:
        raise ValueError(
            f"Expected {count} WIG20 futures contracts, "
            f"found {len(frame)}. "
            "Check Ikaros contract discovery."
        )

    return frame
