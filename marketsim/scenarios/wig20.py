"""Prepare a 21-market WIG20 scenario from Ikaros Snowflake."""

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import yaml

from marketsim.market.historical_prices import load_simulation_dates

OHLCV = ("open", "high", "low", "close", "volume")

PRICE_QUANTUM = Decimal("0.01")


def futures_expiration_day(expiry_date, simulation_dates):
    expiry = pd.Timestamp(expiry_date).date()

    for day, session_date in enumerate(simulation_dates):
        if session_date >= expiry:
            return day

    # Contract expires after the simulation horizon.
    return len(simulation_dates)


def to_price(value) -> Decimal:
    """Convert source values to Daedalus price precision."""
    if isinstance(value, Decimal):
        result = value
    else:
        result = Decimal(str(value))

    if not result.is_finite() or result <= 0:
        raise ValueError(f"Invalid price: {value!r}")

    return result.quantize(
        PRICE_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def connect():
    import snowflake.connector

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


def query(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [item[0].lower() for item in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def constituents(conn, as_of: date):
    frame = query(
        conn,
        """
        SELECT SECURITY_SYMBOL AS SYMBOL, WEIGHT,
               VALID_FROM, VALID_TO, SOURCE
        FROM INDEX_CONSTITUENTS
        WHERE INDEX_SYMBOL = 'WIG20'
          AND VALID_FROM <= %s
          AND (VALID_TO IS NULL OR VALID_TO >= %s)
        ORDER BY SECURITY_SYMBOL
    """,
        (as_of, as_of),
    )

    if len(frame) != 20 or frame.symbol.isna().any() or frame.symbol.duplicated().any():
        raise ValueError(f"WIG20 on {as_of}: expected 20 unique constituents; got {len(frame)}")

    frame["weight"] = pd.to_numeric(frame.weight, errors="raise")

    if not frame.weight.map(math.isfinite).all() or (frame.weight <= 0).any():
        raise ValueError("Invalid WIG20 weights")

    return frame


def candidate_dates(conn, max_date: date):
    """Get possible cutoff dates from the stock price table."""
    frame = query(
        conn,
        """
        SELECT DISTINCT TO_DATE("DATE") AS SESSION_DATE
        FROM PRICES
        WHERE TO_DATE("DATE") <= %s
        ORDER BY SESSION_DATE DESC
        LIMIT 90
    """,
        (max_date,),
    )

    return frame.session_date.tolist()


def stock_history(
    conn,
    symbol: str,
    cutoff: date,
    sessions: int,
):
    frame = query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE,
               "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"
        FROM PRICES
        WHERE "SYMBOL" = %s
          AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC
        LIMIT %s
    """,
        (symbol, cutoff, sessions),
    )

    if len(frame) != sessions:
        raise ValueError(f"{symbol}: only {len(frame)} of {sessions} historical sessions")

    frame["date"] = pd.to_datetime(frame.date)

    if frame.date.duplicated().any() or frame.date.max().date() != cutoff:
        raise ValueError(f"{symbol}: duplicate sessions or missing cutoff date")

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

    return frame.sort_values("date").reset_index(drop=True)


def fetch_index_close(conn, cutoff: date) -> float:
    """Load the official WIG20 EoD close from Ikaros."""

    frame = query(
        conn,
        """
        SELECT "CLOSE" AS INDEX_CLOSE
        FROM INDEX_PRICES
        WHERE "INDEX_SYMBOL" = %s
          AND TO_DATE("DATE") = %s
    """,
        ("WIG20", cutoff),
    )

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


def resolve(conn, sessions=250):
    """Resolve the latest complete stock and index snapshot."""

    if sessions < 2:
        raise ValueError("At least two lookback sessions required")

    diagnostics = []

    # Exclude today, since today's ingest may be partial.
    max_date = date.today() - timedelta(days=1)

    for cutoff in candidate_dates(conn, max_date):
        try:
            members = constituents(conn, cutoff)

            history = {
                symbol: stock_history(conn, symbol, cutoff, sessions) for symbol in members.symbol
            }

            calendars = {tuple(frame.date.dt.date) for frame in history.values()}

            if len(calendars) != 1:
                raise ValueError("Stock trading calendars do not match")

            index_level = fetch_index_close(conn, cutoff)

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

    cutoff, members, history, index_level = resolve(conn, lookback)

    output = root / "marketsim" / "input"
    data_dir = output / "market_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_name = "wig20_auto.ohlcv.csv"
    yaml_name = "wig20_auto.yaml"

    frames = []

    for symbol, frame in history.items():
        data = frame.copy()
        data["symbol"] = symbol
        data["date"] = data.date.dt.strftime("%Y-%m-%d")

        frames.append(data[["symbol", "date", *OHLCV]])

    pd.concat(frames, ignore_index=True).to_csv(
        data_dir / csv_name,
        index=False,
    )

    profile = settings.get("agents_profile", "gpw_50d.yaml")

    with (output / profile).open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)

    contracts = futures_contracts(
        conn,
        cutoff,
        count=int(settings.get("futures_count", 1)),
    )

    base["days"] = int(settings.get("days", 50))
    base["sim_time"] = int(settings.get("sim_time", 60))
    base["seed"] = int(settings.get("seed", 67))
    base["historical_prices"] = dict(
        enabled=True,
        path=f"market_data/{csv_name}",
        lookback_sessions=lookback,
        cutoff_date=cutoff.isoformat(),
    )

    base["start_date"] = (cutoff + timedelta(days=1)).isoformat()

    simulation_dates = load_simulation_dates(base)
    base["markets"] = build_markets(
        members,
        history,
        index_level,
        contracts=contracts,
        simulation_dates=simulation_dates,
    )

    with (output / yaml_name).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(base, stream, sort_keys=False)

    return output / yaml_name, cutoff, index_level


def index_history(conn, cutoff, sessions=250):
    frame = query(
        conn,
        """
        SELECT TO_DATE("DATE") AS DATE,
               "OPEN", "HIGH", "LOW", "CLOSE"
        FROM INDEX_PRICES
        WHERE "INDEX_SYMBOL" = %s
          AND TO_DATE("DATE") <= %s
        ORDER BY TO_DATE("DATE") DESC
        LIMIT %s
    """,
        ("WIG20", cutoff, sessions),
    )

    if len(frame) != sessions:
        raise ValueError(f"WIG20: expected {sessions} historical observations, got {len(frame)}")

    frame["date"] = pd.to_datetime(frame["date"])

    for column in ("open", "high", "low", "close"):
        frame[column] = frame[column].map(to_price)

    frame["volume"] = 0

    return frame.sort_values("date").set_index("date")


def futures_contracts(conn, cutoff, count=3):
    frame = query(
        conn,
        """
        SELECT CONTRACT_ID, SYMBOL,
               UNDERLYING_SYMBOL, EXPIRY_DATE,
               MULTIPLIER
        FROM DERIVATIVE_CONTRACTS
        WHERE INSTRUMENT_TYPE = 'FUTURE'
          AND UNDERLYING_SYMBOL = 'WIG20'
          AND EXPIRY_DATE > %s
        ORDER BY EXPIRY_DATE, CONTRACT_ID
    """,
        (cutoff,),
    )

    frame = frame.drop_duplicates(subset=["contract_id"]).head(count)

    if len(frame) != count:
        raise ValueError(
            f"Expected {count} WIG20 futures contracts, "
            f"found {len(frame)}. "
            "Check Ikaros contract discovery."
        )

    return frame
