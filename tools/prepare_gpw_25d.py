
"""Prepare a 15-day GPW simulation using Snowflake historical market data.

Run from the Daedalus project root:

    python -m tools.prepare_gpw_25d

Credentials:
    SNOWFLAKE_ACCOUNT
    SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD

No Ikaros dependencies.
Snowflake access is read-only.
"""

from __future__ import annotations

import argparse
import math
import os
from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


SYMBOLS = ("PKO.WA", "PKN.WA", "KGH.WA", "ALE.WA", "PCO.WA", "TPE.WA",)

CUTOFF = date(2026, 10, 27)

# 15 GPW trading sessions after October 27.
# November 11 is a Polish exchange holiday.
SIMULATION_DATES = (
    "2026-10-28",
    "2026-10-29",
    "2026-10-30",
    "2026-11-02",
    "2026-11-03",
    "2026-11-04",
    "2026-11-05",
    "2026-11-06",
    "2026-11-09",
    "2026-11-10",
    "2026-11-12",
    "2026-11-13",
    "2026-11-16",
    "2026-11-17",
    "2026-11-18",
    "2026-11-19",
    "2026-11-20",
    "2026-11-23",
    "2026-11-24",
    "2026-11-25",
    "2026-11-26",
    "2026-11-27",
    "2026-11-30",
    "2026-12-01",
    "2026-12-02",
)


SQL = """
SELECT
    "SYMBOL",
    TO_DATE("DATE") AS SESSION_DATE,
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "VOLUME"
FROM "PRICES"
WHERE
    "SYMBOL" = %s
    AND TO_DATE("DATE") <= %s
ORDER BY TO_DATE("DATE") DESC
LIMIT 250
"""


def fetch_history(connection, symbol: str) -> pd.DataFrame:
    """Fetch and validate 250 historical OHLCV sessions."""

    with closing(connection.cursor()) as cursor:
        cursor.execute(
            SQL,
            (symbol, CUTOFF.isoformat()),
        )
        rows = cursor.fetchall()

    frame = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ],
    )

    if len(frame) != 250:
        raise ValueError(
            f"{symbol}: expected 250 sessions, got {len(frame)}"
        )

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    if frame["date"].duplicated().any():
        raise ValueError(
            f"{symbol}: duplicate historical dates"
        )

    last = frame["date"].iloc[-1].date()

    if last != CUTOFF:
        raise ValueError(
            f"{symbol}: latest session is {last}, "
            f"expected {CUTOFF}"
        )

    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    frame[numeric] = frame[numeric].apply(
        pd.to_numeric,
        errors="raise",
    )

    if frame[numeric].isna().any().any():
        raise ValueError(
            f"{symbol}: incomplete OHLCV"
        )

    if not frame[numeric].map(math.isfinite).all().all():
        raise ValueError(
            f"{symbol}: nonfinite OHLCV"
        )

    if not (
        frame[["open", "high", "low", "close"]] > 0
    ).all().all():
        raise ValueError(
            f"{symbol}: nonpositive OHLC prices"
        )

    if not (frame["volume"] >= 0).all():
        raise ValueError(
            f"{symbol}: negative volume"
        )

    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    return frame


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--daedalus-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--database",
        default="DAEDALUS",
    )

    parser.add_argument(
        "--schema",
        default="MARKET_DATA",
    )

    parser.add_argument(
        "--warehouse",
        default="DAEDALUS_WH",
    )

    parser.add_argument(
        "--role",
        default=None,
    )

    parser.add_argument(
        "--authenticator",
        default="snowflake",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--source-yaml",
        default="gpw_20d_auto.yaml",
    )

    parser.add_argument(
        "--output-yaml",
        default="gpw_25d_auto.yaml",
    )

    args = parser.parse_args()

    # Direct connection to Snowflake.
    # Do not use Daedalus's simulation repository here:
    # that repository creates simulation schemas and tables.
    import snowflake.connector

    required = (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
    )

    missing = [
        name
        for name in required
        if not os.getenv(name)
    ]

    if (
        args.authenticator == "snowflake"
        and not os.getenv("SNOWFLAKE_PASSWORD")
    ):
        missing.append("SNOWFLAKE_PASSWORD")

    if missing:
        parser.error(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    def identifier(value: str) -> str:
        if (
            not value
            or not value.replace("_", "").isalnum()
            or not value[0].isalpha()
        ):
            parser.error(
                f"Invalid Snowflake identifier: {value!r}"
            )

        return value.upper()

    database = identifier(args.database)
    schema = identifier(args.schema)
    warehouse = identifier(args.warehouse)

    connection_args = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
        "authenticator": args.authenticator,
    }

    if args.authenticator == "snowflake":
        connection_args["password"] = os.environ[
            "SNOWFLAKE_PASSWORD"
        ]

    if args.role:
        connection_args["role"] = args.role

    input_dir = (
        args.daedalus_root.resolve()
        / "marketsim"
        / "input"
    )

    source_path = input_dir / args.source_yaml

    if not source_path.is_file():
        parser.error(
            f"Source scenario not found: {source_path}"
        )

    with source_path.open(
        encoding="utf-8"
    ) as stream:
        scenario = yaml.safe_load(stream)

    frames = []

    connection = snowflake.connector.connect(
        **connection_args
    )

    try:
        for symbol in SYMBOLS:
            frame = fetch_history(
                connection,
                symbol,
            )

            frames.append(frame)

            print(
                f"{symbol}: "
                f"{len(frame)} sessions, "
                f"last={frame['date'].iloc[-1]}, "
                f"mean volume="
                f"{frame['volume'].mean():,.0f}, "
                f"last close="
                f"{frame['close'].iloc[-1]:.2f}"
            )

    finally:
        connection.close()

    all_history = pd.concat(
        frames,
        ignore_index=True,
    ).sort_values(
        ["symbol", "date"]
    )

    # Keep Daedalus's existing YAML structure.
    scenario["days"] = len(SIMULATION_DATES)
    scenario["sim_time"] = 50
    scenario["simulation_dates"] = list(SIMULATION_DATES)

    scenario["historical_prices"] = {
        "enabled": True,
        "path": "market_data/gpw_25d_auto.ohlcv.csv",
        "lookback_sessions": 250,
        "cutoff_date": CUTOFF.isoformat(),
    }

    markets = scenario.setdefault(
        "markets",
        {},
    )

    # Add Allegro and Pepco if they are not already in the source YAML.
    history_by_symbol = {
        frame["symbol"].iloc[0]: frame
        for frame in frames
    }

    for symbol in ("ALE.WA", "PCO.WA", "TPE.WA"):
        key = symbol.lower()

        markets[key] = {
            "name": symbol,
            "short_name": symbol,
            "symbol": symbol,
            "instrument_class": "stock",
            "market_type": "continuous",
            "reference_price": float(
                history_by_symbol[symbol]["close"].iloc[-1]
            ),
            "currency": "PLN",
        }

    # Update reference prices for all five assets.
    for market in markets.values():
        symbol = market.get("symbol")

        if symbol in SYMBOLS:
            hist = next(
                f
                for f in frames
                if f["symbol"].iloc[0] == symbol
            )

            market["reference_price"] = float(
                hist["close"].iloc[-1]
            )

    # Preserve dynamic market-maker sizing.
    mm = scenario["agents"]["market_makers"]["config"]

    mm.pop("volume", None)

    csv_path = input_dir / "market_data" / "gpw_25d_auto.ohlcv.csv"

    yaml_path = (
        input_dir
        / args.output_yaml
    )

    if (
        not args.overwrite
        and (
            csv_path.exists()
            or yaml_path.exists()
        )
    ):
        parser.error(
            "Output exists; pass --overwrite "
            "to regenerate it"
        )

    # Write only after all four assets pass validation.
    all_history.to_csv(
        csv_path,
        index=False,
    )

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(
            scenario,
            stream,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"Created {csv_path}")
    print(f"Created {yaml_path}")

    print(
        f"{len(SIMULATION_DATES)} simulation sessions:",
        ", ".join(SIMULATION_DATES),
    )


if __name__ == "__main__":
    main()
