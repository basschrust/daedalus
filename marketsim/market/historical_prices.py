from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_historical_prices(
    config: dict,
    config_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load historical OHLCV snapshots keyed by stock symbol."""

    settings = config.get("historical_prices", {})

    if not settings.get("enabled", False):
        return {}

    path = Path(settings["path"])

    if not path.is_absolute():
        path = config_dir / path

    df = pd.read_csv(path)

    required = {"symbol", "date", *OHLCV_COLUMNS}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Historical OHLCV missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    cutoff = pd.Timestamp(settings["cutoff_date"])
    df = df.loc[df["date"] <= cutoff].copy()

    expected_sessions = settings.get("lookback_sessions")
    allow_short_history = set(settings.get("allow_short_history", []))

    result = {}

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date").drop_duplicates("date", keep="last")

        if expected_sessions is not None:
            group = group.tail(int(expected_sessions))

            if len(group) != int(expected_sessions):
                if symbol not in allow_short_history or group.empty:
                    raise ValueError(
                        f"{symbol}: expected {expected_sessions} "
                        f"historical sessions, got {len(group)}"
                    )

        frame = group.set_index("date")[OHLCV_COLUMNS]

        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column].map(lambda value: Decimal(str(value)))

        frame["volume"] = pd.to_numeric(
            frame["volume"],
            errors="raise",
        )

        if frame.isna().any().any():
            raise ValueError(f"{symbol}: historical OHLCV contains null values")

        result[str(symbol)] = frame

    return result


def load_simulation_dates(config: dict) -> list[date] | None:
    """Generate trading dates, excluding weekends and specified holidays."""

    start_value = config.get("start_date")

    if start_value is None:
        return None

    start = date.fromisoformat(str(start_value))
    days = int(config["days"])

    if days <= 0:
        raise ValueError("days must be positive")

    non_trading_dates = {
        date.fromisoformat(str(value)) for value in config.get("non_trading_dates", [])
    }

    dates = []
    current = start

    while len(dates) < days:
        if current.weekday() < 5 and current not in non_trading_dates:
            dates.append(current)

        current += timedelta(days=1)

    return dates
