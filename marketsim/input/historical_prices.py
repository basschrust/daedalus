from __future__ import annotations

from datetime import date
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

    result = {}

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date").drop_duplicates("date", keep="last")

        if expected_sessions is not None:
            group = group.tail(int(expected_sessions))

            if len(group) != int(expected_sessions):
                raise ValueError(
                    f"{symbol}: expected {expected_sessions} historical sessions, got {len(group)}"
                )

        frame = group.set_index("date")[OHLCV_COLUMNS]

        for column in OHLCV_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="raise",
            )

        if frame.isna().any().any():
            raise ValueError(f"{symbol}: historical OHLCV contains null values")

        result[str(symbol)] = frame

    return result


def load_simulation_dates(config: dict) -> list[date] | None:
    """Read explicit simulated trading dates from scenario YAML."""

    values = config.get("simulation_dates")

    if values is None:
        return None

    dates = [date.fromisoformat(str(value)) for value in values]

    if len(dates) != config["days"]:
        raise ValueError(f"Expected {config['days']} simulation dates, got {len(dates)}")

    if dates != sorted(set(dates)):
        raise ValueError("Simulation dates must be unique and increasing")

    return dates
