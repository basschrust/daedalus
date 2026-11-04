"""Exchange-agnostic, point-in-time 52-week-range momentum selection.

Only completed daily OHLCV bars are accepted. No database dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class MomentumSettings:
    lookback_sessions: int = 252
    turnover_sessions: int = 30
    min_median_turnover: float = 250_000.0
    portfolio_size: int = 10

    def __post_init__(self) -> None:
        if min(self.lookback_sessions, self.turnover_sessions, self.portfolio_size) < 1:
            raise ValueError("Window lengths and portfolio size must be positive")
        if self.min_median_turnover < 0:
            raise ValueError("Minimum turnover cannot be negative")


def rank_momentum(
    histories: Mapping[str, pd.DataFrame],
    *,
    as_of: date,
    settings: MomentumSettings = MomentumSettings(),
) -> pd.DataFrame:
    """Rank symbols using completed bars dated <= as_of.

    Each frame needs date, close, volume. `as_of` is the last completed
    session, not the session in which orders are being placed.
    Turnover is measured in each instrument's quote currency; convert
    currencies upstream before applying a common threshold.
    """
    rows = []
    for symbol, frame in histories.items():
        required = {"date", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{symbol}: missing columns {sorted(required - set(frame.columns))}")
        data = frame.loc[:, ["date", "close", "volume"]].copy()
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        data = data.loc[data["date"].dt.date <= as_of].sort_values("date")
        if data["date"].duplicated().any():
            raise ValueError(f"{symbol}: duplicate session dates")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
        if len(data) < max(settings.lookback_sessions, settings.turnover_sessions):
            continue
        data = data.tail(max(settings.lookback_sessions, settings.turnover_sessions))
        if (
            data["close"].isna().any()
            or (data["close"] <= 0).any()
            or data["volume"].isna().any()
            or (data["volume"] < 0).any()
        ):
            continue
        close = data["close"].tail(settings.lookback_sessions)
        low, high = float(close.min()), float(close.max())
        if high <= low:
            continue
        turnover = (data["close"] * data["volume"]).tail(settings.turnover_sessions)
        median_turnover = float(turnover.median())
        if median_turnover < settings.min_median_turnover:
            continue
        rows.append(
            {
                "symbol": symbol,
                "score": (float(close.iloc[-1]) - low) / (high - low),
                "median_turnover": median_turnover,
                "last_close": float(close.iloc[-1]),
            }
        )
    columns = ["symbol", "score", "median_turnover", "last_close"]
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["score", "symbol"], ascending=[False, True])
        .head(settings.portfolio_size)
        .reset_index(drop=True)
    )


def equal_weight_targets(ranking: pd.DataFrame) -> dict[str, float]:
    """Cash-only if nothing qualifies; otherwise equally weight selected names."""
    n = len(ranking)
    return {symbol: 1.0 / n for symbol in ranking["symbol"]} if n else {}
