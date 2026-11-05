from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MarketSeries:
    symbol: str
    data: pd.DataFrame

    def __post_init__(self) -> None:
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(self.data.columns)

        if missing:
            raise ValueError(f"{self.symbol}: missing OHLCV columns: {sorted(missing)}")

        df = self.data[list(required)].copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")

        for column in required - {"date"}:
            df[column] = pd.to_numeric(df[column], errors="raise")

        df = df.sort_values("date").reset_index(drop=True)

        if df["date"].isna().any() or df["date"].duplicated().any():
            raise ValueError(f"{self.symbol}: invalid or duplicate dates")

        values = df[["open", "high", "low", "close", "volume"]]

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{self.symbol}: non-finite OHLCV values")

        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"{self.symbol}: prices must be positive")

        if (df["volume"] < 0).any():
            raise ValueError(f"{self.symbol}: volume cannot be negative")

        if (
            (df["high"] < df[["open", "close", "low"]].max(axis=1))
            | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        ).any():
            raise ValueError(f"{self.symbol}: inconsistent OHLC values")

        self.data = df

    @property
    def returns(self) -> pd.Series:
        """Daily log returns, indexed by trading date."""
        close = self.data.set_index("date")["close"]
        return np.log(close).diff().dropna().rename("return")

    @property
    def absolute_returns(self) -> pd.Series:
        return self.returns.abs()

    @property
    def squared_returns(self) -> pd.Series:
        return self.returns.pow(2)

    @property
    def volume(self) -> pd.Series:
        return self.data.set_index("date")["volume"].rename("volume")

    @property
    def turnover(self) -> pd.Series:
        """Approximate traded notional in quote currency."""
        df = self.data.set_index("date")
        return (df["close"] * df["volume"]).rename("turnover")

    def rolling_volatility(self, window: int = 20) -> pd.Series:
        """Annualized volatility, assuming 252 trading sessions/year."""
        if window < 2:
            raise ValueError("window must be at least 2")

        return (
            self.returns.rolling(window).std(ddof=1).mul(np.sqrt(252)).rename("realized_volatility")
        )
