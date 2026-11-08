from __future__ import annotations

import pandas as pd

from marketsim.analysis.market_series import MarketSeries
from marketsim.analysis.volatility_analyzer import VolatilityAnalyzer
from marketsim.analysis.volume_analyzer import VolumeAnalyzer


class MarketRealismAnalyzer:
    def __init__(
        self,
        historical: MarketSeries,
        simulated: MarketSeries,
    ):
        if historical.symbol != simulated.symbol:
            raise ValueError("Historical and simulated symbols must match")

        self.historical = historical
        self.simulated = simulated

    @staticmethod
    def _metrics(series: MarketSeries) -> dict:
        return {
            **VolatilityAnalyzer(series).summary(),
            **VolumeAnalyzer(series).summary(),
        }

    def compare(self) -> pd.DataFrame:
        historical = self._metrics(self.historical)
        simulated = self._metrics(self.simulated)

        rows = []

        for metric in historical:
            historical_value = historical[metric]
            simulated_value = simulated[metric]

            rows.append(
                {
                    "symbol": self.historical.symbol,
                    "metric": metric,
                    "historical": historical_value,
                    "simulated": simulated_value,
                    "difference": simulated_value - historical_value,
                }
            )

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        self.compare().to_csv(path, index=False)
