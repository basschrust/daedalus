from __future__ import annotations

import numpy as np

from marketsim.analysis.market_series import MarketSeries


class VolumeAnalyzer:
    def __init__(self, series: MarketSeries):
        self.series = series

    def summary(self) -> dict:
        volume = self.series.volume
        turnover = self.series.turnover
        absolute_returns = self.series.absolute_returns

        aligned = volume.to_frame().join(
            absolute_returns.rename("absolute_return"),
            how="inner",
        )

        correlation = (
            aligned["volume"].corr(aligned["absolute_return"])
            if len(aligned) >= 3
            and aligned["volume"].nunique() > 1
            and aligned["absolute_return"].nunique() > 1
            else np.nan
        )

        return {
            "observations": len(volume),
            "mean_volume": float(volume.mean()),
            "median_volume": float(volume.median()),
            "volume_std": float(volume.std(ddof=1)),
            "mean_turnover": float(turnover.mean()),
            "median_turnover": float(turnover.median()),
            "zero_volume_fraction": float((volume == 0).mean()),
            "volume_abs_return_correlation": float(correlation),
        }

    def rolling_volume(self, window: int = 20):
        if window < 1:
            raise ValueError("window must be positive")

        return self.series.volume.rolling(window).mean()
