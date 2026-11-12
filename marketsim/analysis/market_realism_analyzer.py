from __future__ import annotations

import numpy as np
import pandas as pd

from marketsim.analysis.market_series import MarketSeries
from marketsim.analysis.volatility_analyzer import VolatilityAnalyzer
from marketsim.analysis.volume_analyzer import VolumeAnalyzer


class MarketRealismAnalyzer:
    def __init__(
        self,
        historical: MarketSeries,
        simulated: MarketSeries,
        historical_benchmark: MarketSeries | None = None,
        simulated_benchmark: MarketSeries | None = None,
    ):
        if historical.symbol != simulated.symbol:
            raise ValueError("Historical and simulated symbols must match")
        if (historical_benchmark is None) != (simulated_benchmark is None):
            raise ValueError("Both historical and simulated benchmarks are required")
        if (
            historical_benchmark is not None
            and historical_benchmark.symbol != simulated_benchmark.symbol
        ):
            raise ValueError("Benchmark symbols must match")
        self.historical = historical
        self.simulated = simulated
        self.historical_benchmark = historical_benchmark
        self.simulated_benchmark = simulated_benchmark

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

    @staticmethod
    def _benchmark_statistics(series: MarketSeries, benchmark: MarketSeries | None) -> dict:
        result = {
            "beta_to_benchmark": float("nan"),
            "correlation_to_benchmark": float("nan"),
            "benchmark_overlap_days": 0,
        }
        if benchmark is None:
            return result
        aligned = pd.concat([series.returns, benchmark.returns], axis=1, join="inner").dropna()
        result["benchmark_overlap_days"] = len(aligned)
        if len(aligned) < 3:
            return result
        x = aligned.iloc[:, 0].to_numpy(dtype=float)
        y = aligned.iloc[:, 1].to_numpy(dtype=float)
        variance = float(np.var(y, ddof=1))
        if variance > 0:
            result["beta_to_benchmark"] = float(np.cov(x, y, ddof=1)[0, 1] / variance)
        if np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0:
            result["correlation_to_benchmark"] = float(np.corrcoef(x, y)[0, 1])
        return result

    def scorecard(self) -> dict:
        """Summarize simulated vs historical market behavior.

        Benchmark metrics are computed on intersecting *calendar dates*, never
        by pairing unrelated rows by position. NaN means insufficient data.
        """
        metrics = self.compare().set_index("metric")

        def ratio(metric: str) -> float:
            historical = float(metrics.loc[metric, "historical"])
            simulated = float(metrics.loc[metric, "simulated"])
            if not np.isfinite(historical) or historical == 0:
                return float("nan")
            return simulated / historical

        def difference(metric: str) -> float:
            return float(metrics.loc[metric, "difference"])

        historical_benchmark = self._benchmark_statistics(
            self.historical, self.historical_benchmark
        )
        simulated_benchmark = self._benchmark_statistics(self.simulated, self.simulated_benchmark)
        historical_beta = historical_benchmark["beta_to_benchmark"]
        simulated_beta = simulated_benchmark["beta_to_benchmark"]
        historical_corr = historical_benchmark["correlation_to_benchmark"]
        simulated_corr = simulated_benchmark["correlation_to_benchmark"]
        return {
            "symbol": self.historical.symbol,
            "historical_days": len(self.historical.data),
            "simulated_days": len(self.simulated.data),
            "volatility_ratio": ratio("annualized_volatility"),
            "volume_ratio": ratio("mean_volume"),
            "turnover_ratio": ratio("mean_turnover"),
            "volume_dispersion_ratio": ratio("volume_std"),
            "return_autocorrelation_difference": difference("return_autocorrelation_1"),
            "volatility_clustering_difference": difference("squared_return_autocorrelation_1"),
            "price_volume_correlation_difference": difference("volume_abs_return_correlation"),
            "historical_beta_to_benchmark": historical_beta,
            "simulated_beta_to_benchmark": simulated_beta,
            "beta_difference": simulated_beta - historical_beta,
            "historical_correlation_to_benchmark": historical_corr,
            "simulated_correlation_to_benchmark": simulated_corr,
            "correlation_difference": simulated_corr - historical_corr,
            "historical_benchmark_overlap_days": historical_benchmark["benchmark_overlap_days"],
            "simulated_benchmark_overlap_days": simulated_benchmark["benchmark_overlap_days"],
            "short_sample_warning": len(self.historical.data) < 30 or len(self.simulated.data) < 30,
        }
