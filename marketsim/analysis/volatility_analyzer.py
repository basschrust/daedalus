from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketsim.analysis.market_series import MarketSeries
from marketsim.analysis.statistics import safe_correlation


@dataclass(frozen=True)
class GarchResult:
    omega: float
    alpha: float
    beta: float
    persistence: float
    conditional_volatility: pd.Series


class VolatilityAnalyzer:
    def __init__(self, series: MarketSeries):
        self.series = series

    def summary(self) -> dict:
        returns = self.series.returns

        if len(returns) < 3:
            raise ValueError("At least 4 closing prices are required")

        return {
            "observations": len(returns),
            "mean_daily_return": float(returns.mean()),
            "annualized_volatility": float(returns.std(ddof=1) * np.sqrt(252)),
            "skewness": float(returns.skew()),
            "excess_kurtosis": float(returns.kurt()),
            "return_autocorrelation_1": safe_correlation(
                returns.iloc[1:],
                returns.shift(1).iloc[1:],
            ),
            "squared_return_autocorrelation_1": safe_correlation(
                returns.pow(2).iloc[1:],
                returns.pow(2).shift(1).iloc[1:],
            ),
        }

    def fit_garch(
        self,
        min_observations: int = 250,
    ) -> GarchResult:
        try:
            from arch import arch_model
        except ImportError as exc:
            raise ImportError("GARCH analysis requires the 'arch' package") from exc

        returns = self.series.returns

        if len(returns) < min_observations:
            raise ValueError(
                f"GARCH requires at least {min_observations} returns; received {len(returns)}"
            )

        # Percentage returns improve numerical conditioning.
        percentage_returns = returns * 100

        model = arch_model(
            percentage_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t",
            rescale=False,
        )

        result = model.fit(disp="off")

        if result.convergence_flag != 0:
            raise RuntimeError("GARCH optimization did not converge")

        params = result.params
        alpha = float(params["alpha[1]"])
        beta = float(params["beta[1]"])

        # Convert percentage volatility back to decimal returns.
        conditional_volatility = (result.conditional_volatility / 100).rename("garch_volatility")

        return GarchResult(
            omega=float(params["omega"]),
            alpha=alpha,
            beta=beta,
            persistence=alpha + beta,
            conditional_volatility=conditional_volatility,
        )
