import numpy as np
import pandas as pd
import pytest

from marketsim.analysis.market_realism_analyzer import MarketRealismAnalyzer
from marketsim.analysis.market_series import MarketSeries


def test_realism_scorecard_identical_series():
    rng = np.random.default_rng(67)
    n = 100

    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))

    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1000, 10000, n),
        }
    )

    historical = MarketSeries("TEST", frame.copy())
    simulated = MarketSeries("TEST", frame.copy())

    score = MarketRealismAnalyzer(historical, simulated).scorecard()

    for metric in (
        "volatility_ratio",
        "volume_ratio",
        "turnover_ratio",
        "volume_dispersion_ratio",
    ):
        assert score[metric] == pytest.approx(1.0)

    for metric in (
        "return_autocorrelation_difference",
        "volatility_clustering_difference",
        "price_volume_correlation_difference",
    ):
        assert score[metric] == pytest.approx(0.0)
