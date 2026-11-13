import numpy as np
import pandas as pd
import pytest

from marketsim.analysis.market_realism_analyzer import MarketRealismAnalyzer
from marketsim.analysis.market_series import MarketSeries


def _series(symbol, changes, dates=None):
    if dates is None:
        dates = pd.bdate_range("2025-01-01", periods=len(changes) + 1)
    prices = 100 * np.exp(np.r_[0, np.cumsum(changes)])
    return MarketSeries(
        symbol,
        pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": 1000,
            }
        ),
    )


def test_benchmark_beta_and_correlation():
    bench = _series("WIG20", [0.01, -0.02, 0.03, -0.01, 0.02])
    stock = _series("ALE", [0.02, -0.04, 0.06, -0.02, 0.04])
    score = MarketRealismAnalyzer(stock, stock, bench, bench).scorecard()
    assert score["historical_beta_to_benchmark"] == pytest.approx(2.0)
    assert score["simulated_correlation_to_benchmark"] == pytest.approx(1.0)
    assert score["beta_difference"] == pytest.approx(0.0)
    assert score["short_sample_warning"]


def test_no_benchmark_preserves_legacy_scorecard():
    stock = _series("ALE", [0.01, -0.02, 0.03, -0.01, 0.02])
    score = MarketRealismAnalyzer(stock, stock).scorecard()
    assert "volume_ratio" in score
    assert np.isnan(score["historical_beta_to_benchmark"])
    assert score["historical_benchmark_overlap_days"] == 0


def test_benchmark_requires_pair():
    stock = _series("ALE", [0.01, 0.02, 0.03])
    with pytest.raises(ValueError, match="Both"):
        MarketRealismAnalyzer(stock, stock, historical_benchmark=stock)
