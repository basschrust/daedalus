import numpy as np
import pandas as pd
import pytest

from marketsim.analysis.market_realism_analyzer import MarketRealismAnalyzer
from marketsim.analysis.market_series import MarketSeries
from marketsim.analysis.volatility_analyzer import VolatilityAnalyzer
from marketsim.analysis.volume_analyzer import VolumeAnalyzer


@pytest.fixture
def market_series():
    n = 300
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(0.001 + 0.01 * np.sin(np.arange(n))))

    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1000),
        }
    )

    return MarketSeries("TEST", frame)


def test_log_returns(market_series):
    returns = market_series.returns

    assert len(returns) == 299
    assert np.isfinite(returns).all()


def test_rolling_volatility(market_series):
    volatility = market_series.rolling_volatility(20)

    assert volatility.notna().sum() == 280
    assert (volatility.dropna() >= 0).all()


def test_volatility_summary(market_series):
    summary = VolatilityAnalyzer(market_series).summary()

    assert summary["observations"] == 299
    assert summary["annualized_volatility"] > 0


def test_volume_summary(market_series):
    summary = VolumeAnalyzer(market_series).summary()

    assert summary["mean_volume"] == 1000
    assert summary["zero_volume_fraction"] == 0


def test_market_realism_comparison(market_series):
    analyzer = MarketRealismAnalyzer(
        historical=market_series,
        simulated=market_series,
    )

    result = analyzer.compare()

    assert not result.empty
    assert np.allclose(
        result["difference"].dropna(),
        0,
    )


def test_rejects_negative_volume(market_series):
    frame = market_series.data.copy()
    frame.loc[0, "volume"] = -1

    with pytest.raises(ValueError, match="volume"):
        MarketSeries("TEST", frame)
