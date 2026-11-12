import warnings

import numpy as np
import pandas as pd
import pytest

from marketsim.analysis.market_series import MarketSeries
from marketsim.analysis.statistics import safe_correlation
from marketsim.analysis.volatility_analyzer import VolatilityAnalyzer
from marketsim.analysis.volume_analyzer import VolumeAnalyzer


@pytest.mark.parametrize(
    "x,y,expected",
    [
        ([1, 2, 3, 4], [2, 4, 6, 8], 1.0),
        ([1, 2, 3, 4], [8, 6, 4, 2], -1.0),
        ([1, 2, 3, 4], [1, 0, 0, 1], 0.0),
    ],
)
def test_known_correlations(x, y, expected):
    assert safe_correlation(x, y) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "x,y",
    [
        ([1, 1, 1], [1, 2, 3]),
        ([1, 2, 3], [2, 2, 2]),
        ([1], [2]),
        ([], []),
        ([1, np.nan, np.nan], [2, 3, 4]),
        ([1, np.inf, 3], [1, 2, 3]),
    ],
)
def test_undefined_returns_nan_without_warning(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert np.isnan(safe_correlation(x, y))


def test_pairwise_nan_removal():
    assert safe_correlation([1, 2, np.nan, 4], [2, 4, 100, 8]) == pytest.approx(1)


def test_index_alignment():
    x = pd.Series([1, 2, 3], index=["a", "b", "c"])
    y = pd.Series([30, 10, 20], index=["c", "a", "b"])
    assert safe_correlation(x, y) == pytest.approx(1)


def test_lag_one_matches_pandas():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.04, 0.01, -0.03])
    assert safe_correlation(r.iloc[1:], r.shift(1).iloc[1:]) == pytest.approx(r.autocorr(lag=1))


def series(close, volume):
    return MarketSeries(
        "TEST",
        pd.DataFrame(
            {
                "date": pd.bdate_range("2026-01-05", periods=len(close)),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
            }
        ),
    )


def test_constant_close_volatility_summary():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = VolatilityAnalyzer(series([100] * 8, list(range(8)))).summary()
    assert np.isnan(result["return_autocorrelation_1"])
    assert np.isnan(result["squared_return_autocorrelation_1"])


def test_constant_volume_summary():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = VolumeAnalyzer(series([100, 101, 99, 102, 98, 103, 97, 104], [10] * 8)).summary()
    assert np.isnan(result["volume_abs_return_correlation"])
