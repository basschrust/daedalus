from datetime import date
from types import SimpleNamespace

import pandas as pd

from marketsim.analysis.report import (
    MarketRealismReport,
    daily_ohlcv,
)


def test_daily_ohlcv_aggregation():
    intraday = pd.DataFrame(
        {
            "time_tick": [0, 1, 2],
            "open": [100, 100, 102],
            "high": [100, 103, 104],
            "low": [100, 99, 101],
            "close": [100, 102, 103],
            "volume": [0, 20, 30],
        }
    )

    result = daily_ohlcv(intraday, date(2026, 11, 5))

    assert result["open"] == 100
    assert result["high"] == 104
    assert result["low"] == 99
    assert result["close"] == 103
    assert result["volume"] == 50


def test_collect_day_uses_existing_market_data():
    market = SimpleNamespace(
        instrument_class="stock",
        symbol="TEST",
        traded_prices_df=pd.DataFrame(
            {
                "time_tick": [0, 1],
                "open": [100, 100],
                "high": [100, 102],
                "low": [100, 100],
                "close": [100, 102],
                "volume": [0, 25],
            }
        ),
    )

    simulator = SimpleNamespace(
        markets={1: market},
        simulation_dates=[date(2026, 11, 5)],
    )

    report = MarketRealismReport()
    report.collect_day(simulator, day=0)

    assert len(report.observations["TEST"]) == 1
    assert report.observations["TEST"][0]["volume"] == 25
    assert report.observations["TEST"][0]["close"] == 102
