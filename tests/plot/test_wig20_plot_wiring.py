"""WIG20 plotting and historical-data regression checks."""

from unittest.mock import MagicMock, patch

import pandas as pd

from marketsim.market.instruments.option import Option
from marketsim.plot.daedalus_plot_extensions import combine_candles


def test_option_receives_historical_series(tmp_path):
    history = pd.DataFrame(
        {"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.0], "volume": [100]},
        index=pd.to_datetime(["2026-01-02"]),
    )
    dates = [pd.Timestamp("2026-01-05").date()]
    underlying = MagicMock()
    underlying.last_traded_price = 3000
    underlying.short_name = "WIG20"
    with (
        patch.object(Option, "get_theoretical_price", return_value=10),
        patch(
            "marketsim.market.instruments.security.setup_market_logger", return_value=MagicMock()
        ),
    ):
        option = Option(
            derivatives_config={"strike": 3000, "expiration": 40, "option_side": "CALL"},
            underlying=underlying,
            repository=MagicMock(),
            name="test",
            output_dir=tmp_path,
            historical_ohlcv=history,
            simulation_dates=dates,
        )
    assert option.historical_ohlcv is history
    assert option.simulation_dates == dates
    combined, _ = combine_candles(
        history,
        pd.DataFrame(
            {"day": [0], "open": [11], "high": [12], "low": [10], "close": [11], "volume": [3]}
        ),
        dates,
    )
    assert list(combined.source) == ["historical", "simulated"]


def test_candle_directory_created(tmp_path):
    from marketsim.plot.candle import plot_candlestick

    df = pd.DataFrame({"open": [1], "high": [2], "low": [1], "close": [2], "volume": [4]})
    path = tmp_path / "candle_plots" / "candlestick_test.png"
    plot_candlestick(df, output_file=str(path))
    assert path.is_file()
