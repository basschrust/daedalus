"""Tests for 250 historical + 10 simulated candles. Run with pytest -v."""

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from marketsim.market.historical_prices import load_simulation_dates
from marketsim.plot.daedalus_plot_extensions import combine_candles, plot_combined_candles


@pytest.fixture
def historical():
    idx = pd.bdate_range(end="2026-10-23", periods=250)
    return pd.DataFrame(
        {
            "open": [100 + i / 10 for i in range(250)],
            "high": [101 + i / 10 for i in range(250)],
            "low": [99 + i / 10 for i in range(250)],
            "close": [100.5 + i / 10 for i in range(250)],
            "volume": [1000 + i for i in range(250)],
        },
        index=idx,
    )


@pytest.fixture
def dates():
    return pd.to_datetime(
        [
            "2026-10-26",
            "2026-10-27",
            "2026-10-28",
            "2026-10-29",
            "2026-10-30",
            "2026-11-02",
            "2026-11-03",
            "2026-11-04",
            "2026-11-05",
            "2026-11-06",
        ]
    ).date.tolist()


@pytest.fixture
def simulated():
    return pd.DataFrame(
        {
            "day": range(10),
            "open": [125 + i for i in range(10)],
            "high": [126 + i for i in range(10)],
            "low": [124 + i for i in range(10)],
            "close": [125.5 + i for i in range(10)],
            "volume": [200 + i for i in range(10)],
        }
    )


def test_250_plus_10(historical, simulated, dates):
    combined, first = combine_candles(historical, simulated, dates)
    assert len(combined) == 260
    assert combined.source.value_counts().to_dict() == {"historical": 250, "simulated": 10}
    assert first == pd.Timestamp("2026-10-26")
    assert combined.index.max() == pd.Timestamp("2026-11-06")


def test_day_mapping(historical, simulated, dates):
    combined, _ = combine_candles(historical, simulated, dates)
    assert combined.loc["2026-10-26", "close"] == simulated.loc[0, "close"]
    assert combined.loc["2026-11-06", "close"] == simulated.loc[9, "close"]


def test_historical_values_preserved(historical, simulated, dates):
    combined, _ = combine_candles(historical, simulated, dates)
    pd.testing.assert_frame_equal(
        combined.loc[historical.index, list(historical.columns)],
        historical,
        check_freq=False,
    )


def test_inputs_not_mutated(historical, simulated, dates):
    h, s = historical.copy(deep=True), simulated.copy(deep=True)
    combine_candles(historical, simulated, dates)
    pd.testing.assert_frame_equal(h, historical)
    pd.testing.assert_frame_equal(s, simulated)


def test_date_column_accepted(historical, simulated, dates):
    combined, _ = combine_candles(historical.reset_index(names="date"), simulated, dates)
    assert len(combined) == 260


@pytest.mark.parametrize(
    "bad_dates",
    [
        ["2026-10-23"],
        ["2026-10-26"] * 10,
        ["2026-10-27", "2026-10-26"],
        [],
    ],
)
def test_invalid_calendar_rejected(historical, simulated, bad_dates):
    with pytest.raises(ValueError):
        combine_candles(historical, simulated, bad_dates)


@pytest.mark.parametrize(
    "bad_days",
    [
        list(range(1, 11)),
        [0] * 10,
        list(range(9)),
    ],
)
def test_invalid_day_mapping_rejected(historical, simulated, dates, bad_days):
    frame = simulated.iloc[: len(bad_days)].copy()
    frame["day"] = bad_days
    with pytest.raises(ValueError):
        combine_candles(historical, frame, dates)


def test_duplicate_historical_date_rejected(historical, simulated, dates):
    with pytest.raises(ValueError):
        combine_candles(pd.concat([historical, historical.tail(1)]), simulated, dates)


def test_unsorted_historical_date_rejected(historical, simulated, dates):
    with pytest.raises(ValueError):
        combine_candles(historical.iloc[::-1], simulated, dates)


@pytest.mark.parametrize("side,col", [("historical", "open"), ("simulated", "volume")])
def test_missing_ohlcv_column_rejected(historical, simulated, dates, side, col):
    if side == "historical":
        historical = historical.drop(columns=col)
    else:
        simulated = simulated.drop(columns=col)
    with pytest.raises(ValueError, match="missing columns"):
        combine_candles(historical, simulated, dates)


def test_empty_history_rejected(simulated, dates):
    with pytest.raises(ValueError):
        combine_candles(
            pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            simulated,
            dates,
        )


def test_plot_png_created(tmp_path, historical, simulated, dates):
    out = tmp_path / "nested" / "stock.png"
    plot_combined_candles(historical, simulated, dates, out, "PKO")
    assert out.exists() and out.stat().st_size > 5000


def test_plot_boundary_and_grey_background(tmp_path, historical, simulated, dates):
    captured = []
    real_close = plt.close

    def capture(fig):
        captured.append(fig)
        real_close(fig)

    with patch("marketsim.plot.daedalus_plot_extensions.plt.close", side_effect=capture):
        plot_combined_candles(historical, simulated, dates, tmp_path / "chart.png")

    price, volume = captured[0].axes
    boundary = mdates.date2num(pd.Timestamp("2026-10-24 12:00"))
    for axis in (price, volume):
        dashed = [line for line in axis.lines if line.get_linestyle() == "--"]
        assert len(dashed) == 1
        assert float(dashed[0].get_xdata()[0]) == pytest.approx(boundary)
        assert axis.patches  # historical shaded region, plus candle bodies/bars
    assert any("Simulation starts" in text.get_text() for text in price.texts)


def test_stock_receives_history(tmp_path, historical, dates):
    from marketsim.market import Stock

    repo = MagicMock()
    stock = Stock(
        repository=repo,
        symbol="PKO.WA",
        exchange="GPW",
        historical_ohlcv=historical,
        simulation_dates=dates,
        output_dir=tmp_path,
    )
    assert stock.historical_ohlcv is historical
    assert stock.simulation_dates is dates
    repo.save_security.assert_called_once_with(stock)


def test_snapshot_loader(tmp_path):
    loader = pytest.importorskip(
        "marketsim.input.historical_prices",
        reason="Historical snapshot loader has not been added yet",
    )
    rows = [
        {
            "symbol": symbol,
            "date": day,
            "open": 100,
            "high": 102,
            "low": 99,
            "close": 101,
            "volume": 1000,
        }
        for symbol in ("PKO.WA", "KGH.WA")
        for day in ("2026-10-22", "2026-10-23")
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "history.csv", index=False)
    config = {
        "historical_prices": {
            "enabled": True,
            "path": "history.csv",
            "cutoff_date": "2026-10-23",
            "lookback_sessions": 2,
        }
    }
    frames = loader.load_historical_prices(config, tmp_path)
    assert set(frames) == {"PKO.WA", "KGH.WA"}
    assert all(len(f) == 2 for f in frames.values())


def test_snapshot_loader_rejects_short_history(tmp_path):
    loader = pytest.importorskip("marketsim.input.historical_prices")
    pd.DataFrame(
        [
            {
                "symbol": "PKO.WA",
                "date": "2026-10-23",
                "open": 100,
                "high": 102,
                "low": 99,
                "close": 101,
                "volume": 1000,
            }
        ]
    ).to_csv(tmp_path / "history.csv", index=False)
    config = {
        "historical_prices": {
            "enabled": True,
            "path": "history.csv",
            "cutoff_date": "2026-10-23",
            "lookback_sessions": 2,
        }
    }
    with pytest.raises(ValueError, match="expected 2"):
        loader.load_historical_prices(config, tmp_path)


def test_calendar_loader_generates_trading_days():
    config = {
        "days": 5,
        "start_date": "2026-11-09",
        "non_trading_dates": ["2026-11-11"],
    }

    dates = load_simulation_dates(config)

    assert dates == [
        date(2026, 11, 9),
        date(2026, 11, 10),
        date(2026, 11, 12),
        date(2026, 11, 13),
        date(2026, 11, 16),
    ]
