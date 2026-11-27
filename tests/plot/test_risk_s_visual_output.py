"""Regression tests for the default option labels and the flat LOB directory."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from marketsim.market.instruments.security import Security


def test_default_option_names_agree_with_contract_types():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "input" / "market_structure.yaml").read_text())
    markets = config["markets"]
    call = markets["security_2"]
    put = markets["put_on_tnt"]
    assert call["derivatives_config"]["option_side"] == "CALL"
    assert "CALL" in call["name"] and "PUT" not in call["name"]
    assert "90" in call["name"]
    assert put["derivatives_config"]["option_side"] == "PUT"
    assert "PUT" in put["name"] and "CALL" not in put["name"]
    assert all(str(strike) in put["name"] for strike in put["derivatives_config"]["strikes"])


def test_lob_plots_share_one_dir_and_day_tick_prevent_overwrites(tmp_path):
    market = object.__new__(Security)
    market.is_tradable = True
    market.asset_id = 17
    market.current_day = 3
    market.output_dir = tmp_path
    market.short_name = "PUT TNT 25 100"
    market.name = "An option with a very long description containing spaces"
    market.logger = Mock()
    market.order_book = SimpleNamespace(
        buy_unmatched=SimpleNamespace(heap=[]),
        sell_unmatched=SimpleNamespace(heap=[]),
    )
    market.aggregate_order_queue = Mock(side_effect=[{100: 5}, {101: 4}])

    with patch("marketsim.market.instruments.security.plot_order_book") as plot:
        market.plot_lob(current_time=25)
    plot.assert_called_once()
    filename = Path(plot.call_args.kwargs["output_file"])
    assert filename == tmp_path / "LOB" / "lob_17_d003_t0025.png"
    assert " " not in filename.name
    assert "PUT TNT" not in str(filename.parent)

    market.current_day = 4
    market.aggregate_order_queue = Mock(side_effect=[{100: 5}, {101: 4}])
    with patch("marketsim.market.instruments.security.plot_order_book") as plot_next:
        market.plot_lob(current_time=25)
    assert Path(plot_next.call_args.kwargs["output_file"]) != filename


def test_stock_candle_plot_names_are_short_and_specific():
    stock = object.__new__(Security)
    stock.instrument_class = "stock"
    stock.asset_id = 0
    stock.symbol = "TNT"
    stock.short_name = "stock 0"
    assert stock.candle_plot_stem() == "0000_stock_TNT"


def test_option_filename_uses_real_contract_side_strike_and_days():
    from decimal import Decimal

    stock = SimpleNamespace(symbol="TNT", short_name="stock 0")
    option = object.__new__(Security)
    option.instrument_class = "option"
    option.asset_id = 1
    option.short_name = "PUT TNT"  # Deliberately misleading label
    option.underlying = stock
    option.option_side = "CALL"
    option.strike = Decimal("105.00")
    option.expiration = 20
    assert option.candle_plot_stem() == "0001_option_TNT_CALL_105_20D"
    option.expiration = 248
    option.option_side = "PUT"
    assert option.candle_plot_stem() == "0001_option_TNT_PUT_105_1Y"


def test_future_filename_and_sorting():
    stock = object.__new__(Security)
    stock.instrument_class = "stock"
    stock.asset_id = 0
    stock.symbol = "TNT"
    stock.short_name = "stock 0"

    future = object.__new__(Security)
    future.instrument_class = "future"
    future.asset_id = 2
    future.underlying = stock
    future.expiration = 20

    assert sorted([future.candle_plot_stem(), stock.candle_plot_stem()]) == [
        "0000_stock_TNT",
        "0002_future_TNT_20D",
    ]


def test_legacy_derivative_candle_file_is_renamed_by_summary(tmp_path, monkeypatch):
    # The Option/Future overrides still render theoretical-price curves under
    # their old paths. Security.show_summary normalizes the final filename.
    market = object.__new__(Security)
    market.output_dir = tmp_path
    market.instrument_class = "option"
    market.asset_id = 1
    market.underlying = SimpleNamespace(symbol="TNT")
    market.option_side = "PUT"
    market.strike = 105
    market.expiration = 20
    market.short_name = "PUT TNT 105"
    market.historical_ohlcv = None
    market.simulation_dates = None
    market.logger = Mock()
    market.last_traded_price = 12
    market.repository = Mock()
    market.repository.get_eod_prices.return_value = __import__("pandas").DataFrame(
        [{"day": 0, "open": 12, "high": 13, "low": 11, "close": 12, "volume": 1}]
    )

    def legacy_plot(*, traded_prices):
        directory = tmp_path / "candle_plots"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"candlestick_{market}.png").write_bytes(b"valid legacy plot")

    monkeypatch.setattr(market, "plot_history", legacy_plot)
    market.show_summary()
    assert (tmp_path / "candle_plots" / "0001_option_TNT_PUT_105_20D.png").is_file()
    assert not (tmp_path / "candle_plots" / f"candlestick_{market}.png").exists()


def test_index_candle_plot_name_uses_zero_padded_id():
    market = object.__new__(Security)
    market.instrument_class = "index"
    market.asset_id = 7
    market.short_name = "WIG20"
    market.symbol = "WIG20"
    assert market.candle_plot_stem() == "0007_index_WIG20"
