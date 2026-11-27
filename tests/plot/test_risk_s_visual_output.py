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
    market = object.__new__(Security)
    market.instrument_class = "stock"
    market.asset_id = 0
    market.short_name = "TNT"
    assert market.candle_plot_stem() == "stock_0_TNT"


def test_option_candle_plot_names_are_short_and_specific():
    underlying = SimpleNamespace(short_name="TNT")
    market = object.__new__(Security)
    market.instrument_class = "option"
    market.short_name = "PUT TNT 25 105"
    market.underlying = underlying
    market.option_side = "PUT"
    market.strike = 105
    market.expiration = 1
    assert market.candle_plot_stem() == "option_PUT_TNT_105_1Y"
