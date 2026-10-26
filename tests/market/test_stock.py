"""Regression tests for Stock(Security), based on feature/ikaros_ge.

Run: python -m pytest tests/market/instruments/test_stock.py -v
"""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from marketsim.market import Price, Security, Stock
from marketsim.market.instruments.stock import Stock as DirectStock


@pytest.fixture
def repository():
    return MagicMock()


@pytest.fixture
def historical_ohlcv():
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [103.0, 104.0],
            "low": [99.0, 100.0],
            "close": [102.0, 103.0],
            "volume": [1000, 1200],
        },
        index=pd.to_datetime(["2026-10-22", "2026-10-23"]),
    )


@pytest.fixture
def stock_factory(tmp_path, repository):
    def make(**kwargs):
        defaults = dict(
            repository=repository,
            symbol="PKO.WA",
            exchange="GPW",
            name="PKO Bank Polski",
            short_name="PKO",
            reference_price=Price(109.54),
            output_dir=tmp_path,
        )
        defaults.update(kwargs)
        return Stock(**defaults)

    return make


def test_stock_export_and_inheritance():
    assert Stock is DirectStock
    assert issubclass(Stock, Security)


def test_stock_is_equity_with_expected_identity(stock_factory):
    stock = stock_factory()
    assert isinstance(stock, Security)
    assert stock.instrument_class == "stock"
    assert stock.symbol == "PKO.WA"
    assert stock.exchange == "GPW"
    assert stock.name == "PKO Bank Polski"
    assert stock.short_name == "PKO"
    assert stock.currency == "PLN"
    assert stock.market_type == "continuous"


def test_stock_reference_and_last_traded_price(stock_factory):
    stock = stock_factory(reference_price=Price(109.54))
    assert stock.reference_price == Price(109.54)
    assert stock.last_traded_price == stock.reference_price


def test_stock_defaults(stock_factory):
    stock = stock_factory(
        symbol=None,
        exchange=None,
        reference_price=None,
        name=None,
        short_name=None,
    )
    assert stock.symbol is None
    assert stock.exchange is None
    assert stock.reference_price == Price(100)
    assert stock.short_name.startswith("stock ")


@pytest.mark.parametrize("currency", ["PLN", "USD", "EUR"])
def test_stock_currency(stock_factory, currency):
    assert stock_factory(currency=currency).currency == currency


@pytest.mark.parametrize("market_type", ["continuous", "discrete"])
def test_stock_market_type(stock_factory, market_type):
    assert stock_factory(market_type=market_type).market_type == market_type


def test_stock_saves_security_once_on_initialization(stock_factory, repository):
    stock = stock_factory()
    repository.save_security.assert_called_once_with(stock)


def test_stock_has_unique_asset_ids(stock_factory):
    a = stock_factory()
    b = stock_factory(symbol="KGH.WA", short_name="KGH")
    assert a.asset_id != b.asset_id


def test_stock_initial_state(stock_factory):
    stock = stock_factory()
    assert stock.status == "active"
    assert stock.eod_status == "closed"
    assert stock.current_day == 0
    assert stock.agents == {}
    assert stock.agent_groups == set()
    assert stock.indices_member == set() or stock.indices_member == {}


def test_stock_keeps_historical_plot_inputs(stock_factory, historical_ohlcv):
    dates = [date(2026, 10, 26), date(2026, 10, 27)]
    stock = stock_factory(
        historical_ohlcv=historical_ohlcv,
        simulation_dates=dates,
    )
    assert stock.historical_ohlcv is historical_ohlcv
    assert stock.simulation_dates is dates
    assert stock.historical_ohlcv.index.max() < pd.Timestamp(dates[0])


def test_stock_plot_inputs_default_to_none(stock_factory):
    stock = stock_factory()
    assert stock.historical_ohlcv is None
    assert stock.simulation_dates is None


def test_stock_can_hold_different_historical_data(stock_factory, historical_ohlcv):
    a = stock_factory(historical_ohlcv=historical_ohlcv)
    b = stock_factory(symbol="KGH.WA")
    assert a.historical_ohlcv is historical_ohlcv
    assert b.historical_ohlcv is None


def test_stock_plotting_inputs_do_not_change_reference_price(stock_factory, historical_ohlcv):
    stock = stock_factory(historical_ohlcv=historical_ohlcv)
    assert stock.reference_price == Price(109.54)


def test_stock_inherits_security_operations(stock_factory):
    stock = stock_factory()
    for method in ("sod", "eod", "step", "add_orders", "show_summary"):
        assert callable(getattr(stock, method))


def test_stock_registers_agents(stock_factory):
    stock = stock_factory()
    agent = MagicMock()
    agent.get_id.return_value = 42
    agent.group = "noise"
    stock.add_agents([agent])
    assert stock.agents[42] is agent
    assert "noise" in stock.agent_groups


def test_stock_default_save_is_not_equity_specific(stock_factory, repository):
    """Documents current inherited persistence behavior pending refactor."""
    stock = stock_factory()
    repository.save_security.assert_called_once_with(stock)


def test_stock_can_be_used_as_option_underlying(stock_factory):
    """Option and other derivatives should accept Stock through Security typing."""
    stock = stock_factory()
    assert isinstance(stock, Security)


def test_stock_is_not_mutated_by_historical_frame_copy(stock_factory, historical_ohlcv):
    frame = historical_ohlcv.copy()
    stock = stock_factory(historical_ohlcv=frame)
    historical_ohlcv.loc[historical_ohlcv.index[0], "close"] = -1
    assert stock.historical_ohlcv.iloc[0]["close"] == 102.0
