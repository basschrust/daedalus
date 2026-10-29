from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from marketsim.agent import Agent, MomentumAgent
from marketsim.market import Future, FXRateProvider
from marketsim.market.price import Price


def make_agent(**config):
    market = MagicMock()
    market.asset_id = 101
    market.instrument_class = "stock"
    market.last_traded_price = 110.0
    market.order_book.get_best_ask.return_value = 110.1
    market.order_book.get_best_bid.return_value = 109.9

    market.traded_prices = {
        0: {"close": 100.0},
        5: {"close": 105.0},
    }

    market.historical_ohlcv = pd.DataFrame(
        {
            "date": pd.date_range("2026-10-12", periods=10, freq="D"),
            "close": np.arange(100.0, 110.0),
        }
    )
    market.simulation_dates = [date(2026, 10, 26)]
    market.completed_daily_closes = []

    agent = MomentumAgent(
        markets=[market],
        repository=MagicMock(),
        configuration={
            "lam": 1.0,
            "period": 5,
            "threshold": 0.001,
            "mean_volume": 100,
            "q_max": 1000,
            "aggression": 0.003,
            **config,
        },
    )
    return agent, market


def test_collateral_transfer_preserves_equity(
    futures_agents: dict[str, Agent],
    future_short: Future,
):
    agent = futures_agents["buyer"]
    agent.cash = {"PLN": Price(100_000)}

    equity_before = agent.cash["PLN"]

    agent.update_futures_position(
        quantity=2,
        contract=future_short,
    )

    values = agent.futures_value[future_short.asset_id]

    equity_after = (
        agent.get_cash_balance(currency="PLN") + values["collateral"] + values["unsettled_pnl"]
    )

    assert equity_after == equity_before


def test_futures_collateral_uses_decimal():
    position = 2000
    price = Price("3000.00")
    multiplier = 20
    margin_rate = Decimal("0.10")

    collateral = Decimal(position) * Decimal(str(price)) * Decimal(multiplier) * margin_rate

    assert collateral == Decimal("12000000.00")
    assert isinstance(collateral, Decimal)


def test_large_portfolio_value_is_not_limited_by_price():
    agent, contract = make_agent()

    agent.fx_rate_provider = FXRateProvider()
    agent.base_currency = "PLN"

    # Isolate the test from existing positions.
    agent.position = {asset_id: 0 for asset_id in agent.position}
    agent.futures_position = {asset_id: 0 for asset_id in agent.futures_position}
    agent.futures_value = {}

    agent.cash = {"PLN": Decimal("-101230431.15")}

    value = agent.calculate_portfolio_value(day=0)

    assert value == Decimal("-101230431.15")
    assert isinstance(value, Decimal)


def test_large_portfolio_value_is_not_limited_by_price_2():
    agent, contract = make_agent()

    agent.position = {asset_id: 0 for asset_id in agent.position}
    agent.futures_position = {asset_id: 0 for asset_id in agent.futures_position}
    agent.futures_value = {}

    agent.cash = {"PLN": Decimal("-101230431.15")}

    value = agent.calculate_portfolio_value(day=0)

    assert value == Decimal("-101230431.15")
    assert isinstance(value, Decimal)
