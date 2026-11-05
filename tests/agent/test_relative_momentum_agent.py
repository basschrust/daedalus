"""Focused tests for agent history assembly and order sizing."""

import math
from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from marketsim.agent.relative_momentum_agent import RelativeMomentumAgent
from marketsim.agent.strategies.relative_momentum import MomentumSettings


def _agent():
    agent = object.__new__(RelativeMomentumAgent)
    agent.current_day = 0
    agent.agent_id = 123
    agent.base_currency = "PLN"
    agent.settings = MomentumSettings(
        lookback_sessions=3, turnover_sessions=2, min_median_turnover=0, portfolio_size=1
    )
    agent.rebalance_sessions = 1
    agent.configuration = {"aggression": 0.0, "order_validity_ticks": 2}
    agent._simulated_bars = {}
    agent._last_rebalance_day = None
    agent.position = {1: 0}
    agent.get_cash_balance = MagicMock(return_value=Decimal("1000"))
    return agent


def _market():
    from marketsim.market import Price

    start = date(2026, 11, 4)
    hist = pd.DataFrame(
        {
            "date": [start - timedelta(days=i) for i in (3, 2, 1)],
            "close": [10, 11, 12],
            "volume": [100, 100, 100],
        }
    )
    book = SimpleNamespace(get_best_ask=lambda: Price(12), get_best_bid=lambda: Price(12))
    return SimpleNamespace(
        asset_id=1,
        symbol="ABC",
        currency="PLN",
        instrument_class="stock",
        is_tradable=True,
        historical_ohlcv=hist,
        simulation_dates=[start],
        last_traded_price=Price(12),
        order_book=book,
        withdraw_all=MagicMock(),
        add_orders=MagicMock(),
    )


def test_first_day_uses_completed_history_only():
    agent = _agent()
    market = _market()
    market.historical_ohlcv.loc[len(market.historical_ohlcv)] = [date(2026, 11, 4), 1000, 100]
    agent.markets = {1: market}
    histories, _, as_of = agent._histories()
    assert as_of == date(2026, 11, 3)
    assert len(histories["ABC"]) == 3


def test_rebalance_submits_cash_funded_order_only_once():
    agent = _agent()
    market = _market()
    agent.markets = {1: market}
    agent.take_action(0)
    assert market.add_orders.call_count == 1
    order = market.add_orders.call_args.args[0][0]
    assert order.day == 0
    assert order.quantity == 83
    agent.take_action(0)
    assert market.add_orders.call_count == 1


def test_different_currency_is_not_sized_as_base_currency():
    agent = _agent()
    market = _market()
    market.currency = "USD"
    agent.markets = {1: market}
    agent.take_action(0)
    market.add_orders.assert_not_called()


@pytest.mark.parametrize("best_ask", [float("inf"), float("-inf"), float("nan")])
def test_rebalance_falls_back_to_last_price_for_invalid_ask(best_ask):
    agent = _agent()
    market = _market()
    market.order_book.get_best_ask = lambda: best_ask
    agent.markets = {1: market}

    agent.take_action(0)

    market.add_orders.assert_called_once()
    order = market.add_orders.call_args.args[0][0]

    assert math.isfinite(float(order.price))
    assert float(order.price) == 12.0
    assert order.quantity == 83


def test_rebalance_uses_valid_best_ask():
    agent = _agent()
    market = _market()
    market.order_book.get_best_ask = lambda: 12.50
    agent.markets = {1: market}

    agent.take_action(0)

    order = market.add_orders.call_args.args[0][0]

    assert float(order.price) == 12.50
    assert order.quantity == 80


def test_rebalance_falls_back_to_last_price_for_invalid_bid():
    agent = _agent()
    market = _market()
    market.order_book.get_best_bid = lambda: float("-inf")
    agent.position[1] = 100
    agent.get_cash_balance.return_value = Decimal("0")
    agent.markets = {1: market}

    # No selected instruments => sell existing position.
    from unittest.mock import patch

    with patch(
        "marketsim.agent.relative_momentum_agent.equal_weight_targets",
        return_value={"ABC": 0.0},
    ):
        agent.take_action(0)

    market.add_orders.assert_called_once()
    order = market.add_orders.call_args.args[0][0]

    assert math.isfinite(float(order.price))
    assert float(order.price) == 12.0
    assert order.quantity == 100


@pytest.mark.parametrize("day", range(60))
def test_exactly_one_agent_rebalances_each_day(day):
    active_offsets = [offset for offset in range(20) if (day - offset) % 20 == 0]

    assert active_offsets == [day % 20]


@pytest.mark.parametrize("offset", range(20))
def test_agent_rebalances_every_twenty_sessions(offset):
    scheduled_days = [day for day in range(60) if (day - offset) % 20 == 0]

    assert scheduled_days == [
        offset,
        offset + 20,
        offset + 40,
    ]
