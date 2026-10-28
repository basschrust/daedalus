from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from marketsim.agent.momentum import MomentumAgent
from marketsim.market.fourheap.constants import BUY, SELL


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


def test_daily_momentum_uses_historical_prices():
    agent, market = make_agent()
    result = agent.get_daily_momentum(market, lookback=5)
    assert result == pytest.approx(109 / 104 - 1)


def test_daily_momentum_includes_completed_simulated_day():
    agent, market = make_agent()
    market.completed_daily_closes = [120.0]

    result = agent.get_daily_momentum(market, lookback=5)

    assert result == pytest.approx(120 / 105 - 1)


def test_daily_momentum_ignores_future_historical_rows():
    agent, market = make_agent()
    future = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-10-27")],
            "close": [9999.0],
        }
    )
    market.historical_ohlcv = pd.concat(
        [market.historical_ohlcv, future],
        ignore_index=True,
    )

    result = agent.get_daily_momentum(market, lookback=5)

    assert result == pytest.approx(109 / 104 - 1)


def test_daily_mode_trades_from_tick_zero():
    agent, market = make_agent(
        momentum_mode="daily",
        daily_lookback=5,
        threshold=0.01,
    )

    with patch("marketsim.agent.momentum.np.random.poisson", return_value=100):
        agent.take_action(current_time=0)

    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == BUY
    assert order.quantity == 100


def test_intraday_mode_waits_for_lookback():
    agent, market = make_agent(momentum_mode="intraday")

    agent.take_action(current_time=4)

    market.add_orders.assert_not_called()


def test_intraday_mode_buys_on_positive_momentum():
    agent, market = make_agent(momentum_mode="intraday")

    with patch("marketsim.agent.momentum.np.random.poisson", return_value=100):
        agent.take_action(current_time=5)

    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == BUY
    assert float(order.price) >= 110.1


def test_position_limit_prevents_additional_buying():
    agent, market = make_agent(
        momentum_mode="daily",
        daily_lookback=5,
        threshold=0.01,
    )
    agent.position[market.asset_id] = agent.q_max

    agent.take_action(current_time=0)

    market.add_orders.assert_not_called()


@pytest.mark.parametrize("quote", [None, float("inf")])
def test_missing_ask_prevents_buy_order(quote):
    agent, market = make_agent(
        momentum_mode="daily",
        daily_lookback=5,
        threshold=0.01,
    )
    market.order_book.get_best_ask.return_value = quote

    agent.take_action(current_time=0)

    market.add_orders.assert_not_called()


def test_daily_bearish_signal_creates_sell_order():
    agent, market = make_agent(
        momentum_mode="daily",
        daily_lookback=5,
        threshold=0.01,
    )
    market.completed_daily_closes = [90.0]

    with patch("marketsim.agent.momentum.np.random.poisson", return_value=100):
        agent.take_action(current_time=0)

    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == SELL
    assert float(order.price) <= 109.9
