from unittest.mock import MagicMock

import pytest

from marketsim.agent.regime_trader import RegimeTraderAgent
from marketsim.market.fourheap.constants import BUY, SELL


def _agent(fundamental, price=100.0, **config):
    market = MagicMock()
    market.asset_id = 101
    market.instrument_class = "stock"
    market.symbol = "TEST"
    market.last_traded_price = price
    market.order_book.get_best_ask.return_value = 100.1
    market.order_book.get_best_bid.return_value = 99.9
    del market.regime_sentiment
    agent = RegimeTraderAgent(
        markets=[market],
        repository=MagicMock(),
        configuration={
            "lam": 1,
            "fundamental_price_discovery": True,
            "fundamental_gap_threshold": 0.002,
            **config,
        },
    )
    regime = MagicMock()
    regime.symbols = ("TEST",)
    regime.value.return_value = fundamental
    agent.fundamental_return_regime = regime
    agent.fundamental_initial_prices = {"TEST": 100.0}
    return agent, market


@pytest.mark.parametrize("fundamental,expected", [(110, BUY), (90, SELL)])
def test_fundamental_gap_crosses_spread(fundamental, expected):
    agent, market = _agent(fundamental)
    agent.take_action(0)
    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == expected
    if expected == BUY:
        assert float(order.price) >= 100.1
    else:
        assert float(order.price) <= 99.9


def test_neutral_gap_does_not_trade():
    agent, market = _agent(100.1)
    agent.take_action(0)
    market.add_orders.assert_not_called()


def test_no_opposite_liquidity_no_trade():
    agent, market = _agent(110)
    market.order_book.get_best_ask.return_value = None
    agent.take_action(0)
    market.add_orders.assert_not_called()


def test_position_limit_still_applies():
    agent, market = _agent(110, max_position=10)
    agent.position[101] = 10
    agent.take_action(0)
    market.add_orders.assert_not_called()
