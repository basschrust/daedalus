from unittest.mock import MagicMock, patch

import pytest

from marketsim.agent.regime_trader import RegimeTraderAgent
from marketsim.fourheap.constants import BUY, SELL


def make_agent(**config):
    market = MagicMock()
    market.asset_id = 101
    market.instrument_class = "stock"
    market.order_book.get_best_ask.return_value = 100.1
    market.order_book.get_best_bid.return_value = 99.9
    market.last_traded_price = 100
    del market.regime_sentiment  # MagicMock must not pretend it already exists
    repo = MagicMock()
    agent = RegimeTraderAgent(markets=[market], repository=repo, configuration=config)
    return agent, market


def test_bull_crosses_ask():
    agent, market = make_agent(
        lam=1, bullish_probability=1, bearish_probability=0, regime_switch_probability=0
    )
    agent.take_action(0)
    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == BUY
    assert float(order.price) >= 100.1
    assert order.valid_until == 2


def test_bear_crosses_bid():
    agent, market = make_agent(
        lam=1, bullish_probability=0, bearish_probability=1, regime_switch_probability=0
    )
    agent.take_action(0)
    order = market.add_orders.call_args.args[0][0]
    assert order.order_type == SELL
    assert float(order.price) <= 99.9


def test_neutral_submits_nothing():
    agent, market = make_agent(lam=1, bullish_probability=0, bearish_probability=0)
    agent.take_action(0)
    market.add_orders.assert_not_called()


def test_missing_liquidity_submits_nothing():
    agent, market = make_agent(lam=1, bullish_probability=1, bearish_probability=0)
    market.order_book.get_best_ask.return_value = None
    agent.take_action(0)
    market.add_orders.assert_not_called()


def test_shared_sentiment_updates_once_per_tick():
    agent1, market = make_agent(
        lam=0, bullish_probability=1, bearish_probability=0, regime_switch_probability=1
    )
    agent2 = RegimeTraderAgent(
        markets=[market], repository=MagicMock(), configuration=agent1.configuration
    )
    with (
        patch.object(agent1, "_draw_regime", return_value=BUY) as draw1,
        patch.object(agent2, "_draw_regime", return_value=SELL) as draw2,
    ):
        agent1.take_action(0)
        agent2.take_action(0)
    assert market.regime_sentiment["regime"] == BUY
    assert draw1.call_count == 2  # initial draw and forced switch
    draw2.assert_not_called()


def test_invalid_probabilities():
    with pytest.raises(ValueError):
        make_agent(bullish_probability=0.8, bearish_probability=0.8)


@pytest.mark.parametrize(
    "bullish, bearish, quote",
    [
        (1.0, 0.0, float("inf")),
        (0.0, 1.0, float("-inf")),
    ],
)
def test_empty_order_book_does_not_create_orders(bullish, bearish, quote):
    agent, market = make_agent(
        lam=1,
        bullish_probability=bullish,
        bearish_probability=bearish,
        regime_switch_probability=0,
    )

    if bullish:
        market.order_book.get_best_ask.return_value = quote
    else:
        market.order_book.get_best_bid.return_value = quote

    agent.take_action(current_time=0)

    market.add_orders.assert_not_called()
