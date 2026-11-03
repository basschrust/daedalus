from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from marketsim.agent.arbitrageur import IndexFuturesArbitrageAgent
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL


@pytest.fixture
def make_agent(tmp_path):
    def factory(
        *,
        fair=3000,
        bid=3000,
        ask=3001,
        position=0,
        config=None,
    ):
        stock = SimpleNamespace(
            asset_id=11,
            instrument_class="stock",
            status="active",
            order_book=SimpleNamespace(
                get_best_bid=MagicMock(return_value=Price(99)),
                get_best_ask=MagicMock(return_value=Price(100)),
            ),
            withdraw_all=MagicMock(),
            add_orders=MagicMock(),
        )

        index = SimpleNamespace(
            base_index_level=3000,
            components={
                11: {
                    "security": stock,
                    "weight": 1,
                    "base_price": 100,
                }
            },
            calculate_value=MagicMock(return_value=Price(fair)),
        )

        future = SimpleNamespace(
            asset_id=101,
            instrument_class="future",
            status="active",
            multiplier=20,
            underlying=index,
            order_book=SimpleNamespace(
                get_best_bid=MagicMock(return_value=Price(bid) if bid is not None else None),
                get_best_ask=MagicMock(return_value=Price(ask) if ask is not None else None),
            ),
            withdraw_all=MagicMock(),
            add_orders=MagicMock(),
        )

        with patch(
            "marketsim.agent.agent.setup_agent_logger",
            return_value=MagicMock(),
        ):
            agent = IndexFuturesArbitrageAgent(
                markets=[future, stock],
                repository=MagicMock(),
                configuration=config,
                output_dir=tmp_path,
            )

        agent.futures_position[future.asset_id] = position
        return agent, future, stock

    return factory


def get_order(future):
    if not future.add_orders.called:
        return None
    return future.add_orders.call_args.args[0][0]


def test_overpriced_future_is_sold(make_agent):
    agent, future, _ = make_agent(bid=3030, ask=3031)

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == SELL
    assert order.price == Price(3030)
    assert order.quantity == 2


def test_underpriced_future_is_bought(make_agent):
    agent, future, _ = make_agent(bid=2969, ask=2970)

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == BUY
    assert order.price == Price(2970)


def test_no_trade_within_threshold(make_agent):
    agent, future, _ = make_agent(bid=3002, ask=3003)

    agent.take_action(1)

    future.add_orders.assert_not_called()


@pytest.mark.parametrize(
    "position,bid,ask",
    [
        (-20, 3030, 3031),
        (20, 2969, 2970),
    ],
)
def test_position_limit(make_agent, position, bid, ask):
    agent, future, _ = make_agent(
        position=position,
        bid=bid,
        ask=ask,
    )

    agent.take_action(1)

    future.add_orders.assert_not_called()


def test_adaptive_order_size(make_agent):
    agent, future, _ = make_agent(
        bid=3060,
        ask=3061,
        config={
            "order_quantity": 5,
            "max_order_quantity": 40,
            "max_position": 100,
            "volume_aggression": 5.0,
        },
    )

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == SELL
    assert 5 < order.quantity <= 40


def test_adaptive_size_respects_capacity(make_agent):
    agent, future, _ = make_agent(
        bid=2960,
        ask=2970,
        position=98,
        config={
            "order_quantity": 10,
            "max_order_quantity": 50,
            "max_position": 100,
            "volume_aggression": 8.0,
        },
    )

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == BUY
    assert order.quantity == 2


def test_missing_quotes(make_agent):
    agent, future, _ = make_agent(bid=None, ask=None)

    agent.take_action(1)

    future.add_orders.assert_not_called()


def test_inactive_future(make_agent):
    agent, future, _ = make_agent(bid=3030, ask=3031)
    future.status = "inactive"

    agent.take_action(1)

    future.add_orders.assert_not_called()


def test_order_cooldown(make_agent):
    agent, future, _ = make_agent(
        bid=3030,
        ask=3031,
        config={
            "min_time_between_orders": 3,
        },
    )

    agent.take_action(1)
    agent.take_action(2)
    agent.take_action(3)

    assert future.add_orders.call_count == 1

    agent.take_action(4)

    assert future.add_orders.call_count == 2


def test_closing_long_never_reverses_position(make_agent):
    agent, future, _ = make_agent(
        bid=3000,
        ask=3001,
        position=1,
        config={
            "order_quantity": 10,
            "max_order_quantity": 50,
            "max_position": 100,
            "volume_aggression": 8.0,
        },
    )

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == SELL
    assert order.quantity == 1


def test_closing_short_never_reverses_position(make_agent):
    agent, future, _ = make_agent(
        bid=2999,
        ask=3000,
        position=-1,
        config={
            "order_quantity": 10,
            "max_order_quantity": 50,
            "max_position": 100,
            "volume_aggression": 8.0,
        },
    )

    agent.take_action(1)

    order = get_order(future)
    assert order.order_type == BUY
    assert order.quantity == 1


def test_stock_hedge_deltas_use_filled_positions(make_agent):
    agent, future, stock = make_agent(position=-2)

    # One stock in the fixture:
    # index coefficient = 3000 / 100 = 30
    # multiplier = 20, contracts = -2
    # target = 1200 shares
    agent.position[stock.asset_id] = 400

    deltas = agent.get_stock_hedge_deltas(future)

    assert deltas[stock.asset_id] == 800


def test_stock_hedge_deltas_reverse_for_long_future(make_agent):
    agent, future, stock = make_agent(position=1)

    deltas = agent.get_stock_hedge_deltas(future)

    assert deltas[stock.asset_id] == -600


def test_short_futures_generates_stock_buy_orders(make_agent):
    agent, future, stock = make_agent(position=-2)

    stock.status = "active"
    stock.order_book = SimpleNamespace(
        get_best_ask=MagicMock(return_value=Price(100)),
        get_best_bid=MagicMock(return_value=Price(99)),
    )
    stock.withdraw_all = MagicMock()
    stock.add_orders = MagicMock()

    agent._submit_stock_hedge(
        future=future,
        current_time=1,
    )

    order = stock.add_orders.call_args.args[0][0]

    assert order.order_type == BUY
    assert order.quantity == 1200
    assert order.price == Price(100)


def test_long_futures_generates_stock_sell_orders(make_agent):
    agent, future, stock = make_agent(position=2)

    stock.status = "active"
    stock.order_book = SimpleNamespace(
        get_best_ask=MagicMock(return_value=Price(100)),
        get_best_bid=MagicMock(return_value=Price(99)),
    )
    stock.withdraw_all = MagicMock()
    stock.add_orders = MagicMock()

    agent._submit_stock_hedge(
        future=future,
        current_time=1,
    )

    order = stock.add_orders.call_args.args[0][0]

    assert order.order_type == SELL
    assert order.quantity == 1200
    assert order.price == Price(99)


def test_partial_hedge_fill_generates_remaining_order(make_agent):
    agent, future, stock = make_agent(position=-2)

    stock.status = "active"
    stock.order_book = SimpleNamespace(
        get_best_ask=MagicMock(return_value=Price(100)),
        get_best_bid=MagicMock(return_value=Price(99)),
    )
    stock.withdraw_all = MagicMock()
    stock.add_orders = MagicMock()

    # 1200 shares required, 500 already executed.
    agent.position[stock.asset_id] = 500

    agent._submit_stock_hedge(
        future=future,
        current_time=2,
    )

    order = stock.add_orders.call_args.args[0][0]

    assert order.order_type == BUY
    assert order.quantity == 700


def test_flat_futures_unwinds_remaining_stock_hedge(make_agent):
    agent, future, stock = make_agent(position=0)

    stock.status = "active"
    stock.order_book = SimpleNamespace(
        get_best_ask=MagicMock(return_value=Price(100)),
        get_best_bid=MagicMock(return_value=Price(99)),
    )
    stock.withdraw_all = MagicMock()
    stock.add_orders = MagicMock()

    agent.position[stock.asset_id] = 800

    agent._submit_stock_hedge(
        future=future,
        current_time=3,
    )

    order = stock.add_orders.call_args.args[0][0]

    assert order.order_type == SELL
    assert order.quantity == 800


def test_flat_portfolio_does_not_trade_stocks(make_agent):
    agent, future, stock = make_agent(
        position=0,
        bid=3000,
        ask=3001,
    )

    agent.take_action(current_time=1)

    stock.add_orders.assert_not_called()
    future.add_orders.assert_not_called()
