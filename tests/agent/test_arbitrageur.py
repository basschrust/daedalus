"""Tests for the index-futures basis arbitrageur."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from marketsim.agent.arbitrageur import (
    IndexFuturesArbitrageAgent,
)
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
        book = SimpleNamespace(
            get_best_bid=MagicMock(return_value=Price(bid) if bid is not None else None),
            get_best_ask=MagicMock(return_value=Price(ask) if ask is not None else None),
        )

        future = SimpleNamespace(
            asset_id=101,
            instrument_class="future",
            status="active",
            underlying=SimpleNamespace(calculate_value=MagicMock(return_value=Price(fair))),
            order_book=book,
            withdraw_all=MagicMock(),
            add_orders=MagicMock(),
        )

        repository = MagicMock()

        # The base Agent registers itself in the repository.
        with patch(
            "marketsim.agent.agent.setup_agent_logger",
            return_value=MagicMock(),
        ):
            agent = IndexFuturesArbitrageAgent(
                markets=[future],
                repository=repository,
                output_dir=tmp_path,
                configuration=config,
            )

        agent.futures_position[future.asset_id] = position

        return agent, future

    return factory


def submitted_order(future):
    """Return the most recently submitted order, if any."""
    if not future.add_orders.called:
        return None

    return future.add_orders.call_args.args[0][0]


def test_sells_overpriced_futures(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=3030,
        ask=3031,
    )

    agent.take_action(current_time=1)

    order = submitted_order(future)

    assert order is not None
    assert order.order_type == SELL
    assert order.quantity == 2
    assert order.price == Price(3030)
    assert order.asset_id == future.asset_id


def test_buys_underpriced_futures(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=2969,
        ask=2970,
    )

    agent.take_action(current_time=1)

    order = submitted_order(future)

    assert order is not None
    assert order.order_type == BUY
    assert order.quantity == 2
    assert order.price == Price(2970)


def test_no_trade_inside_entry_band(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=3002,
        ask=3003,
    )

    agent.take_action(current_time=1)

    future.add_orders.assert_not_called()


def test_closes_long_when_basis_converges(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=2999,
        ask=3000,
        position=5,
    )

    agent.take_action(current_time=1)

    order = submitted_order(future)

    assert order is not None
    assert order.order_type == SELL
    assert order.quantity == 2


def test_closes_short_when_basis_converges(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=3000,
        ask=3001,
        position=-3,
    )

    agent.take_action(current_time=1)

    order = submitted_order(future)

    assert order is not None
    assert order.order_type == BUY
    assert order.quantity == 2


@pytest.mark.parametrize(
    ("position", "bid", "ask"),
    [
        (-20, 3030, 3031),
        (20, 2969, 2970),
    ],
)
def test_respects_maximum_position(
    make_agent,
    position,
    bid,
    ask,
):
    agent, future = make_agent(
        position=position,
        bid=bid,
        ask=ask,
    )

    agent.take_action(current_time=1)

    future.add_orders.assert_not_called()


def test_limits_quantity_to_remaining_capacity(make_agent):
    agent, future = make_agent(
        position=19,
        bid=2969,
        ask=2970,
    )

    agent.take_action(current_time=1)

    order = submitted_order(future)

    assert order is not None
    assert order.order_type == BUY
    assert order.quantity == 1


def test_missing_quotes_do_not_generate_orders(make_agent):
    agent, future = make_agent(
        bid=None,
        ask=None,
    )

    agent.take_action(current_time=1)

    future.add_orders.assert_not_called()


def test_inactive_future_is_ignored(make_agent):
    agent, future = make_agent(
        bid=3030,
        ask=3031,
    )
    future.status = "inactive"

    agent.take_action(current_time=1)

    future.add_orders.assert_not_called()


def test_minimum_time_between_orders(make_agent):
    agent, future = make_agent(
        bid=3030,
        ask=3031,
        config={
            "min_time_between_orders": 3,
        },
    )

    agent.take_action(current_time=1)
    assert future.add_orders.call_count == 1

    agent.take_action(current_time=2)
    assert future.add_orders.call_count == 1

    agent.take_action(current_time=4)
    assert future.add_orders.call_count == 2


def test_no_trade_when_basis_does_not_cover_costs(make_agent):
    agent, future = make_agent(
        fair=3000,
        bid=3010,
        ask=3011,
        config={
            "entry_threshold": 0.003,
            "transaction_cost": 0.001,
        },
    )

    agent.take_action(current_time=1)

    future.add_orders.assert_not_called()


def test_invalid_configuration_rejected(make_agent):
    with pytest.raises(ValueError):
        make_agent(
            config={
                "entry_threshold": 0.001,
                "exit_threshold": 0.002,
            }
        )
