from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from marketsim.agent.noise_agent import NoiseAgent
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL


def act(agent, *, side=BUY, poisson=2, activity_draw=0.0):
    """Make stochastic decisions deterministic."""
    with (
        patch(
            "marketsim.agent.noise_agent.random.random",
            return_value=activity_draw,
        ),
        patch(
            "marketsim.agent.noise_agent.random.choice",
            return_value=side,
        ),
        patch(
            "marketsim.agent.noise_agent.np.random.poisson",
            return_value=poisson,
        ),
        patch(
            "marketsim.agent.noise_agent.np.random.normal",
            return_value=0.0,
        ),
    ):
        agent.take_action(current_time=1)


@pytest.fixture
def make_agent(tmp_path):
    """Create a NoiseAgent with mocked market infrastructure."""

    def factory(instrument_class="future", config=None):
        if instrument_class == "future":
            market = SimpleNamespace(
                asset_id=101,
                instrument_class=instrument_class,
                underlying=SimpleNamespace(
                    instrument_class="index",
                    calculate_value=MagicMock(return_value=Price(3100)),
                    last_traded_price=Price(3100),
                ),
                calculate_theoretical_value=MagicMock(return_value=Price(3100)),
                last_traded_price=Price(3000),
                order_book=SimpleNamespace(
                    get_best_bid=MagicMock(return_value=Price(2999)),
                    get_best_ask=MagicMock(return_value=Price(3001)),
                ),
                add_orders=MagicMock(),
                withdraw_all=MagicMock(),
            )
        else:
            market = SimpleNamespace(
                asset_id=101,
                instrument_class=instrument_class,
                last_traded_price=Price(3000),
                order_book=SimpleNamespace(
                    get_best_bid=MagicMock(return_value=Price(2999)),
                    get_best_ask=MagicMock(return_value=Price(3001)),
                ),
                add_orders=MagicMock(),
                withdraw_all=MagicMock(),
            )

        with patch(
            "marketsim.agent.agent.setup_agent_logger",
            return_value=MagicMock(),
        ):
            agent = NoiseAgent(
                markets=[market],
                repository=MagicMock(),
                configuration=config or {},
                output_dir=tmp_path,
            )

        return agent, market

    return factory


def submitted_order(market):
    if not market.add_orders.called:
        return None
    return market.add_orders.call_args.args[0][0]


def test_futures_defaults(make_agent):
    agent, _ = make_agent()

    assert agent.configuration["future_lam_multiplier"] == 0.25
    assert agent.configuration["future_mean_contracts"] == 1.5
    assert agent.configuration["future_max_order_contracts"] == 5
    assert agent.configuration["future_max_position_contracts"] == 20


def test_futures_activity_is_reduced(make_agent):
    agent, market = make_agent(config={"lam": 0.2, "future_lam_multiplier": 0.25})

    # Effective futures activity = 0.2 * 0.25 = 0.05
    act(agent, activity_draw=0.10)

    market.add_orders.assert_not_called()


def test_futures_activity_below_threshold(make_agent):
    agent, market = make_agent(config={"lam": 0.2, "future_lam_multiplier": 0.25})

    act(agent, activity_draw=0.04)

    assert submitted_order(market) is not None


def test_futures_activity_at_threshold(make_agent):
    agent, market = make_agent(config={"lam": 0.2, "future_lam_multiplier": 0.25})

    act(agent, activity_draw=0.05)

    market.add_orders.assert_not_called()


def test_futures_poisson_mean(make_agent):
    agent, market = make_agent(config={"lam": 1, "future_mean_contracts": 1.5})

    with (
        patch(
            "marketsim.agent.noise_agent.random.random",
            return_value=0.0,
        ),
        patch(
            "marketsim.agent.noise_agent.random.choice",
            return_value=BUY,
        ),
        patch(
            "marketsim.agent.noise_agent.np.random.poisson",
            return_value=2,
        ) as poisson,
        patch(
            "marketsim.agent.noise_agent.np.random.normal",
            return_value=0.0,
        ),
    ):
        agent.take_action(1)

    poisson.assert_called_once_with(1.5)
    assert submitted_order(market).quantity == 2


def test_futures_order_quantity_cap(make_agent):
    agent, market = make_agent(config={"future_max_order_contracts": 5})

    act(agent, poisson=100)

    assert submitted_order(market).quantity == 5


def test_futures_zero_quantity_skips_order(make_agent):
    agent, market = make_agent()

    act(agent, poisson=0)

    market.add_orders.assert_not_called()


@pytest.mark.parametrize(
    "position,side,expected",
    [
        (0, BUY, 5),
        (0, SELL, 5),
        (18, BUY, 2),
        (20, BUY, 0),
        (20, SELL, 5),
        (-18, SELL, 2),
        (-20, SELL, 0),
        (-20, BUY, 5),
    ],
)
def test_futures_position_limits(make_agent, position, side, expected):
    agent, market = make_agent(
        config={
            "future_max_order_contracts": 5,
            "future_max_position_contracts": 20,
        }
    )
    agent.futures_position[market.asset_id] = position

    act(agent, side=side, poisson=100)

    order = submitted_order(market)

    if expected == 0:
        assert order is None
    else:
        assert order is not None
        assert order.order_type == side
        assert order.quantity == expected
        assert abs(position + side * order.quantity) <= 20


def test_stock_activity_is_not_reduced(make_agent):
    agent, market = make_agent(
        instrument_class="stock",
        config={"lam": 0.2, "future_lam_multiplier": 0.01},
    )

    # Above futures activity threshold, below stock threshold.
    with patch.object(agent, "get_order_quantity", return_value=12):
        act(agent, activity_draw=0.10)

    assert submitted_order(market).quantity == 12


def test_stock_uses_existing_quantity_model(make_agent):
    agent, market = make_agent(instrument_class="stock")

    with patch.object(
        agent,
        "get_order_quantity",
        return_value=37,
    ) as get_quantity:
        act(agent)

    get_quantity.assert_called_once_with(
        market.asset_id,
        fallback=agent.mean_volume,
        max_quantity=agent.q_max,
    )
    assert submitted_order(market).quantity == 37


def test_futures_do_not_use_stock_quantity_model(make_agent):
    agent, market = make_agent()

    with patch.object(
        agent,
        "get_order_quantity",
        side_effect=AssertionError("Stock quantity model used"),
    ):
        act(agent, poisson=3)

    assert submitted_order(market).quantity == 3


def test_futures_order_price_and_validity(make_agent):
    agent, market = make_agent(config={"validity_period": 4})

    act(agent, side=BUY, poisson=2)

    order = submitted_order(market)

    assert order.price == Price(3100)
    assert order.quantity == 2
    assert order.valid_until == 5


def test_withdraw_old_orders(make_agent):
    agent, market = make_agent(config={"withdraw_old": True})

    act(agent)

    market.withdraw_all.assert_called_once_with(agent_id=agent.agent_id)


def test_index_is_not_traded(make_agent):
    agent, market = make_agent(instrument_class="index")

    act(agent)

    market.add_orders.assert_not_called()


def test_single_stock_future_uses_theoretical_price(make_agent):
    agent, market = make_agent(
        config={"lam": 1, "future_lam_multiplier": 1}
    )

    market.underlying = SimpleNamespace(
        instrument_class="stock",
        last_traded_price=Price(125),
    )
    market.calculate_theoretical_value.return_value = Price(125)

    act(agent, side=BUY, poisson=1)

    order = submitted_order(market)

    assert order is not None
    assert order.price == Price(125)
    market.calculate_theoretical_value.assert_called_once_with()

