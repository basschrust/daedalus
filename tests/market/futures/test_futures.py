from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from marketsim.agent import NoiseAgent
from marketsim.market.fourheap.constants import BUY
from marketsim.market.price import Price


@pytest.fixture
def make_agent(tmp_path):
    """Create a NoiseAgent with mocked market infrastructure."""

    def factory(instrument_class="future", config=None):
        if instrument_class == "future":
            market = SimpleNamespace(
                asset_id=101,
                instrument_class=instrument_class,
                underlying=SimpleNamespace(
                    calculate_value=MagicMock(return_value=Price(3100)),
                ),
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


def submitted_order(market):
    if not market.add_orders.called:
        return None
    return market.add_orders.call_args.args[0][0]


################    test functions         ###################


def test_future_uses_theoretical_price_not_last_trade(make_agent):
    agent, market = make_agent()

    market.last_traded_price = Price(2900)
    market.underlying.calculate_value.return_value = Price(3100)

    market.order_book.get_best_bid.return_value = Price(2999)
    market.order_book.get_best_ask.return_value = Price(3001)

    act(agent, side=BUY, poisson=1)

    order = submitted_order(market)

    assert order.price == Price(3100)
    market.underlying.calculate_value.assert_called_once()


###  trading


#### expiration / exercise ###


### reporting
