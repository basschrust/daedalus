from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import Security
from marketsim.market.price import Price

################    test functions         ###################


def test_future_uses_theoretical_price_not_last_trade(make_agent):
    agent, market = make_agent()

    market.last_traded_price = Price(2900)
    market.underlying.calculate_value.return_value = Price(3000)

    # Remove existing quotes so they cannot affect the test.
    market.order_book.get_best_bid.return_value = None
    market.order_book.get_best_ask.return_value = None

    with (
        patch("marketsim.agent.noise_agent.random.random", return_value=0.0),
        patch("marketsim.agent.noise_agent.random.choice", return_value=BUY),
        patch("marketsim.agent.noise_agent.np.random.poisson", return_value=1),
        patch("marketsim.agent.noise_agent.np.random.normal", return_value=0.0),
    ):
        agent.take_action(current_time=1)

    market.add_orders.assert_called_once()

    order = market.add_orders.call_args.args[0][0]

    assert order.price == Price(3000)
    market.underlying.calculate_value.assert_called_once()


###  trading


#### expiration / exercise ###


### reporting
