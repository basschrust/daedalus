
import pytest

from marketsim.market import Price


def test_opening_future_reserves_collateral(
    futures_agents, future_short
):
    buyer = futures_agents["buyer"]
    buyer.cash = { "PLN": Price(100_000) }

    initial_cash = buyer.cash
    buyer.update_futures_position(
        quantity=2,
        contract=future_short,
    )

    contract_id = future_short.asset_id
    collateral = buyer.futures_value[contract_id]["collateral"]

    expected_collateral = (
        future_short.last_traded_price
        * future_short.multiplier
        * future_short.margin_rate
        * 2
    )

    assert collateral == pytest.approx(expected_collateral)
    assert buyer.cash["PLN"] == pytest.approx(
        initial_cash["PLN"] - expected_collateral
    )
    assert buyer.futures_position[contract_id] == 2


@pytest.mark.parametrize(
    "quantity, expected_position",
    [
        (1, 1),
        (-1, -1),
        (3, 3),
        (-3, -3),
    ],
)
def test_opening_long_and_short_positions(
    futures_agents, future_short,
    quantity, expected_position,
):
    agent = futures_agents["buyer"]
    agent.cash = { "PLN": Price(100_000) }

    agent.update_futures_position(
        quantity=quantity,
        contract=future_short,
    )

    assert (
        agent.futures_position[future_short.asset_id]
        == expected_position
    )


def test_increasing_position_does_not_double_count(
    futures_agents, future_short
):
    agent = futures_agents["buyer"]
    agent.cash = { "PLN": Price(100_000) }

    agent.update_futures_position(
        quantity=2, contract=future_short
    )
    agent.update_futures_position(
        quantity=1, contract=future_short
    )

    assert agent.futures_position[future_short.asset_id] == 3


def test_closing_position_releases_collateral(
    futures_agents, future_short
):
    agent = futures_agents["buyer"]
    initial_cash = { "PLN": Price(100_000) }
    agent.cash = initial_cash

    agent.update_futures_position(
        quantity=2, contract=future_short
    )
    agent.update_futures_position(
        quantity=-2, contract=future_short
    )

    contract_id = future_short.asset_id

    assert agent.futures_position[contract_id] == 0
    assert agent.futures_value[contract_id]["collateral"] == 0
    assert agent.cash == initial_cash
