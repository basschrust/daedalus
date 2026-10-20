import pytest

from marketsim.market import Future, Price
from marketsim.agent import Agent


def test_collateral_transfer_preserves_equity(
    futures_agents: dict[str, Agent],
    future_short: Future,
):
    agent = futures_agents["buyer"]
    agent.cash = { "PLN": Price(100_000) }

    equity_before = agent.cash["PLN"]

    agent.update_futures_position(
        quantity=2,
        contract=future_short,
    )

    values = agent.futures_value[future_short.asset_id]

    equity_after = (
        agent.get_cash_balance(currency="PLN")
        + values["collateral"]
        + values["unsettled_pnl"]
    )

    assert equity_after == equity_before
