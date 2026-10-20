import pytest

from marketsim.agent import Agent
from marketsim.market import Price, Future

def test_expiration_removes_only_expired_future(
    futures_agents: dict[str, Agent], future_short: Future, future_long: Future
):
    agent = futures_agents["buyer"]
    agent.cash = Price(100_000)

    agent.update_futures_position(
        quantity=2, contract=future_short
    )
    agent.update_futures_position(
        quantity=1, contract=future_long
    )

    short_id = future_short.asset_id
    long_id = future_long.asset_id

    agent.settle(future_short)

    assert agent.futures_position[short_id] == 0
    assert short_id not in agent.futures_value
    assert short_id not in agent.markets
    assert short_id in agent.abandoned_markets

    assert agent.futures_position[long_id] == 1
    assert long_id in agent.futures_value
    assert long_id in agent.markets


def test_final_settlement_returns_collateral_and_pnl(
    futures_agents: dict[str, Agent], future_short: Future
):
    agent = futures_agents["buyer"]
    agent.cash = Price(100_000)

    agent.update_futures_position(
        quantity=2, contract=future_short
    )

    contract_id = future_short.asset_id
    values = agent.futures_value[contract_id]

    values["unsettled_pnl"] = Price(200)

    cash_before = agent.cash
    collateral = values["collateral"]

    agent.settle(future_short)

    assert agent.cash == pytest.approx(
        cash_before + collateral + 200
    )
    assert agent.futures_position[contract_id] == 0
