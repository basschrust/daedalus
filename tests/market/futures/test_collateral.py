from decimal import Decimal

from marketsim.agent import Agent
from marketsim.market import Future
from marketsim.market.price import Price


def test_collateral_transfer_preserves_equity(
    futures_agents: dict[str, Agent],
    future_short: Future,
):
    agent = futures_agents["buyer"]
    agent.cash = {"PLN": Price(100_000)}

    equity_before = agent.cash["PLN"]

    agent.update_futures_position(
        quantity=2,
        contract=future_short,
    )

    values = agent.futures_value[future_short.asset_id]

    equity_after = (
        agent.get_cash_balance(currency="PLN") + values["collateral"] + values["unsettled_pnl"]
    )

    assert equity_after == equity_before


def test_futures_collateral_uses_decimal():
    position = 2000
    price = Price("3000.00")
    multiplier = 20
    margin_rate = Decimal("0.10")

    collateral = Decimal(position) * Decimal(str(price)) * Decimal(multiplier) * margin_rate

    assert collateral == Decimal("12000000.00")
    assert isinstance(collateral, Decimal)
