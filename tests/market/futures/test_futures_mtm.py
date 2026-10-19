
import pytest

from marketsim.market import Price, Future
from marketsim.agent import Agent

@pytest.fixture
def mtm_agents(futures_agents: dict[str, Agent], future_short: Future):
    """Two counterparties with opposing futures positions."""
    buyer = futures_agents["buyer"]
    seller = futures_agents["seller"]

    for agent in (buyer, seller):
        agent.cash = Price(100_000)
        agent.sod()

    buyer.update_futures_position(
        quantity=2, contract=future_short
    )
    seller.update_futures_position(
        quantity=-2, contract=future_short
    )

    return buyer, seller


@pytest.fixture
def mtm_scenario(mtm_agents: tuple[Agent, Agent], future_short: Future):
    """Deterministic daily settlement scenario."""
    buyer, seller = mtm_agents

    return {
        "buyer": buyer,
        "seller": seller,
        "contract": future_short,
        "initial_price": Price(
            future_short.last_traded_price
        ),
        "quantity": 2,
        "multiplier": future_short.multiplier,
    }


def expected_vm(quantity, multiplier, old_price, new_price):
    return Price(
        quantity * multiplier * (new_price - old_price)
    )


def test_daily_mtm_long_profit(mtm_scenario):
    s = mtm_scenario
    buyer = s["buyer"]
    contract = s["contract"]

    old_price = s["initial_price"]
    new_price = Price(old_price + 5)

    cash_before = buyer.cash
    contract.last_traded_price = new_price

    buyer.eod()

    vm = expected_vm(
        s["quantity"], s["multiplier"],
        old_price, new_price,
    )

    assert buyer.cash == cash_before + vm


def test_daily_mtm_short_loss(mtm_scenario):
    s = mtm_scenario
    seller = s["seller"]
    contract = s["contract"]

    old_price = s["initial_price"]
    new_price = Price(old_price + 5)

    cash_before = seller.cash
    contract.last_traded_price = new_price

    seller.eod()

    vm = expected_vm(
        -s["quantity"], s["multiplier"],
        old_price, new_price,
    )

    assert seller.cash == cash_before + vm


def test_daily_mtm_zero_sum(mtm_scenario):
    s = mtm_scenario
    buyer, seller = s["buyer"], s["seller"]
    contract = s["contract"]

    cash_before = buyer.cash + seller.cash

    contract.last_traded_price = Price(
        s["initial_price"] + 7
    )

    buyer.eod()
    seller.eod()

    assert buyer.cash + seller.cash == cash_before


def test_mtm_resets_unsettled_pnl(mtm_scenario):
    s = mtm_scenario
    buyer = s["buyer"]
    contract = s["contract"]

    contract.last_traded_price = Price(
        s["initial_price"] + 5
    )

    buyer.eod()

    values = buyer.futures_value[contract.asset_id]

    assert values["unsettled_pnl"] == Price(0)
    assert values["settlement_price"] == (
        contract.last_traded_price
    )


def test_eod_is_idempotent(mtm_scenario):
    s = mtm_scenario
    buyer = s["buyer"]
    contract = s["contract"]

    contract.last_traded_price = Price(
        s["initial_price"] + 5
    )

    buyer.eod()
    cash_after_first_eod = buyer.cash

    buyer.eod()

    assert buyer.cash == cash_after_first_eod
    assert buyer.current_day == 1




@pytest.mark.parametrize(
    "day,price,expected_long_vm",
    [
        (1, 105, 200),
        (2, 102, -120),
        (3, 110, 320),
    ],
)
def test_mtm_formula(
    day, price, expected_long_vm
):
    previous_prices = {
        1: 100,
        2: 105,
        3: 102,
    }

    vm = expected_vm(
        quantity=2,
        multiplier=20,
        old_price=Price(previous_prices[day]),
        new_price=Price(price),
    )

    assert vm == Price(expected_long_vm)


def test_mtm_over_multiple_days(mtm_scenario):
    s = mtm_scenario
    buyer = s["buyer"]
    contract = s["contract"]

    initial_price = s["initial_price"]
    initial_cash = buyer.cash

    # Price movements relative to opening price
    daily_changes = [5, -3, 8]
    expected_cumulative_vm = Price(0)
    current_price = initial_price

    for day, change in enumerate(daily_changes):
        previous_price = current_price
        current_price = Price(current_price + change)

        contract.last_traded_price = current_price

        expected_cumulative_vm += expected_vm(
            quantity=s["quantity"],
            multiplier=s["multiplier"],
            old_price=previous_price,
            new_price=current_price,
        )

        buyer.eod()

        assert buyer.cash == (
            initial_cash + expected_cumulative_vm
        )

        assert (
            buyer.futures_value[contract.asset_id][
                "settlement_price"
            ] == current_price
        )

        assert (
            buyer.futures_value[contract.asset_id][
                "unsettled_pnl"
            ] == Price(0)
        )

        if day < len(daily_changes) - 1:
            buyer.sod()



def test_settlement_is_independent_between_contracts(
    futures_agents,
    future_short,
    future_long,
):
    buyer = futures_agents["buyer"]
    buyer.cash = Price(100_000)
    buyer.sod()

    buyer.update_futures_position(
        quantity=2, contract=future_short
    )
    buyer.update_futures_position(
        quantity=1, contract=future_long
    )

    short_id = future_short.asset_id
    long_id = future_long.asset_id

    short_price = Price(future_short.last_traded_price)
    long_price = Price(future_long.last_traded_price)

    cash_before = buyer.cash

    future_short.last_traded_price = Price(short_price + 5)
    future_long.last_traded_price = Price(long_price - 3)

    buyer.eod()

    expected_cash_change = (
        2 * future_short.multiplier * 5
        - future_long.multiplier * 3
    )

    assert buyer.cash == (
        cash_before + Price(expected_cash_change)
    )

    assert buyer.futures_position[short_id] == 2
    assert buyer.futures_position[long_id] == 1

    assert (
        buyer.futures_value[short_id]["settlement_price"]
        == future_short.last_traded_price
    )
    assert (
        buyer.futures_value[long_id]["settlement_price"]
        == future_long.last_traded_price
    )
