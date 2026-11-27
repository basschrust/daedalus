"""Domain tests; do not require a repository or a running simulation."""

from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

import pytest

from marketsim.risk.portfolio_snapshot import FuturesHolding, PortfolioSnapshot


def _future(*, quantity: int = -2) -> FuturesHolding:
    return FuturesHolding(
        quantity=quantity,
        avg_initial_price=Decimal("2500.12"),
        settlement_price=Decimal("2490.34"),
        collateral=Decimal("10000.25"),
        unsettled_pnl=Decimal("0"),
    )


def test_negative_day_snapshot_is_defensively_immutable() -> None:
    positions = {5: 10}
    cash = {"PLN": Decimal("123.45")}
    futures = {9: _future()}
    snapshot = PortfolioSnapshot(
        agent_id=3,
        day=-1,
        as_of_date=date(2026, 11, 20),
        base_currency="pln",
        positions=positions,
        futures_positions=futures,
        cash=cash,
        portfolio_value=Decimal("30000.10"),
    )
    positions[5] = 999
    cash["PLN"] = Decimal("0")
    futures.clear()
    assert snapshot.positions[5] == 10
    assert snapshot.cash["PLN"] == Decimal("123.45")
    assert snapshot.futures_positions[9].quantity == -2
    assert snapshot.base_currency == "PLN"
    with pytest.raises(TypeError):
        snapshot.positions[5] = 11
    with pytest.raises(TypeError):
        snapshot.cash["PLN"] = Decimal("2")
    with pytest.raises(FrozenInstanceError):
        snapshot.day = 0


def test_capture_live_agent_does_not_mutate_source() -> None:
    agent = SimpleNamespace(
        agent_id=7,
        base_currency="PLN",
        position={1: -10},
        futures_position={2: 3},
        futures_value={
            2: {
                "avg_initial_price": Decimal("2700"),
                "settlement_price": Decimal("2701"),
                "collateral": Decimal("2000"),
                "unsettled_pnl": Decimal("0"),
            }
        },
        cash={"PLN": Decimal("500")},
        portfolio_value=Decimal("2500"),
    )
    snapshot = PortfolioSnapshot.from_agent(agent=agent, day=0)
    agent.position[1] = 100
    agent.futures_value[2]["collateral"] = Decimal("0")
    agent.cash["PLN"] = Decimal("0")
    assert snapshot.positions[1] == -10
    assert snapshot.futures_positions[2].collateral == Decimal("2000")
    assert snapshot.cash["PLN"] == Decimal("500")
    assert snapshot.as_of_date is None  # legacy calendar-free simulation


@pytest.mark.parametrize("quantity", [1.2, True, "3"])
def test_reject_noninteger_quantity(quantity: object) -> None:
    with pytest.raises(ValueError):
        PortfolioSnapshot(
            agent_id=1,
            day=0,
            base_currency="PLN",
            positions={1: quantity},
            futures_positions={},
            cash={},
            portfolio_value=Decimal("0"),
        )


def test_reject_nonfinite_money_and_missing_futures_accounting() -> None:
    with pytest.raises(ValueError):
        PortfolioSnapshot(
            agent_id=1,
            day=-4,
            base_currency="PLN",
            positions={},
            futures_positions={},
            cash={"PLN": "NaN"},
            portfolio_value=Decimal("0"),
        )
    agent = SimpleNamespace(
        agent_id=1,
        base_currency="PLN",
        position={},
        futures_position={12: 2},
        futures_value={},
        cash={},
        portfolio_value=Decimal("0"),
    )
    with pytest.raises(ValueError, match="Missing futures accounting"):
        PortfolioSnapshot.from_agent(agent=agent, day=0)
