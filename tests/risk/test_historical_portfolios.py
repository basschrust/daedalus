"""Offline historical equity snapshots and frozen-book backtesting."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from marketsim.risk.historical_portfolios import build_historical_equity_snapshots
from marketsim.risk.hypothetical_backtesting import one_day_hpl


@pytest.fixture
def history() -> tuple[list[date], dict[date, dict[int, Decimal]]]:
    days = [date(2026, 11, 19), date(2026, 11, 20), date(2026, 11, 23)]
    prices = {
        days[0]: {1: Decimal("100"), 2: Decimal("20")},
        days[1]: {1: Decimal("98"), 2: Decimal("22")},
        days[2]: {1: Decimal("101"), 2: Decimal("19")},
    }
    return days, prices


def test_negative_days_and_cash_accounting(*, history) -> None:
    days, prices = history
    snapshots = build_historical_equity_snapshots(
        agent_id=5,
        dates=days,
        holdings={1: 10, 2: -5},
        instrument_classes={1: "stock", 2: "stock"},
        closes=prices,
        initial_cash={"PLN": Decimal("250")},
    )
    assert [s.day for s in snapshots] == [-3, -2, -1]
    assert [s.as_of_date for s in snapshots] == days
    assert [s.portfolio_value for s in snapshots] == [
        Decimal("1150"),
        Decimal("1120"),
        Decimal("1165"),
    ]
    with pytest.raises(TypeError):
        snapshots[0].positions[1] = 100


def test_frozen_positions_hpl_and_var_exceptions(*, history) -> None:
    days, prices = history
    snapshots = build_historical_equity_snapshots(
        agent_id=5,
        dates=days,
        holdings={1: 10, 2: -5},
        instrument_classes={1: "stock", 2: "stock"},
        closes=prices,
        initial_cash={"PLN": Decimal("250")},
    )
    forecasts = pd.DataFrame(
        [
            {
                "method": "parametric",
                "day": -3,
                "agent_id": 5,
                "group_name": "Test",
                "confidence_level": 0.99,
                "horizon_days": 1,
                "var_amount": 25.0,
            },
            {
                "method": "parametric",
                "day": -2,
                "agent_id": 5,
                "group_name": "Test",
                "confidence_level": 0.99,
                "horizon_days": 1,
                "var_amount": 50.0,
            },
            {
                "method": "parametric",
                "day": -1,
                "agent_id": 5,
                "group_name": "Test",
                "confidence_level": 0.99,
                "horizon_days": 1,
                "var_amount": 1.0,
            },
        ]
    )
    observations = one_day_hpl(
        forecasts=forecasts,
        snapshots=snapshots,
        instrument_classes={1: "stock", 2: "stock"},
        closes=prices,
    )
    assert observations.hypothetical_pnl.tolist() == [-30.0, 45.0]
    assert observations.exception.tolist() == [True, False]


def test_missing_closes_fail_instead_of_fabricating_zero(*, history) -> None:
    dates, closes = history
    broken = {d: dict(values) for d, values in closes.items()}
    del broken[dates[1]][1]
    with pytest.raises(ValueError, match="Missing close"):
        build_historical_equity_snapshots(
            agent_id=1,
            dates=dates,
            holdings={1: 1},
            instrument_classes={1: "stock"},
            closes=broken,
            initial_cash={"PLN": Decimal(0)},
        )


def test_missing_forward_mark_prevents_hpl(*, history) -> None:
    dates, prices = history
    snapshots = build_historical_equity_snapshots(
        agent_id=1,
        dates=dates,
        holdings={1: 1},
        instrument_classes={1: "stock"},
        closes=prices,
        initial_cash={"PLN": Decimal(0)},
    )
    missing = {d: dict(v) for d, v in prices.items()}
    del missing[dates[1]][1]
    forecasts = pd.DataFrame([{"day": -3, "agent_id": 1, "horizon_days": 1, "var_amount": 1.0}])
    with pytest.raises(ValueError, match="Missing historical close"):
        one_day_hpl(
            forecasts=forecasts,
            snapshots=snapshots,
            instrument_classes={1: "stock"},
            closes=missing,
        )


def test_rejects_non_base_cash(*, history) -> None:
    dates, prices = history
    with pytest.raises(ValueError, match="base_currency"):
        build_historical_equity_snapshots(
            agent_id=1,
            dates=dates,
            holdings={1: 1},
            instrument_classes={1: "stock"},
            closes=prices,
            initial_cash={"PLN": Decimal(0), "EUR": Decimal(1)},
        )


def test_unsupported_option_holding_fails_closed(*, history) -> None:
    dates, closes = history
    with pytest.raises(ValueError, match="Unsupported historical holding"):
        build_historical_equity_snapshots(
            agent_id=1,
            dates=dates,
            holdings={1: 1},
            instrument_classes={1: "option"},
            closes=closes,
            initial_cash={"PLN": Decimal(0)},
        )


def test_historical_book_does_not_read_future_closes(*, history) -> None:
    dates, closes = history
    first_day_only = build_historical_equity_snapshots(
        agent_id=1,
        dates=dates[:1],
        holdings={1: 3},
        instrument_classes={1: "stock"},
        closes={dates[0]: closes[dates[0]]},
        initial_cash={"PLN": Decimal(100)},
    )
    assert first_day_only[0].portfolio_value == Decimal(400)
    assert first_day_only[0].day == -1
