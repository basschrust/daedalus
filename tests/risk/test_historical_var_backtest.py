"""Historical point-in-time VaR -> frozen-book HPL -> Kupiec pipeline."""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pytest

from marketsim.risk.historical_portfolios import build_historical_equity_snapshots
from marketsim.risk.historical_var_backtest import walk_forward_equity_var


def _inputs(*, sessions: int = 43):
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(sessions)]
    prices = {
        session: {
            1: Decimal(str(round(100 * np.exp(0.002 * i + 0.02 * np.sin(i * 0.6)), 6))),
            2: Decimal(str(round(90 * np.exp(0.001 * i + 0.03 * np.cos(i * 0.3)), 6))),
        }
        for i, session in enumerate(dates)
    }
    snapshots = build_historical_equity_snapshots(
        agent_id=5,
        dates=dates,
        holdings={1: 10, 2: -4},
        instrument_classes={1: "stock", 2: "stock"},
        closes=prices,
        initial_cash={"PLN": Decimal("2500")},
    )
    return dates, prices, snapshots


def _run(*, prices, snapshots):
    return walk_forward_equity_var(
        snapshots=snapshots,
        closes=prices,
        symbols={1: "AAA", 2: "BBB"},
        instrument_classes={1: "stock", 2: "stock"},
        risk_config={
            "confidence_levels": [0.95, 0.99],
            "historical_min_scenarios": 30,
            "montecarlo_scenarios": 1500,
            "montecarlo_seed": 42,
        },
        lookback_sessions=35,
        group_names={5: "TEST"},
    )


def test_all_three_var_models_produce_kupiec_observations() -> None:
    dates, prices, snapshots = _inputs()
    forecasts, observations, report = _run(prices=prices, snapshots=snapshots)
    assert len(forecasts) == (len(dates) - 31) * 3 * 2
    assert len(observations) == len(forecasts)
    assert set(forecasts.method) == {"historical", "parametric", "montecarlo"}
    assert set(forecasts.confidence_level) == {0.95, 0.99}
    assert set(forecasts.horizon_days) == {1}
    assert set(forecasts.group_name) == {"TEST"}
    assert set(report.method) == {"historical", "parametric", "montecarlo"}
    assert len(report) == 6
    assert all(forecasts.var_amount >= 0)
    # Forecast day t uses the exact session t, not t+1.
    assert forecasts.iloc[0].as_of_date == dates[30]
    assert forecasts.iloc[0].day == snapshots[30].day
    assert set(report.assessment) == {"LOW_POWER"}


def test_future_price_shock_does_not_change_previous_forecast() -> None:
    dates, prices, snapshots = _inputs()
    first, _, _ = _run(prices=prices, snapshots=snapshots)
    altered = {session: dict(row) for session, row in prices.items()}
    altered[dates[-1]][1] *= Decimal("20")
    # The model inputs at day t are untouched by a price jump at the final day.
    # The final snapshot must be valued at the altered close too. Otherwise the
    # input is inconsistent, and the backtest rightly rejects it.
    last = snapshots[-1]
    change = last.positions[1] * (altered[dates[-1]][1] - prices[dates[-1]][1])
    repriced = [
        *snapshots[:-1],
        replace(last, portfolio_value=last.portfolio_value + change),
    ]
    second, _, _ = _run(prices=altered, snapshots=repriced)
    assert first.equals(second)


def test_insufficient_training_history_fails_explicitly() -> None:
    _, prices, snapshots = _inputs(sessions=30)
    with pytest.raises(ValueError, match="at least 32"):
        _run(prices=prices, snapshots=snapshots)


def test_missing_factor_close_fails_explicitly() -> None:
    dates, prices, snapshots = _inputs()
    altered = {session: dict(row) for session, row in prices.items()}
    del altered[dates[25]][2]
    with pytest.raises(ValueError, match="Missing close for asset 2"):
        _run(prices=altered, snapshots=snapshots)


def test_no_rebalancing_and_no_future_position() -> None:
    _, prices, snapshots = _inputs()
    _, observations, _ = _run(prices=prices, snapshots=snapshots)
    assert observations.hypothetical_pnl.notna().all()
    assert not observations.exception.isna().any()
    assert set(observations.agent_id) == {5}
