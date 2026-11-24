"""Regression tests for shared Monte Carlo innovations, checkpointing and VaR."""

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from marketsim.risk.eod_var import EodVaR
from marketsim.risk.shared_montecarlo import SharedGaussianPaths


def paths(*, horizons=(1, 5, 10), seed=67, n=20_000):
    mean = np.array([0.0007, -0.0004])
    factor = np.array([[0.018, 0.0], [0.007, 0.012]])
    return SharedGaussianPaths(
        calibration_id="SIM-20261124-test",
        seed=seed,
        scenarios=n,
        mean=mean,
        factor=factor,
        horizons=horizons,
    )


def test_marginal_gaussian_statistics_are_unbiased_by_path_reuse():
    model = paths()
    covariance = model.factor @ model.factor.T
    for horizon in (1, 5, 10):
        samples = model.at(horizon)
        expected_mean = horizon * model.mean
        expected_covariance = horizon * covariance
        mean_standard_error = np.sqrt(np.diag(expected_covariance) / model.scenarios)
        assert np.all(np.abs(samples.mean(axis=0) - expected_mean) <= 5 * mean_standard_error)
        np.testing.assert_allclose(
            np.cov(samples, rowvar=False), expected_covariance, rtol=0.06, atol=0
        )


def test_horizons_share_prefix_increments_and_are_order_independent():
    first = paths(horizons=(10, 1, 5))
    ten = first.at(10)
    one = first.at(1)
    five = first.at(5)
    assert first.at(1) is one
    assert first.at(10) is ten
    assert five is first.window(5)[5]
    np.testing.assert_array_equal(ten - five, first.at(10) - first.at(5))

    second = paths(horizons=(1, 5))
    np.testing.assert_array_equal(one, second.at(1))
    np.testing.assert_array_equal(five, second.at(5))
    # Distinct horizons are coupled through their scenario identity, not equal.
    assert not np.array_equal(one, five)
    # The later increment is independent of the earlier cumulative return.
    correlation = np.corrcoef(one[:, 0], (five - one)[:, 0])[0, 1]
    assert abs(correlation) < 0.04
    # No per-day snapshots for days 2,3,4,6,7,8,9 are retained.
    assert first.cached_checkpoints == (1, 5, 10)


def test_late_requested_expiry_prefix_is_identical():
    model = paths(horizons=(1, 10))
    model.at(10)
    expiry = model.at(3)  # requested by a second agent after 10-day paths exist

    ref = paths(horizons=(1, 3, 10))
    np.testing.assert_array_equal(expiry, ref.at(3))
    assert model.cached_checkpoints == (1, 3, 10)


def test_seed_and_calibration_control_scenarios():
    one = paths(seed=67)
    same = paths(seed=67)
    different = paths(seed=68)
    np.testing.assert_array_equal(one.at(5), same.at(5))
    assert not np.array_equal(one.at(5), different.at(5))


def test_window_is_lazy_and_rejects_out_of_horizon_access():
    model = paths(horizons=(1, 5, 10))
    window = model.window(5)
    assert len(window) == 5
    assert list(window) == [1, 2, 3, 4, 5]
    assert model.cached_checkpoints == ()
    assert next(iter(window.values())).shape == (model.scenarios, 2)
    assert model.cached_checkpoints == (1,)
    with pytest.raises(KeyError):
        _ = window[6]


def _calibration():
    n = 80
    x = np.arange(n, dtype=float)
    series = 101.0 * np.exp(0.001 * x + 0.015 * np.sin(x * 0.53))
    sessions = [date(2026, 4, 1) + timedelta(days=i) for i in range(n)]
    return SimpleNamespace(
        latest_calibration_id="SIM-20261124-integration",
        series={"AAA": dict(zip(sessions, series))},
        lookback_sessions=n,
    )


def _agent():
    stock = SimpleNamespace(
        instrument_class="stock",
        symbol="AAA",
        last_traded_price=105.0,
        currency="PLN",
    )
    return SimpleNamespace(
        agent_id=9,
        base_currency="PLN",
        portfolio_value=100000,
        position={1: 20},
        futures_position={},
        markets={1: stock},
        fx_rate_provider=SimpleNamespace(convert=lambda amount, **kwargs: amount),
    )


def test_montecarlo_var_at_shared_horizons_unchanged_by_longer_horizon_config():
    common = {
        "confidence_levels": [0.95, 0.99],
        "montecarlo_scenarios": 3000,
        "montecarlo_seed": 67,
    }
    short = EodVaR(_calibration(), {**common, "horizons_days": [1, 5]})
    long = EodVaR(_calibration(), {**common, "horizons_days": [1, 5, 10]})
    short_rows = short.rows_for_agent(_agent(), 1)["montecarlo"]
    long_rows = long.rows_for_agent(_agent(), 1)["montecarlo"]

    def indexed(rows):
        return {(r["horizon_days"], r["confidence_level"]): r["var_amount"] for r in rows}

    for key, value in indexed(short_rows).items():
        assert indexed(long_rows)[key] == value
    assert long._montecarlo.cached_checkpoints == (1, 5, 10)
    # Different agents can share the very same sampled returns at every horizon.
    assert long.rows_for_agent(_agent(), 1)["montecarlo"] == long_rows
