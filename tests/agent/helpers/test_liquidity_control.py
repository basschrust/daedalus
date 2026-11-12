"""Tick-count invariance tests for the shared daily liquidity policy."""

import math

import numpy as np
import pytest
from marketsim.agent.helpers.liquidity_control import (
    LiquidityPolicy,
    daily_intensity,
    expected_daily_events,
    tick_probability,
)


@pytest.mark.parametrize("ticks", [10, 20, 40, 80, 160, 320])
def test_expected_daily_submitted_volume_is_invariant(ticks):
    policy = LiquidityPolicy(daily_expected_events=8.0, daily_submitted_volume=25000)
    expected = expected_daily_events(policy.daily_expected_events, ticks) * policy.mean_quantity(
        ticks
    )
    assert expected == pytest.approx(25000)


@pytest.mark.parametrize("ticks", [10, 20, 40, 80, 160, 320])
def test_arrival_probability_in_range(ticks):
    assert 0 < tick_probability(8, ticks) < 1


def test_legacy_activity_preserved_at_reference_tick_count():
    lam = 0.2
    intensity = daily_intensity(lam, reference_ticks=40)
    assert tick_probability(intensity, 40) == pytest.approx(lam)


def test_monte_carlo_daily_volume_is_stable_across_tick_counts():
    # Orders sampled in native units; test submitted, not executed, volume.
    policy = LiquidityPolicy(daily_expected_events=8, daily_submitted_volume=25000)
    means = []
    for ticks in (20, 40, 80, 160):
        rng = np.random.default_rng(13579)
        arrivals = rng.random((2000, ticks)) < policy.probability(ticks)
        quantities = rng.poisson(policy.mean_quantity(ticks), size=(2000, ticks))
        means.append(float((arrivals * quantities).sum(axis=1).mean()))
    for observed in means:
        assert observed == pytest.approx(25000, rel=0.06)


def test_zero_historical_volume_does_not_generate_orders():
    policy = LiquidityPolicy.from_history(0, 0.1, 0.2)
    assert policy.mean_quantity(40) == 0


@pytest.mark.parametrize("events,ticks", [(-1, 40), (1, 0), (math.nan, 40)])
def test_invalid_intensity_or_ticks_rejected(events, ticks):
    with pytest.raises(ValueError):
        tick_probability(events, ticks)
