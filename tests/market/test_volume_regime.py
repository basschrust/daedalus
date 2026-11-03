import numpy as np
import pytest

from marketsim.market.volume_regime import VolumeRegime


def test_same_day_asset_returns_same_factor():
    regime = VolumeRegime(seed=42)

    first = regime.factor(day=1, asset_id=101)
    second = regime.factor(day=1, asset_id=101)

    assert first == second


def test_factors_are_positive():
    regime = VolumeRegime(seed=42)

    factors = [regime.factor(day, asset_id) for day in range(20) for asset_id in (101, 102, 103)]

    assert all(np.isfinite(x) and x > 0 for x in factors)


def test_reproducibility():
    a = VolumeRegime(seed=42)
    b = VolumeRegime(seed=42)

    for day in range(30):
        for asset_id in (101, 102, 103):
            assert a.factor(day, asset_id) == b.factor(day, asset_id)


def test_market_factor_is_shared_across_assets():
    regime = VolumeRegime(seed=42)

    regime.factor(day=5, asset_id=101)
    regime.factor(day=5, asset_id=102)

    assert len(regime.market_factors) == 1
    assert len(regime.stock_factors) == 2


def test_daily_volume_variability():
    regime = VolumeRegime(seed=42)

    factors = np.array([regime.factor(day, 101) for day in range(500)])

    cv = factors.std() / factors.mean()

    # Broad sanity bounds, not a market calibration target.
    assert 0.35 < cv < 1.5


def test_zero_volatility_produces_constant_activity():
    regime = VolumeRegime(
        seed=42,
        market_sigma=0,
        stock_sigma=0,
    )

    factors = [regime.factor(day, 101) for day in range(30)]

    assert factors == [1.0] * 30


def test_unconditional_mean_is_approximately_one():
    regime = VolumeRegime(seed=42)

    factors = np.array([regime.factor(day, 101) for day in range(2000)])

    assert factors.mean() == pytest.approx(1.0, abs=0.08)
