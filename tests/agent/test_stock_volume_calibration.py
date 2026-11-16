import pandas as pd
import pytest

from marketsim.agent.stock_volume_calibration import calibrated_mean_order_size


def sizing(volumes, **kwargs):
    defaults = dict(fraction=0.02, ticks_per_day=25, arrival_probability=0.4, fallback=150.0)
    defaults.update(kwargs)
    return calibrated_mean_order_size(
        historical_ohlcv=pd.DataFrame({"volume": volumes}), **defaults
    )


def test_order_sizes_scale_linearly_with_turnover():
    assert sizing([5_000_000] * 10) / sizing([50_000] * 10) == pytest.approx(100)


def test_daily_expected_submissions_match_participation():
    mean = sizing([5_000_000] * 10)
    assert mean * 25 * 0.4 == pytest.approx(100_000)


def test_regime_scales_order_size():
    assert sizing([100_000], regime_factor=1.5) == pytest.approx(1.5 * sizing([100_000]))


def test_missing_history_falls_back():
    assert (
        calibrated_mean_order_size(
            historical_ohlcv=None,
            fraction=0.02,
            ticks_per_day=25,
            arrival_probability=0.4,
            fallback=150,
        )
        == 150
    )


def test_zero_volume_produces_zero_order_size():
    assert sizing([0, 0]) == 0


@pytest.mark.parametrize("volumes", [[-1, 10], [float("inf"), 10]])
def test_invalid_history_rejected(volumes):
    with pytest.raises(ValueError):
        sizing(volumes)


def test_invalid_fraction_rejected():
    with pytest.raises(ValueError):
        sizing([1000], fraction=0)
