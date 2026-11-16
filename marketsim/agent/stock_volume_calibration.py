"""Historical-turnover-based stock order sizing; independent of trade matching."""

from __future__ import annotations

import math

import pandas as pd


def calibrated_mean_order_size(
    *,
    historical_ohlcv,
    fraction: float,
    ticks_per_day: int,
    arrival_probability: float,
    fallback: float,
    regime_factor: float = 1.0,
) -> float:
    """Expected shares per submitted order for a single stock agent.

    A fraction of average historical daily turnover is allocated to this agent.
    The regime factor scales expected participation, not recorded executions.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if ticks_per_day <= 0 or not 0 < arrival_probability <= 1:
        raise ValueError("invalid simulation ticks or arrival probability")
    if not math.isfinite(regime_factor) or regime_factor <= 0:
        raise ValueError("regime_factor must be positive and finite")
    if not math.isfinite(fallback) or fallback < 0:
        raise ValueError("fallback must be nonnegative and finite")
    if historical_ohlcv is None or historical_ohlcv.empty or "volume" not in historical_ohlcv:
        return fallback
    volumes = pd.to_numeric(historical_ohlcv["volume"], errors="coerce")
    valid = volumes.dropna()
    if valid.empty or (valid < 0).any() or not valid.map(math.isfinite).all():
        raise ValueError("invalid historical stock volume")
    historical_daily_volume = float(valid.mean())
    if historical_daily_volume == 0:
        return 0.0
    return (
        historical_daily_volume * fraction * regime_factor / (ticks_per_day * arrival_probability)
    )
