"""Daily-scale liquidity policy shared by order-generating agents.

Quantities are in native units (shares or derivative contracts). The policy
never modifies executed trades; it sizes *submitted* orders only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def tick_probability(daily_expected_events: float, ticks_per_day: int) -> float:
    """Poisson-arrival probability for one tick; stable across tick counts."""
    if ticks_per_day < 1 or not math.isfinite(daily_expected_events) or daily_expected_events < 0:
        raise ValueError("Invalid daily event intensity or ticks_per_day")
    return -math.expm1(-daily_expected_events / ticks_per_day)


def daily_intensity(reference_tick_probability: float, reference_ticks: int = 40) -> float:
    """Convert legacy per-tick activity into a daily Poisson intensity."""
    if reference_ticks < 1 or not 0 <= reference_tick_probability < 1:
        raise ValueError("Invalid reference activity or reference_ticks")
    return -reference_ticks * math.log1p(-reference_tick_probability)


def expected_daily_events(daily_expected_events: float, ticks_per_day: int) -> float:
    """Expected count when at most one event is permitted per tick."""
    return ticks_per_day * tick_probability(daily_expected_events, ticks_per_day)


def mean_order_quantity(
    daily_submitted_volume: float, daily_expected_events: float, ticks_per_day: int
) -> float:
    """Mean native-unit quantity per arrival, accounting for tick discretization."""
    if not math.isfinite(daily_submitted_volume) or daily_submitted_volume < 0:
        raise ValueError("Invalid daily volume")
    events = expected_daily_events(daily_expected_events, ticks_per_day)
    return daily_submitted_volume / events if events else 0.0


@dataclass(frozen=True)
class LiquidityPolicy:
    """Agent/instrument daily order budget; not a cap on market-wide executions."""

    daily_expected_events: float
    daily_submitted_volume: float

    def probability(self, ticks_per_day: int) -> float:
        return tick_probability(self.daily_expected_events, ticks_per_day)

    def mean_quantity(self, ticks_per_day: int) -> float:
        return mean_order_quantity(
            self.daily_submitted_volume, self.daily_expected_events, ticks_per_day
        )

    @classmethod
    def from_history(
        cls,
        historical_daily_volume: float,
        participation_fraction: float,
        legacy_lam: float,
        reference_ticks: int = 40,
    ) -> "LiquidityPolicy":
        if not math.isfinite(historical_daily_volume) or historical_daily_volume < 0:
            raise ValueError("Invalid historical daily volume")
        if not math.isfinite(participation_fraction) or not 0 <= participation_fraction <= 1:
            raise ValueError("Invalid participation fraction")
        return cls(
            daily_intensity(legacy_lam, reference_ticks),
            historical_daily_volume * participation_fraction,
        )
