"""Shared persistent, correlated daily activity factors.

Mean-one lognormal factors preserve the long-run volume level while increasing
day-to-day dispersion. Market and stock components each follow a stationary
Gaussian AR(1) process in log space.
"""

from __future__ import annotations

import math

import numpy as np


class VolumeRegime:
    def __init__(
        self,
        *,
        seed: int = 67,
        market_sigma: float = 0.55,
        stock_sigma: float = 0.30,
        market_persistence: float = 0.65,
        stock_persistence: float = 0.35,
    ):
        for name, value in (
            ("market_sigma", market_sigma),
            ("stock_sigma", stock_sigma),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("market_persistence", market_persistence),
            ("stock_persistence", stock_persistence),
        ):
            if not math.isfinite(value) or not -1 < value < 1:
                raise ValueError(f"{name} must be between -1 and 1")

        self.rng = np.random.default_rng(seed)
        self.market_sigma = float(market_sigma)
        self.stock_sigma = float(stock_sigma)
        self.market_persistence = float(market_persistence)
        self.stock_persistence = float(stock_persistence)
        self.market_factors: dict[int, float] = {}
        self.stock_factors: dict[tuple[int, int], float] = {}
        self._market_z: dict[int, float] = {}
        self._stock_z: dict[tuple[int, int], float] = {}

    def _market_state(self, day: int) -> float:
        if day not in self.market_factors:
            rho = self.market_persistence
            z = 0.0
            for _ in range(day + 1):
                z = rho * z + math.sqrt(1 - rho * rho) * float(self.rng.standard_normal())
            self.market_factors[day] = math.exp(self.market_sigma * z - 0.5 * self.market_sigma**2)
        return self.market_factors[day]

    def _stock_state(self, day: int, asset_id: int) -> float:
        for d in range(day + 1):
            key = (d, asset_id)
            if key not in self._stock_z:
                previous = self._stock_z.get((d - 1, asset_id), 0.0)
                rho = self.stock_persistence
                self._stock_z[key] = rho * previous + math.sqrt(1 - rho * rho) * float(
                    self.rng.standard_normal()
                )
                self.stock_factors[key] = math.exp(
                    self.stock_sigma * self._stock_z[key] - 0.5 * self.stock_sigma**2
                )
        return self.stock_factors[(day, asset_id)]

    def factor(self, day: int, asset_id: int) -> float:
        if not isinstance(day, int) or day < 0:
            raise ValueError("day must be a nonnegative integer")
        return self._market_state(day) * self._stock_state(day, asset_id)
