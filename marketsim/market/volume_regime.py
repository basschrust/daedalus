import numpy as np


class VolumeRegime:
    """Correlated daily market and stock activity factors."""

    def __init__(
        self,
        *,
        seed: int = 67,
        market_sigma: float = 0.55,
        stock_sigma: float = 0.30,
    ):
        self.rng = np.random.default_rng(seed)
        self.market_sigma = market_sigma
        self.stock_sigma = stock_sigma

        self.market_factors = {}
        self.stock_factors = {}

    def _draw_factor(self, sigma):
        # Mean-one lognormal distribution.
        return float(
            self.rng.lognormal(
                mean=-0.5 * sigma**2,
                sigma=sigma,
            )
        )

    def factor(self, day: int, asset_id: int) -> float:
        if day not in self.market_factors:
            self.market_factors[day] = self._draw_factor(self.market_sigma)

        key = (day, asset_id)

        if key not in self.stock_factors:
            self.stock_factors[key] = self._draw_factor(self.stock_sigma)

        return self.market_factors[day] * self.stock_factors[key]
