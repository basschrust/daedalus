"""Shared, reproducible Gaussian daily returns for all Monte Carlo VaR horizons.

One daily innovation stream is used for every requested holding horizon. This
preserves the joint distribution of each horizon while introducing useful
common-random-number coupling across horizon estimates.

Only requested checkpoints are retained, not the whole scenario cube. An
additional checkpoint (e.g. derivative expiry) may be requested later. Daily
innovations have stable, day-specific seeds, so checkpoints are unchanged by
request order, agent order, or extending the maximum holding horizon.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256

import numpy as np


class SharedGaussianPaths:
    """A single collection of jointly Gaussian daily innovations.

    Parameters are the daily log-return mean and a factor such that
    ``factor @ factor.T`` equals the daily covariance matrix.
    """

    def __init__(
        self,
        *,
        calibration_id: str,
        seed: int,
        scenarios: int,
        mean: np.ndarray,
        factor: np.ndarray,
        horizons: Sequence[int],
    ) -> None:
        self.calibration_id = str(calibration_id)
        self.seed = int(seed)
        self.scenarios = int(scenarios)
        self.mean = np.asarray(mean, dtype=float)
        self.factor = np.asarray(factor, dtype=float)
        if self.scenarios <= 0 or self.mean.ndim != 1:
            raise ValueError("Invalid Monte Carlo dimensions")
        n = self.mean.size
        if n < 1 or self.factor.shape != (n, n):
            raise ValueError("Monte Carlo factor must be a square risk-factor matrix")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.factor).all():
            raise ValueError("Nonfinite Monte Carlo calibration")
        if not horizons or any(
            isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons
        ):
            raise ValueError("Monte Carlo horizons must be positive integers")
        self.checkpoints = frozenset(horizons)
        self._cached: dict[int, np.ndarray] = {0: np.zeros((self.scenarios, n))}

    def _daily_shock(self, day: int) -> np.ndarray:
        # Do not include the requested horizon in the seed. This guarantees that
        # expanding the horizon set does not change shorter-horizon paths.
        digest = sha256(f"{self.calibration_id}|{self.seed}|day={day}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return self.mean + rng.standard_normal((self.scenarios, self.mean.size)) @ self.factor.T

    def at(self, stop: int) -> np.ndarray:
        """Return cumulative log returns at trading day ``stop``.

        Later checkpoint requests can be satisfied from the nearest preceding
        cached checkpoint. If an earlier maturity is requested after a later
        horizon, deterministic daily draws reproduce that exact prefix.
        """
        if isinstance(stop, bool) or not isinstance(stop, int) or stop < 1:
            raise ValueError("Monte Carlo checkpoint must be a positive integer")
        if stop in self._cached:
            return self._cached[stop]

        anchor = max(k for k in self._cached if k < stop)
        cumulative = self._cached[anchor].copy()
        # Cache all configured horizon checkpoints along the way. This avoids
        # re-drawing them if callers request horizons in a different order.
        targets = sorted(k for k in self.checkpoints if anchor < k < stop) + [stop]
        current = anchor
        for target in targets:
            for day in range(current + 1, target + 1):
                cumulative += self._daily_shock(day)
            self._cached[target] = cumulative.copy()
            current = target
        return self._cached[stop]

    def window(self, horizon: int) -> MonteCarloWindow:
        return MonteCarloWindow(self, horizon)

    @property
    def cached_checkpoints(self) -> tuple[int, ...]:
        """Debug/diagnostic hook: memory is proportional to this set's size."""
        return tuple(sorted(k for k in self._cached if k > 0))


class MonteCarloWindow(Mapping[int, np.ndarray]):
    """Lazy path-prefix view, compatible with the existing VaR _pnl routine."""

    def __init__(self, paths: SharedGaussianPaths, horizon: int) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Monte Carlo horizon must be a positive integer")
        self._paths = paths
        self.horizon = horizon

    def __getitem__(self, stop: int) -> np.ndarray:
        if isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= self.horizon:
            raise KeyError(stop)
        return self._paths.at(stop)

    def __iter__(self) -> Iterator[int]:
        return iter(range(1, self.horizon + 1))

    def __len__(self) -> int:
        return self.horizon
