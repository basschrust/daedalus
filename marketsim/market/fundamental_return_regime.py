"""Reproducible correlated daily fundamental log returns for stock markets.

The regime does not modify traded prices or submit orders. Callers decide how
fundamental values affect their agents' valuations and order flow.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


class FundamentalReturnRegime:
    """Daily multivariate-normal shocks calibrated to Ikaros statistics.

    All symbols are canonicalized to uppercase and sorted to make the RNG
    sequence independent of SQL row order. Days must be nonnegative integers.
    """

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        mean_daily_returns: Mapping[str, float],
        covariance_daily: np.ndarray,
        seed: int = 67,
    ) -> None:
        original = [str(s).upper() for s in symbols]
        if not original or len(original) != len(set(original)):
            raise ValueError("Expected nonempty, unique stock symbols")
        if set(original) != {str(k).upper() for k in mean_daily_returns}:
            raise ValueError("Mean-return symbols must exactly match covariance symbols")

        matrix = np.asarray(covariance_daily, dtype=float)
        n = len(original)
        if matrix.shape != (n, n) or not np.isfinite(matrix).all():
            raise ValueError("Invalid covariance matrix dimensions or values")
        if not np.allclose(matrix, matrix.T, rtol=1e-9, atol=1e-12):
            raise ValueError("Covariance matrix is not symmetric")
        matrix = (matrix + matrix.T) / 2
        permutation = np.argsort(original)
        self.symbols = tuple(original[i] for i in permutation)
        self.mean = np.array(
            [
                float({str(k).upper(): v for k, v in mean_daily_returns.items()}[s])
                for s in self.symbols
            ],
            dtype=float,
        )
        self.covariance = matrix[np.ix_(permutation, permutation)]
        if not np.isfinite(self.mean).all():
            raise ValueError("Nonfinite mean returns")
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        tolerance = max(1e-12, float(np.max(np.abs(eigvals))) * 1e-8)
        if eigvals.min() < -tolerance:
            raise ValueError("Covariance matrix is not positive semidefinite")
        # Eigenfactor handles legitimately singular covariance matrices.
        self.factor = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0)))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._returns: list[np.ndarray] = []
        self._log_values: list[np.ndarray] = [np.zeros(n)]
        self._index = {s: i for i, s in enumerate(self.symbols)}

    def _through(self, day: int) -> None:
        if not isinstance(day, int) or day < 0:
            raise ValueError("day must be a nonnegative integer")
        while len(self._returns) <= day:
            daily = self.mean + self.factor @ self._rng.standard_normal(len(self.symbols))
            self._returns.append(daily)
            self._log_values.append(self._log_values[-1] + daily)

    def returns(self, day: int) -> dict[str, float]:
        """One log-return per stock, cached and stable for the given day."""
        self._through(day)
        return dict(zip(self.symbols, self._returns[day].tolist()))

    def return_for(self, day: int, symbol: str) -> float:
        self._through(day)
        return float(self._returns[day][self._index[symbol.upper()]])

    def value(self, day: int, symbol: str, initial_price: float) -> float:
        """Fundamental price after applying shocks for days 0 through day."""
        if not math.isfinite(initial_price) or initial_price <= 0:
            raise ValueError("initial_price must be positive and finite")
        self._through(day)
        return float(initial_price * np.exp(self._log_values[day + 1][self._index[symbol.upper()]]))

    @classmethod
    def from_snowflake(cls, conn, calibration_id: str, *, symbols: Sequence[str], seed: int = 67):
        """Load an exact calibration; requires SELECT on DAEDALUS.MART L2 tables."""
        canonical = [str(s).upper() for s in symbols]
        if not calibration_id or not canonical or len(set(canonical)) != len(canonical):
            raise ValueError("Provide calibration_id and unique stock symbols")
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
                "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS "
                "WHERE CALIBRATION_ID = %s",
                (calibration_id,),
            )
            instrument_rows = cursor.fetchall()
            cursor.execute(
                "SELECT SYMBOL_1, SYMBOL_2, COVARIANCE_DAILY "
                "FROM DAEDALUS.MART.L2_PAIR_STATISTICS "
                "WHERE CALIBRATION_ID = %s",
                (calibration_id,),
            )
            pair_rows = cursor.fetchall()

        wanted = set(canonical)
        instruments = {}
        for symbol, mean, sigma in instrument_rows:
            symbol = str(symbol).upper()
            if symbol in wanted:
                if symbol in instruments or mean is None or sigma is None:
                    raise ValueError(f"Duplicate or incomplete instrument statistics: {symbol}")
                sigma = float(sigma)
                if not math.isfinite(sigma) or sigma < 0:
                    raise ValueError(f"Invalid sigma: {symbol}")
                instruments[symbol] = (float(mean), sigma)
        if set(instruments) != wanted:
            raise ValueError(f"Missing instrument statistics: {sorted(wanted - set(instruments))}")

        ordered = sorted(wanted)
        positions = {symbol: i for i, symbol in enumerate(ordered)}
        covariance = np.zeros((len(ordered), len(ordered)), dtype=float)
        for symbol, (_, sigma) in instruments.items():
            i = positions[symbol]
            covariance[i, i] = sigma * sigma

        observed_pairs = set()
        for a, b, cov in pair_rows:
            a, b = str(a).upper(), str(b).upper()
            if a not in wanted or b not in wanted:
                continue
            pair = tuple(sorted((a, b)))
            if pair in observed_pairs:
                raise ValueError(f"Duplicate covariance pair: {pair}")
            observed_pairs.add(pair)
            if cov is None or not math.isfinite(float(cov)):
                raise ValueError(f"Invalid covariance: {pair}")
            i, j = positions[a], positions[b]
            covariance[i, j] = covariance[j, i] = float(cov)
            if a == b and not np.isclose(float(cov), instruments[a][1] ** 2, rtol=1e-5, atol=1e-12):
                raise ValueError(f"Diagonal covariance disagrees with sigma: {a}")

        expected = len(ordered) * (len(ordered) - 1) // 2
        if len({p for p in observed_pairs if p[0] != p[1]}) != expected:
            raise ValueError("Missing off-diagonal covariance pairs")
        return cls(
            symbols=ordered,
            mean_daily_returns={s: instruments[s][0] for s in ordered},
            covariance_daily=covariance,
            seed=seed,
        )
