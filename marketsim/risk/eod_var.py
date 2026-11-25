"""EoD VaR for historical, delta-normal and Gaussian Monte Carlo methods.

The three estimates share one close-price calibration and one frozen EoD
portfolio. Historical scenarios retain observed cross-asset paths; Monte Carlo
scenarios draw correlated independent daily increments. European options are
repriced using Black--Scholes with the current day's calibrated volatility.

FX is frozen at the EoD exchange rate (currency-translation risk is not modeled).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, isfinite, sqrt
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np

from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut
from marketsim.risk.shared_montecarlo import SharedGaussianPaths

if TYPE_CHECKING:
    from marketsim.agent.agent import Agent
    from marketsim.risk.daily_calibration import DailyRiskCalibration
    from marketsim.simulator.simulator import Simulator


TRADING_DAYS = 252


def _positive(name: str, value: object) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite: {value!r}")
    return number


def _factor(market: object) -> str:
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).upper()


@dataclass(frozen=True)
class Position:
    kind: str
    index: int
    units: float  # position * multiplier * frozen FX rate
    spot: float  # underlying close
    exposure: float  # mark or delta-equivalent in base currency
    maturity: int | None = None
    strike: float = 0.0
    sigma: float = 0.0
    rate: float = 0.0
    call: bool = True


class EodVaR:
    """One calibrated model, one EoD portfolio, all confidence/horizon combinations."""

    def __init__(
        self, calibration: DailyRiskCalibration, config: dict, *, greeks_config: dict | None = None
    ) -> None:
        self.calibration = calibration
        self.calibration_id = calibration.latest_calibration_id
        self.symbols = tuple(sorted(calibration.series))
        self.index = {symbol: i for i, symbol in enumerate(self.symbols)}
        self.rates = {
            str(k).upper(): float(v)
            for k, v in (greeks_config or {}).get("short_rates", {}).items()
        }
        self.dividends = {
            str(k).upper(): float(v)
            for k, v in (greeks_config or {}).get("dividend_yields", {}).items()
        }
        if "confidence_levels" in config:
            levels = config["confidence_levels"]
        else:
            levels = [config.get("confidence_level", 0.99)]
        self.levels = tuple(float(x) for x in levels)
        if not self.levels or any(not 0.5 < x < 1 or not isfinite(x) for x in self.levels):
            raise ValueError("risk_var.confidence_levels must be in (0.5, 1)")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Duplicate risk_var confidence levels")
        raw_horizons = config.get("horizons_days", [1])
        if any(isinstance(x, bool) or not isinstance(x, int) for x in raw_horizons):
            raise ValueError("risk_var.horizons_days must contain integers")
        self.horizons = tuple(raw_horizons)
        if not self.horizons or any(x <= 0 for x in self.horizons):
            raise ValueError("risk_var.horizons_days must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Duplicate risk_var horizons")
        self.scenarios = int(config.get("montecarlo_scenarios", 10000))
        if self.scenarios < max(1000, ceil(10 / (1 - max(self.levels)))):
            raise ValueError("Insufficient Monte Carlo scenarios for selected confidence")
        self.seed = int(config.get("montecarlo_seed", 67))
        self.history_minimum = int(config.get("historical_min_scenarios", 30))
        if self.history_minimum < 2:
            raise ValueError("historical_min_scenarios must be >= 2")

        sessions = sorted(set.intersection(*(set(calibration.series[s]) for s in self.symbols)))[
            -calibration.lookback_sessions :
        ]
        self.prices = np.asarray(
            [[calibration.series[s][session] for s in self.symbols] for session in sessions],
            dtype=float,
        )
        if (
            self.prices.shape[0] < 3
            or not np.isfinite(self.prices).all()
            or (self.prices <= 0).any()
        ):
            raise ValueError("Risk calibration has insufficient valid prices")
        self.returns = np.diff(np.log(self.prices), axis=0)
        self.mean = self.returns.mean(axis=0)
        self.covariance = np.atleast_2d(np.cov(self.returns, rowvar=False, ddof=1))
        self.covariance = (self.covariance + self.covariance.T) / 2
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        tolerance = max(1e-12, float(np.max(abs(eigvals))) * 1e-8)
        if eigvals.min() < -tolerance:
            raise ValueError("Daily risk covariance is not positive semidefinite")
        self.factor = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))
        self.sigmas = np.sqrt(np.maximum(np.diag(self.covariance), 0)) * sqrt(TRADING_DAYS)
        # One shared innovation stream across horizons and agents. Lazy windows
        # retain only requested horizon/expiry checkpoints, not all daily states.
        self._montecarlo = SharedGaussianPaths(
            calibration_id=self.calibration_id,
            seed=self.seed,
            scenarios=self.scenarios,
            mean=self.mean,
            factor=self.factor,
            horizons=self.horizons,
        )
        self._paths: dict[int, tuple[dict[int, np.ndarray], Mapping[int, np.ndarray]]] = {}

    def _convert(self, agent: Agent, amount: float, currency: str, day: int) -> float:
        return float(
            agent.fx_rate_provider.convert(
                amount,
                day=day,
                base=currency,
                quote=agent.base_currency,
            )
        )

    def _portfolio(
        self, agent: Agent, day: int, horizon: int
    ) -> tuple[list[Position], float, float]:
        positions: list[Position] = []
        covered = 0.0
        excluded = 0.0
        for asset_id, quantity in agent.position.items():
            if not quantity:
                continue
            market = agent.markets[asset_id]
            kind = market.instrument_class
            if kind in {"stock", "index"}:
                symbol = _factor(market)
                if symbol not in self.index:
                    raise ValueError(f"Missing calibrated factor for {symbol}")
                spot = _positive(symbol, market.last_traded_price)
                units = self._convert(agent, float(quantity), market.currency, day)
                positions.append(Position("spot", self.index[symbol], units, spot, units * spot))
                covered += abs(units * spot)
            elif kind == "option":
                if str(market.option_type).upper() != "EUROPEAN":
                    raise ValueError("VaR full revaluation supports European options only")
                symbol = _factor(market.underlying)
                if symbol not in self.index:
                    raise ValueError(f"Missing option underlying risk factor {symbol}")
                ccy = getattr(market, "currency", None) or market.underlying.currency
                if ccy.upper() not in self.rates:
                    raise ValueError(f"Missing option risk-free rate for {ccy}")
                if symbol not in self.dividends or self.dividends[symbol] != 0:
                    raise ValueError(f"Explicit zero-dividend assumption required for {symbol}")
                remaining = int(market.expiration) - day
                if remaining <= 0:
                    raise ValueError(f"Nonzero position in expired option {asset_id}")
                side = str(market.option_side).upper()
                if side not in {"CALL", "PUT"}:
                    raise ValueError(f"Unknown option side: {side}")
                spot = _positive(symbol, market.underlying.last_traded_price)
                units = self._convert(agent, float(quantity) * float(market.multiplier), ccy, day)
                sigma = float(self.sigmas[self.index[symbol]])
                rate = self.rates[ccy.upper()]
                if not isfinite(rate) or not isfinite(sigma):
                    raise ValueError("Invalid calibrated option parameters")
                strike = _positive("strike", market.strike)
                call = side == "CALL"
                model = BSCall if call else BSPut
                delta = float(
                    model(
                        S=spot,
                        K=strike,
                        r=rate,
                        volatility=sigma,
                        Time=remaining / TRADING_DAYS,
                        d=0.0,
                    )["delta"]
                )
                mark = float(market.last_traded_price)
                if not isfinite(mark) or mark < 0:
                    raise ValueError(f"Invalid option mark for asset {asset_id}: {mark}")
                positions.append(
                    Position(
                        "option",
                        self.index[symbol],
                        units,
                        spot,
                        units * spot * delta,
                        remaining,
                        strike,
                        sigma,
                        rate,
                        call,
                    )
                )
                covered += abs(units * mark)
            elif kind == "future":
                # Futures are maintained in agent.futures_position instead.
                if agent.futures_position.get(asset_id, 0) != 0:
                    raise ValueError(f"Future also stored in security positions: {asset_id}")
                excluded += abs(
                    self._convert(
                        agent,
                        float(quantity) * float(market.last_traded_price),
                        market.currency,
                        day,
                    )
                )
            else:
                excluded += abs(
                    self._convert(
                        agent,
                        float(quantity) * float(market.last_traded_price),
                        market.currency,
                        day,
                    )
                )
        for asset_id, quantity in agent.futures_position.items():
            if not quantity:
                continue
            market = agent.markets[asset_id]
            symbol = _factor(market.underlying)
            if symbol not in self.index:
                raise ValueError(f"Missing futures underlying risk factor {symbol}")
            remaining = int(market.expiration) - day
            if remaining <= 0:
                raise ValueError(f"Nonzero position in expired future {asset_id}")
            spot = _positive(symbol, market.underlying.last_traded_price)
            units = self._convert(
                agent, float(quantity) * float(market.multiplier), market.currency, day
            )
            positions.append(
                Position(
                    "future",
                    self.index[symbol],
                    units,
                    spot,
                    units * spot,
                    remaining,
                )
            )
            covered += abs(units * spot)
        return positions, covered, excluded

    def _generate(self, horizon: int) -> tuple[dict[int, np.ndarray], Mapping[int, np.ndarray]]:
        if horizon in self._paths:
            return self._paths[horizon]
        if len(self.prices) - horizon < self.history_minimum:
            raise ValueError(
                f"Historical VaR horizon {horizon}: only {len(self.prices) - horizon} "
                f"scenarios (minimum {self.history_minimum})"
            )
        # Historical VaR retains every intermediate stop for early expiry.
        # Monte Carlo checkpoints are generated lazily and shared by all horizons.
        n = len(self.prices) - horizon
        lp = np.log(self.prices)
        historical = {stop: lp[stop : stop + n] - lp[:n] for stop in range(1, horizon + 1)}
        self._paths[horizon] = historical, self._montecarlo.window(horizon)
        return self._paths[horizon]

    def _pnl(
        self, positions: list[Position], paths: Mapping[int, np.ndarray], horizon: int
    ) -> np.ndarray:
        result = np.zeros(next(iter(paths.values())).shape[0], dtype=float)
        for position in positions:
            stop = min(horizon, position.maturity or horizon)
            change = np.exp(paths[stop][:, position.index])
            if position.kind in {"spot", "future"}:
                result += position.units * position.spot * (change - 1)
            elif position.kind == "option":
                model = BSCall if position.call else BSPut
                current = model(
                    S=position.spot,
                    K=position.strike,
                    r=position.rate,
                    volatility=position.sigma,
                    Time=position.maturity / TRADING_DAYS,
                    d=0.0,
                )["price"]
                new_price = model(
                    S=position.spot * change,
                    K=position.strike,
                    r=position.rate,
                    volatility=position.sigma,
                    Time=(position.maturity - stop) / TRADING_DAYS,
                    d=0.0,
                )["price"]
                result += position.units * (new_price - current)
            else:
                raise ValueError(f"Unknown position kind: {position.kind}")
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite scenario PnL")
        return result

    def _normal(self, positions: list[Position], horizon: int, confidence: float) -> float:
        # Linear approximation and deterministic option theta-to-expiry.
        durations = [min(horizon, p.maturity or horizon) for p in positions]
        exposures = np.asarray([p.exposure for p in positions], dtype=float)
        indexes = [p.index for p in positions]
        drift = sum(e * self.mean[idx] * d for e, idx, d in zip(exposures, indexes, durations))
        for p, d in zip(positions, durations):
            if p.kind == "option":
                model = BSCall if p.call else BSPut
                old = model(
                    S=p.spot,
                    K=p.strike,
                    r=p.rate,
                    volatility=p.sigma,
                    Time=p.maturity / TRADING_DAYS,
                    d=0.0,
                )["price"]
                new = model(
                    S=p.spot,
                    K=p.strike,
                    r=p.rate,
                    volatility=p.sigma,
                    Time=(p.maturity - d) / TRADING_DAYS,
                    d=0.0,
                )["price"]
                # The shared Black-Scholes library returns Price at scalar expiry
                # and float before expiry. Use numeric premiums for VaR arithmetic;
                # keep Price precision semantics unchanged for market valuation.
                drift += p.units * (float(new) - float(old))
        variance = 0.0
        total_absolute = 0.0
        for i, exposure in enumerate(exposures):
            for j, other in enumerate(exposures):
                contribution = (
                    exposure
                    * other
                    * self.covariance[indexes[i], indexes[j]]
                    * min(durations[i], durations[j])
                )
                variance += contribution
                total_absolute += abs(contribution)
        tolerance = max(1e-8, 1e-9 * total_absolute)
        if variance < -tolerance:
            raise ValueError("Negative parametric portfolio variance")
        return max(0.0, float(NormalDist().inv_cdf(confidence) * sqrt(max(variance, 0)) - drift))

    def rows_for_agent(self, agent: Agent, day: int) -> dict[str, list[dict]]:
        output: dict[str, list[dict]] = {
            "parametric": [],
            "historical": [],
            "montecarlo": [],
        }
        for horizon in self.horizons:
            positions, covered, excluded = self._portfolio(agent, day, horizon)
            hist_paths, mc_paths = self._generate(horizon)
            historical_losses = -self._pnl(positions, hist_paths, horizon)
            montecarlo_losses = -self._pnl(positions, mc_paths, horizon)
            for confidence in self.levels:
                params = dict(
                    day=day,
                    agent_id=int(agent.agent_id),
                    confidence_level=confidence,
                    horizon_days=horizon,
                    portfolio_value=float(agent.portfolio_value),
                    base_currency=str(agent.base_currency),
                    covered_exposure=covered,
                    excluded_exposure=excluded,
                    calibration_id=self.calibration_id,
                )
                output["parametric"].append(
                    dict(
                        params,
                        var_amount=self._normal(positions, horizon, confidence),
                    )
                )
                output["historical"].append(
                    dict(
                        params,
                        var_amount=max(
                            0.0,
                            float(
                                np.quantile(
                                    historical_losses,
                                    confidence,
                                    method="linear",
                                )
                            ),
                        ),
                    )
                )
                output["montecarlo"].append(
                    dict(
                        params,
                        var_amount=max(
                            0.0,
                            float(
                                np.quantile(
                                    montecarlo_losses,
                                    confidence,
                                    method="linear",
                                )
                            ),
                        ),
                    )
                )
        return output


def write_eod_var(
    simulator: Simulator, day: int, calibration: DailyRiskCalibration, config: dict
) -> None:
    """Persist three method tables using exactly the same calibrated daily ID."""
    engine = EodVaR(calibration, config, greeks_config=simulator.risk_greeks)
    rows: dict[str, list[dict]] = {"parametric": [], "historical": [], "montecarlo": []}
    for agent in simulator.agents.values():
        agent_rows = engine.rows_for_agent(agent, day)
        for method in rows:
            rows[method].extend(agent_rows[method])
    simulator.repository.save_eod_agent_var_parametric(rows["parametric"])
    simulator.repository.save_eod_agent_var_historical(rows["historical"])
    simulator.repository.save_eod_agent_var_montecarlo(rows["montecarlo"])
