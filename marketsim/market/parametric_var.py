"""One-day delta-normal stock VaR; derivatives are deliberately excluded."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from marketsim.market.fundamental_return_regime import FundamentalReturnRegime


class ParametricVaR:
    def __init__(self, calibration_id, regime, confidence_level=0.99):
        if not 0.5 < confidence_level < 1:
            raise ValueError("Invalid confidence level")
        self.calibration_id = calibration_id
        self.confidence_level = confidence_level
        self.z = NormalDist().inv_cdf(confidence_level)
        self.symbols = regime.symbols
        self.covariance = regime.covariance
        self.index = {s: i for i, s in enumerate(self.symbols)}

    @classmethod
    def load(
        cls, *, repository, calibration_id, symbols, confidence_level=0.99, source_connection=None
    ):
        """Read Snowflake L2 or copy its calibration once to DuckDB local tables."""
        instrument_rows, pair_rows = repository.load_parametric_calibration(
            calibration_id, source_connection=source_connection
        )
        regime = FundamentalReturnRegime.from_statistics_rows(
            calibration_id, instrument_rows, pair_rows, symbols=symbols
        )
        return cls(calibration_id, regime, confidence_level)

    def row_for_agent(self, agent, day):
        exposure = np.zeros(len(self.symbols))
        covered = 0.0
        excluded = 0.0
        for asset_id, quantity in agent.position.items():
            if not quantity:
                continue
            market = agent.markets[asset_id]
            multiplier = (
                getattr(market, "multiplier", 1) if market.instrument_class == "option" else 1
            )
            value = float(
                agent.fx_rate_provider.convert(
                    float(quantity) * float(market.last_traded_price) * multiplier,
                    day=day,
                    base=market.currency,
                    quote=agent.base_currency,
                )
            )
            if (
                market.instrument_class == "stock"
                and getattr(market, "short_name", "").upper() in self.index
            ):
                exposure[self.index[market.short_name.upper()]] += value
                covered += abs(value)
            else:
                excluded += abs(value)
        for asset_id, quantity in agent.futures_position.items():
            if quantity:
                market = agent.markets[asset_id]
                excluded += abs(
                    float(
                        agent.fx_rate_provider.convert(
                            float(quantity)
                            * float(market.last_traded_price)
                            * float(market.multiplier),
                            day=day,
                            base=market.currency,
                            quote=agent.base_currency,
                        )
                    )
                )
        variance = float(exposure @ self.covariance @ exposure)
        if variance < -1e-7:
            raise ValueError("Negative portfolio variance")
        return dict(
            day=day,
            agent_id=agent.agent_id,
            confidence_level=self.confidence_level,
            horizon_days=1,
            var_amount=self.z * np.sqrt(max(0.0, variance)),
            portfolio_value=float(agent.portfolio_value),
            base_currency=agent.base_currency,
            covered_exposure=covered,
            excluded_exposure=excluded,
            calibration_id=self.calibration_id,
        )
