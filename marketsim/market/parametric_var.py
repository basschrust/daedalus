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
        if hasattr(repository, "localdb"):
            db = repository.connection
            count = db.execute(
                "SELECT count(*) FROM risk_l2_instrument_statistics WHERE calibration_id = ?",
                [calibration_id],
            ).fetchone()[0]
            if count == 0:
                if source_connection is None:
                    raise ValueError(
                        "DuckDB risk calibration absent; Snowflake init connection required"
                    )
                with source_connection.cursor() as cur:
                    cur.execute(
                        "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
                        "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS WHERE CALIBRATION_ID = %s",
                        (calibration_id,),
                    )
                    instruments = cur.fetchall()
                    cur.execute(
                        "SELECT SYMBOL_1, SYMBOL_2, COVARIANCE_DAILY "
                        "FROM DAEDALUS.MART.L2_PAIR_STATISTICS WHERE CALIBRATION_ID = %s",
                        (calibration_id,),
                    )
                    pairs = cur.fetchall()
                if not instruments:
                    raise ValueError(f"Missing calibration {calibration_id}")
                db.execute("BEGIN TRANSACTION")
                try:
                    db.executemany(
                        "INSERT INTO risk_l2_instrument_statistics VALUES (?, ?, ?, ?)",
                        [(calibration_id, *row) for row in instruments],
                    )
                    if pairs:
                        db.executemany(
                            "INSERT INTO risk_l2_pair_statistics VALUES (?, ?, ?, ?)",
                            [(calibration_id, *row) for row in pairs],
                        )
                    db.execute("COMMIT")
                except Exception:
                    db.execute("ROLLBACK")
                    raise
            instrument_rows = db.execute(
                "SELECT symbol, mean_daily_return, daily_sigma FROM risk_l2_instrument_statistics "
                "WHERE calibration_id = ?",
                [calibration_id],
            ).fetchall()
            pair_rows = db.execute(
                "SELECT symbol_1, symbol_2, covariance_daily FROM risk_l2_pair_statistics "
                "WHERE calibration_id = ?",
                [calibration_id],
            ).fetchall()
        else:
            with repository.connection.cursor() as cur:
                cur.execute(
                    "SELECT SYMBOL, MEAN_DAILY_RETURN, DAILY_SIGMA "
                    "FROM DAEDALUS.MART.L2_INSTRUMENT_STATISTICS WHERE CALIBRATION_ID = %s",
                    (calibration_id,),
                )
                instrument_rows = cur.fetchall()
                cur.execute(
                    "SELECT SYMBOL_1, SYMBOL_2, COVARIANCE_DAILY "
                    "FROM DAEDALUS.MART.L2_PAIR_STATISTICS WHERE CALIBRATION_ID = %s",
                    (calibration_id,),
                )
                pair_rows = cur.fetchall()

        # Reuse the existing strict covariance validation and ordering logic.
        class Cursor:
            def __init__(self):
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def execute(self, *args):
                self.calls += 1

            def fetchall(self):
                return instrument_rows if self.calls == 1 else pair_rows

        class Connection:
            def cursor(self):
                return Cursor()

        regime = FundamentalReturnRegime.from_snowflake(
            Connection(), calibration_id, symbols=symbols
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
