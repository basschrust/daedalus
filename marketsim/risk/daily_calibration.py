"""Daily rolling risk calibration from Ikaros history and simulated EoD closes.

The Ikaros OHLCV snapshot is read once at simulation initialization.
After each simulated EoD, the just-closed prices extend the same return panel.
Each calibration is immutable and stored under a content-derived ID in the
simulation repository, not in the source DAEDALUS.MART schema.
"""

from __future__ import annotations

import math
from datetime import date
from hashlib import sha256

import numpy as np
import pandas as pd

from marketsim.simulator.simulator import Simulator


def _is_snowflake(connection: object) -> bool:
    module = type(connection).__module__.lower()
    if "snowflake" in module:
        return True
    if "duckdb" in module:
        return False
    raise TypeError(f"Unsupported risk-calibration database: {type(connection)!r}")


def _symbol(market: object) -> str:
    return str(getattr(market, "symbol", None) or getattr(market, "short_name", "")).strip().upper()


def _positive_price(symbol: str, value: object) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{symbol}: risk-calibration close must be finite and positive")
    return number


class DailyRiskCalibration:
    """Maintain a rolling joint close-price panel and persist EoD risk statistics."""

    def __init__(
        self,
        simulator: Simulator,
        *,
        lookback_sessions: int = 250,
    ) -> None:
        if not isinstance(lookback_sessions, int) or lookback_sessions < 3:
            raise ValueError("risk_var.lookback_sessions must be >= 3")
        if not simulator.simulation_dates:
            raise ValueError("Daily risk calibration requires simulation_dates")

        self.connection = simulator.repository.connection
        self.snowflake = _is_snowflake(self.connection)
        self.placeholder = "%s" if self.snowflake else "?"
        self.lookback_sessions = lookback_sessions
        self.next_day = 0

        # Stocks + indices are risk factors. Futures and options are valued
        # against these factors; derivative prices must not inflate the universe.
        self.markets: dict[str, object] = {}
        for market in simulator.markets.values():
            if market.instrument_class not in {"stock", "index"}:
                continue
            symbol = _symbol(market)
            if not symbol or symbol in self.markets:
                raise ValueError(f"Duplicate or missing risk factor: {symbol!r}")
            self.markets[symbol] = market
        if not self.markets:
            raise ValueError("No stock/index risk factors available for calibration")

        self.series: dict[str, dict[date, float]] = {}
        history = {
            str(symbol).upper(): frame
            for symbol, frame in simulator.historical_ohlcv_by_symbol.items()
        }
        for symbol in sorted(self.markets):
            if symbol not in history:
                raise ValueError(f"Missing Ikaros historical OHLCV for {symbol}")
            frame = history[symbol]
            if "close" not in frame:
                raise ValueError(f"{symbol}: historical OHLCV lacks close")
            if frame.index.has_duplicates:
                raise ValueError(f"{symbol}: duplicate historical sessions")
            values: dict[date, float] = {}
            for timestamp, close in frame["close"].items():
                session = pd.Timestamp(timestamp).date()
                if session >= simulator.simulation_dates[0]:
                    raise ValueError(f"{symbol}: historical data overlaps simulation at {session}")
                if session in values:
                    raise ValueError(f"{symbol}: duplicate historical session {session}")
                values[session] = _positive_price(symbol, close)
            if not values:
                raise ValueError(f"{symbol}: empty Ikaros history")
            self.series[symbol] = values

        # A synchronized last historical session is essential; otherwise a
        # simulated return would silently span different dates across assets.
        historical_ends = {max(values) for values in self.series.values()}
        if len(historical_ends) != 1:
            raise ValueError(f"Unaligned Ikaros risk-factor cutoffs: {sorted(historical_ends)}")

        self._ensure_tables()
        self.latest_calibration_id = self._calibrate(day=-1, source="IKAROS")

    def _execute(self, cursor: object, sql: str, params: tuple[object, ...] = ()) -> None:
        cursor.execute(sql, params)

    def _ensure_tables(self) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS risk_l2_instrument_statistics ("
                "calibration_id VARCHAR, symbol VARCHAR, "
                "mean_daily_return DOUBLE, daily_sigma DOUBLE)"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS risk_l2_pair_statistics ("
                "calibration_id VARCHAR, symbol_1 VARCHAR, "
                "symbol_2 VARCHAR, covariance_daily DOUBLE)"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS risk_calibration_history ("
                "day INTEGER, calibration_id VARCHAR, as_of_date DATE, "
                "first_session DATE, observations INTEGER, "
                "lookback_sessions INTEGER, source VARCHAR)"
            )
        finally:
            cursor.close()

    def _calibrate(self, *, day: int, source: str) -> str:
        symbols = sorted(self.series)
        common_sessions = set.intersection(*(set(self.series[symbol]) for symbol in symbols))
        sessions = sorted(common_sessions)[-self.lookback_sessions :]
        # Never mislabel a multi-session jump as a one-day return. Shorter
        # histories are okay, but sessions within the common window must match.
        for symbol in symbols:
            aligned = sorted(
                session
                for session in self.series[symbol]
                if sessions and sessions[0] <= session <= sessions[-1]
            )
            if aligned != sessions:
                raise ValueError(f"Risk-factor trading calendars differ for {symbol}")
        if len(sessions) < 3:
            raise ValueError(f"Too few aligned calibration sessions: {len(sessions)} (need >= 3)")

        # If one risk factor is missing the current EoD, using the intersection
        # would hide that problem. Require a complete synchronized latest close.
        latest = {max(self.series[symbol]) for symbol in symbols}
        if len(latest) != 1 or sessions[-1] != next(iter(latest)):
            raise ValueError("Risk-factor EoD closes are not aligned")

        prices = np.array(
            [[self.series[symbol][session] for symbol in symbols] for session in sessions],
            dtype=np.float64,
        )
        returns = np.diff(np.log(prices), axis=0)
        if not np.isfinite(returns).all():
            raise ValueError("Nonfinite returns during risk calibration")

        mean = returns.mean(axis=0)
        covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
        covariance = (covariance + covariance.T) / 2
        if not np.isfinite(covariance).all():
            raise ValueError("Nonfinite covariance during risk calibration")

        # Identical observations and dates give the same calibration ID.
        digest = sha256()
        for session, prices_on_day in zip(sessions, prices):
            digest.update(session.isoformat().encode("ascii"))
            for symbol, price in zip(symbols, prices_on_day):
                digest.update(f"|{symbol}:{price:.17g}".encode("ascii"))
            digest.update(b"\n")
        calibration_id = f"SIM-{sessions[-1]:%Y%m%d}-{digest.hexdigest()[:24]}"

        instrument_rows = [
            (
                calibration_id,
                symbol,
                float(mean[i]),
                float(math.sqrt(max(0.0, covariance[i, i]))),
            )
            for i, symbol in enumerate(symbols)
        ]
        pair_rows = [
            (calibration_id, symbols[i], symbols[j], float(covariance[i, j]))
            for i in range(len(symbols))
            for j in range(i + 1, len(symbols))
        ]

        cursor = self.connection.cursor()
        try:
            self._execute(
                cursor,
                f"SELECT day FROM risk_calibration_history "
                f"WHERE calibration_id = {self.placeholder}",
                (calibration_id,),
            )
            existing = cursor.fetchall()
            if existing:
                if len(existing) != 1:
                    raise ValueError(f"Duplicate calibration metadata: {calibration_id}")
                # Different simulation days can legitimately have identical
                # prices and the same rolling window; reuse the same statistics.
                return calibration_id

            cursor.execute("BEGIN TRANSACTION")
            try:
                cursor.executemany(
                    "INSERT INTO risk_l2_instrument_statistics "
                    f"(calibration_id, symbol, mean_daily_return, daily_sigma) "
                    f"VALUES ({', '.join([self.placeholder] * 4)})",
                    instrument_rows,
                )
                if pair_rows:
                    cursor.executemany(
                        "INSERT INTO risk_l2_pair_statistics "
                        f"(calibration_id, symbol_1, symbol_2, covariance_daily) "
                        f"VALUES ({', '.join([self.placeholder] * 4)})",
                        pair_rows,
                    )
                self._execute(
                    cursor,
                    "INSERT INTO risk_calibration_history "
                    "(day, calibration_id, as_of_date, first_session, "
                    "observations, lookback_sessions, source) "
                    f"VALUES ({', '.join([self.placeholder] * 7)})",
                    (
                        day,
                        calibration_id,
                        sessions[-1],
                        sessions[0],
                        len(sessions) - 1,
                        self.lookback_sessions,
                        source,
                    ),
                )
                cursor.execute("COMMIT")
            except Exception:
                cursor.execute("ROLLBACK")
                raise
        finally:
            cursor.close()
        return calibration_id

    def append_eod(self, simulator: Simulator, day: int) -> str:
        """Append exactly one simulated EoD and publish its updated calibration."""
        if day != self.next_day:
            raise ValueError(f"Expected EoD day {self.next_day}, got {day}")
        session = simulator.simulation_dates[day]
        # Validate the complete daily cross-section before mutating any series.
        closes: dict[str, float] = {}
        for symbol, market in self.markets.items():
            if session in self.series[symbol]:
                raise ValueError(f"{symbol}: duplicate EoD session {session}")
            closes[symbol] = _positive_price(symbol, market.last_traded_price)
        for symbol, value in closes.items():
            self.series[symbol][session] = value

        # Avoid leaving a partially advanced in-memory history after a SQL
        # or calibration error.
        try:
            calibration_id = self._calibrate(day=day, source="SIMULATED")
        except Exception:
            for symbol in self.markets:
                del self.series[symbol][session]
            raise
        self.latest_calibration_id = calibration_id
        self.next_day += 1
        return calibration_id


class DailyCalibratedSimulator(Simulator):
    """Run risk calibration and all three EoD VaRs with immutable daily IDs."""

    def __init__(self, *, risk_var: dict | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.risk_var = risk_var or {}
        self.daily_risk_calibration: DailyRiskCalibration | None = None
        self.current_risk_calibration_id: str | None = None
        if self.risk_var.get("enabled", False):
            lookback = int(self.risk_var.get("lookback_sessions", 250))
            self.daily_risk_calibration = DailyRiskCalibration(self, lookback_sessions=lookback)
            self.current_risk_calibration_id = self.daily_risk_calibration.latest_calibration_id

    def eod(self, day: int) -> None:
        # Market and agent EoD must finish first: the new calibration contains
        # today's closing prices, and risk uses the already-settled positions.
        super().eod(day)
        if self.daily_risk_calibration is None:
            return
        from marketsim.risk.eod_var import write_eod_var

        self.current_risk_calibration_id = self.daily_risk_calibration.append_eod(self, day)
        write_eod_var(self, day, self.daily_risk_calibration, self.risk_var)
        self.logger.info(f"Risk model and all VaRs EoD {day}: {self.current_risk_calibration_id}")
