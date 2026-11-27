"""Point-in-time one-day VaR backtesting for fixed historical equity books.

Uses the production EodVaR model (parametric, historical and Monte Carlo).
Only information through the forecast session is used for model calibration,
portfolio valuation and volatility. Actual next-day closes are used solely
for hypothetical P&L. No database access, no orders, no mutable live agents.

The caller supplies complete, *base-currency* closing prices. Futures,
options, foreign-currency exposure and dividends are not yet modeled here.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from types import SimpleNamespace
from typing import Mapping, Sequence

import pandas as pd

from marketsim.risk.eod_var import EodVaR
from marketsim.risk.hypothetical_backtesting import one_day_hpl
from marketsim.risk.portfolio_snapshot import PortfolioSnapshot
from marketsim.risk.var_backtesting import kupiec_report


def _valid_price(*, value: object, asset_id: int, session: date) -> float:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid close for asset {asset_id} on {session}") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(f"Invalid close for asset {asset_id} on {session}")
    return float(price)


def _history_at_date(
    *,
    dates: Sequence[date],
    cutoff_index: int,
    asset_ids: Sequence[int],
    symbols: Mapping[int, str],
    closes: Mapping[date, Mapping[int, Decimal]],
    lookback_sessions: int,
) -> SimpleNamespace:
    """Build a point-in-time EodVaR calibration with a content-derived ID."""
    window = dates[max(0, cutoff_index + 1 - lookback_sessions) : cutoff_index + 1]
    series: dict[str, dict[date, float]] = {symbols[i]: {} for i in asset_ids}
    digest = sha256()
    for session in window:
        if session not in closes:
            raise ValueError(f"Missing historical close session {session}")
        digest.update(session.isoformat().encode("ascii"))
        for asset_id in asset_ids:
            if asset_id not in closes[session]:
                raise ValueError(f"Missing close for asset {asset_id} on {session}")
            value = _valid_price(
                value=closes[session][asset_id], asset_id=asset_id, session=session
            )
            symbol = symbols[asset_id]
            series[symbol][session] = value
            digest.update(f"|{symbol}={value:.17g}".encode("ascii"))
        digest.update(b"\n")
    return SimpleNamespace(
        series=series,
        lookback_sessions=lookback_sessions,
        latest_calibration_id=f"HIST-{window[-1]:%Y%m%d}-{digest.hexdigest()[:24]}",
    )


def walk_forward_equity_var(
    *,
    snapshots: Sequence[PortfolioSnapshot],
    closes: Mapping[date, Mapping[int, Decimal]],
    symbols: Mapping[int, str],
    instrument_classes: Mapping[int, str],
    risk_config: Mapping[str, object] | None = None,
    lookback_sessions: int = 250,
    group_names: Mapping[int, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute point-in-time VaR, HPL exceptions and Kupiec reports.

    Return (forecast_rows, HPL_observations, Kupiec_report). Forecasts include
    only one-day horizons; observations exclude the final historical session.

    Historical prices must cover *every* session of `snapshots`, including a
    warm-up window. No VaR forecast is produced before enough historical
    scenarios exist; insufficient history is never backfilled with future data.
    All snapshots for an agent must use matching day/date mapping.
    """
    if isinstance(lookback_sessions, bool) or not isinstance(lookback_sessions, int):
        raise ValueError("lookback_sessions must be an integer")
    if lookback_sessions < 3:
        raise ValueError("lookback_sessions must be >= 3")
    if not snapshots:
        raise ValueError("Historical portfolio snapshots are required")
    config = dict(risk_config or {})
    # This backtest evaluates one-day risk. No silent sqrt-time extrapolation.
    config["horizons_days"] = [1]
    minimum = int(config.get("historical_min_scenarios", 30))
    if minimum < 2:
        raise ValueError("historical_min_scenarios must be >= 2")
    if lookback_sessions <= minimum:
        raise ValueError("lookback_sessions must exceed historical_min_scenarios")

    asset_ids = sorted(symbols)
    if not asset_ids:
        raise ValueError("At least one priced equity risk factor is required")
    names = {asset_id: str(symbols[asset_id]).strip().upper() for asset_id in asset_ids}
    if any(not name for name in names.values()) or len(set(names.values())) != len(names):
        raise ValueError("Equity risk factor symbols must be nonempty and unique")
    if any(instrument_classes.get(i, "").lower() not in {"stock", "index"} for i in asset_ids):
        raise ValueError("Only stocks and indices are supported in historical VaR")

    by_day: dict[int, list[PortfolioSnapshot]] = {}
    day_dates: dict[int, date] = {}
    seen: set[tuple[int, int]] = set()
    for snapshot in snapshots:
        if snapshot.as_of_date is None or type(snapshot.as_of_date) is not date:
            raise ValueError("All historical snapshots require an as_of_date")
        if snapshot.day >= 0:
            raise ValueError("Historical VaR requires negative day indices")
        if any(holding.quantity for holding in snapshot.futures_positions.values()):
            raise ValueError("Historical futures VaR backtesting is not implemented")
        if set(snapshot.cash) != {snapshot.base_currency}:
            raise ValueError("Historical VaR requires base-currency-only cash")
        if any(i not in names for i, qty in snapshot.positions.items() if qty):
            raise ValueError("Historical holdings contain an unpriced asset")
        key = (snapshot.day, snapshot.agent_id)
        if key in seen:
            raise ValueError(f"Duplicate historical snapshot {key}")
        seen.add(key)
        if snapshot.day in day_dates and day_dates[snapshot.day] != snapshot.as_of_date:
            raise ValueError("Snapshots disagree on their day/date mapping")
        day_dates[snapshot.day] = snapshot.as_of_date
        by_day.setdefault(snapshot.day, []).append(snapshot)

    days = sorted(by_day)
    if days != list(range(days[0], days[-1] + 1)):
        raise ValueError("Historical snapshot days must be contiguous")
    if days[-1] != -1:
        raise ValueError("Historical snapshots must terminate at day -1")
    dates = [day_dates[day] for day in days]
    if dates != sorted(set(dates)):
        raise ValueError("Historical dates must be strictly ascending")
    agent_sets = [{snapshot.agent_id for snapshot in by_day[day]} for day in days]
    if any(agent_set != agent_sets[0] for agent_set in agent_sets[1:]):
        raise ValueError("Every historical agent must have a snapshot on every session")
    currencies = {snapshot.base_currency for snapshot in snapshots}
    if len(currencies) != 1:
        raise ValueError("Historical close panel must use one common base currency")
    base_currency = next(iter(currencies))
    if len(dates) < minimum + 2:
        raise ValueError(
            f"Need at least {minimum + 2} historical sessions for a VaR forecast "
            "and one following HPL observation"
        )

    # Validate all provided panel rows before evaluating any forecast, then
    # feed only the prefix through t into the model, never t+1.
    for session in dates:
        if session not in closes:
            raise ValueError(f"Missing historical close session {session}")
        for asset_id in asset_ids:
            if asset_id not in closes[session]:
                raise ValueError(f"Missing close for asset {asset_id} on {session}")
            _valid_price(value=closes[session][asset_id], asset_id=asset_id, session=session)

    # Verify that each supplied snapshot actually matches its same-day close
    # panel. Otherwise its reported equity may come from an incompatible mark.
    for snapshot in snapshots:
        expected = snapshot.cash[snapshot.base_currency]
        for asset_id, quantity in snapshot.positions.items():
            if quantity:
                expected += quantity * Decimal(str(closes[snapshot.as_of_date][asset_id]))
        if abs(expected - snapshot.portfolio_value) > Decimal("0.000001"):
            raise ValueError(
                f"Portfolio value disagrees with closes for agent {snapshot.agent_id} "
                f"on {snapshot.as_of_date}"
            )

    forecasts: list[dict] = []
    # The model needs at least `minimum` full return scenarios; N prices yield
    # N-1 returns and, for horizon 1, N-1 historical scenarios.
    for offset in range(minimum, len(days) - 1):
        session = dates[offset]
        calibration = _history_at_date(
            dates=dates,
            cutoff_index=offset,
            asset_ids=asset_ids,
            symbols=names,
            closes=closes,
            lookback_sessions=lookback_sessions,
        )
        engine = EodVaR(calibration, config)
        markets = {
            asset_id: SimpleNamespace(
                instrument_class=instrument_classes[asset_id].lower(),
                short_name=names[asset_id],
                symbol=names[asset_id],
                currency=base_currency,  # quotes are already in the portfolio base currency
                last_traded_price=closes[session][asset_id],
            )
            for asset_id in asset_ids
        }
        for snapshot in by_day[days[offset]]:
            # Model adapter only; source snapshots remain immutable and are never
            # replaced with a mutable simulated Agent.
            adapter = SimpleNamespace(
                agent_id=snapshot.agent_id,
                base_currency=snapshot.base_currency,
                portfolio_value=snapshot.portfolio_value,
                position=dict(snapshot.positions),
                futures_position={},
                markets=markets,
                fx_rate_provider=_IdentityFx(),
            )
            model_rows = engine.rows_for_agent(adapter, snapshot.day)
            for method, records in model_rows.items():
                for record in records:
                    forecasts.append(
                        {
                            **record,
                            "method": method,
                            "group_name": (group_names or {}).get(snapshot.agent_id, "Historical"),
                            "as_of_date": session,
                        }
                    )

    frame = pd.DataFrame(forecasts)
    if frame.empty:
        raise ValueError("No historical VaR forecasts were produced")
    observations = one_day_hpl(
        forecasts=frame,
        snapshots=snapshots,
        instrument_classes=instrument_classes,
        closes=closes,
    )
    report = kupiec_report(observations)
    return frame, observations, report


class _IdentityFx:
    """Conversion for already base-currency prices, never a lookup of future FX."""

    @staticmethod
    def convert(amount: float, *, day: int, base: str, quote: str) -> float:
        if base.upper() != quote.upper():
            raise ValueError("Historical equity VaR does not perform FX conversions")
        return amount
