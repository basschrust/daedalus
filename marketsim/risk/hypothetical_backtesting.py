"""One-day frozen-book hypothetical P&L (HPL) for dated equity snapshots.

APL already exists in var_backtesting.py. Unlike APL, HPL holds day-t units
constant across t -> t+1 and excludes trades, fees, financing and dividends.
All prices must be same-day, base-currency closes. No derivatives or FX risk.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

import pandas as pd

from marketsim.risk.portfolio_snapshot import PortfolioSnapshot


def _price(*, value: object, asset_id: int, session: date) -> Decimal:
    try:
        price = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid close for {asset_id} on {session}") from exc
    if not price.is_finite() or price <= 0:
        raise ValueError(f"Invalid close for {asset_id} on {session}")
    return price


def one_day_hpl(
    *,
    forecasts: pd.DataFrame,
    snapshots: Sequence[PortfolioSnapshot],
    instrument_classes: Mapping[int, str],
    closes: Mapping[date, Mapping[int, Decimal]],
) -> pd.DataFrame:
    """Join 1-day VaR(t) to frozen-position P&L over consecutive EoD snapshots.

    Use a snapshot from t only, never holdings at t+1. A date missing from
    the price panel fails loudly. Last date has no realized observation.
    The historical day index must be contiguous and advance with calendar.
    The resulting columns match one_day_apl where appropriate, except the
    realized observation is named hypothetical_pnl to prevent APL confusion.
    """
    required = {"day", "agent_id", "horizon_days", "var_amount"}
    if required - set(forecasts.columns):
        raise ValueError(f"Missing forecast fields: {sorted(required - set(forecasts.columns))}")
    if not snapshots:
        return pd.DataFrame(columns=[*forecasts.columns, "hypothetical_pnl", "loss", "exception"])
    indexed: dict[tuple[int, int], PortfolioSnapshot] = {}
    dates: dict[int, date] = {}
    ordered_days = sorted({snapshot.day for snapshot in snapshots})
    for snapshot in snapshots:
        if snapshot.as_of_date is None:
            raise ValueError("Historical HPL requires as_of_date")
        key = (snapshot.day, snapshot.agent_id)
        if key in indexed:
            raise ValueError(f"Duplicate portfolio snapshot: {key}")
        indexed[key] = snapshot
        if snapshot.day in dates and dates[snapshot.day] != snapshot.as_of_date:
            raise ValueError(f"Mixed dates for historical day {snapshot.day}")
        dates[snapshot.day] = snapshot.as_of_date
        for asset_id, quantity in snapshot.positions.items():
            if quantity and str(instrument_classes.get(asset_id, "")).lower() not in {
                "stock",
                "index",
            }:
                raise ValueError(f"Unsupported HPL holding type for asset {asset_id}")
        if snapshot.futures_positions and any(
            holding.quantity for holding in snapshot.futures_positions.values()
        ):
            raise ValueError("Historical HPL for futures is not implemented")
    if ordered_days != list(range(ordered_days[0], ordered_days[-1] + 1)):
        raise ValueError("Historical days must be contiguous")
    if [dates[d] for d in ordered_days] != sorted(dates.values()):
        raise ValueError("Historical day/date mapping is not ascending")
    next_sessions = {day: dates[day + 1] for day in ordered_days[:-1]}
    rows: list[dict] = []
    seen: set[tuple] = set()
    for record in forecasts.to_dict(orient="records"):
        if int(record["horizon_days"]) != 1:
            continue
        day = int(record["day"])
        agent = int(record["agent_id"])
        if day not in next_sessions or (day, agent) not in indexed:
            continue
        key = (
            record.get("method"),
            day,
            agent,
            record.get("confidence_level"),
            int(record["horizon_days"]),
        )
        if key in seen:
            raise ValueError(f"Duplicate VaR forecast: {key}")
        seen.add(key)
        snapshot = indexed[(day, agent)]
        current_day = snapshot.as_of_date
        next_day = next_sessions[day]
        if (day + 1, agent) not in indexed:
            continue
        if current_day not in closes or next_day not in closes:
            raise ValueError(f"Missing prices for {current_day} or {next_day}")
        pnl = Decimal(0)
        for asset_id, quantity in snapshot.positions.items():
            if not quantity:
                continue
            if asset_id not in closes[current_day] or asset_id not in closes[next_day]:
                raise ValueError(
                    f"Missing historical close for {asset_id} at {current_day}/{next_day}"
                )
            p0 = _price(value=closes[current_day][asset_id], asset_id=asset_id, session=current_day)
            p1 = _price(value=closes[next_day][asset_id], asset_id=asset_id, session=next_day)
            pnl += quantity * (p1 - p0)
        try:
            var = Decimal(str(record["var_amount"]))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError("Invalid VaR forecast") from exc
        if not var.is_finite() or var < 0:
            raise ValueError("VaR forecast must be finite and nonnegative")
        loss = -pnl
        rows.append(
            {
                **record,
                "hypothetical_pnl": float(pnl),
                "loss": float(loss),
                "exception": loss > var,
            }
        )
    return pd.DataFrame(rows, columns=[*forecasts.columns, "hypothetical_pnl", "loss", "exception"])
