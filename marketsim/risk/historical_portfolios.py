"""Deterministic, no-trading historical equity portfolios for risk backtesting.

This produces PortfolioSnapshot domain values; it does not persist or simulate
trades. Prices must be point-in-time closing prices already expressed in the
portfolio's base currency. Historical futures/options are intentionally rejected
until contract metadata, currency conversion and settlements are dated correctly.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence

from marketsim.risk.portfolio_snapshot import PortfolioSnapshot


def _money(*, value: object, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def build_historical_equity_snapshots(
    *,
    agent_id: int,
    dates: Sequence[date],
    holdings: Mapping[int, int],
    instrument_classes: Mapping[int, str],
    closes: Mapping[date, Mapping[int, Decimal]],
    initial_cash: Mapping[str, Decimal],
    base_currency: str = "PLN",
) -> list[PortfolioSnapshot]:
    """Value constant share quantities at each historical EoD, days -N..-1.

    This is a fixed-holdings template, NOT a reconstruction of actual trades.
    Caller must provide each close in the base currency as of that day. No
    forwards/futures/options, future FX quotes, or implicit zero prices.

    Cash remains unchanged across dates (no dividends, fees or interest).
    Non-base-currency cash cannot be valued without dated FX and is rejected.
    """
    if not dates or len(set(dates)) != len(dates) or list(dates) != sorted(dates):
        raise ValueError("Historical dates must be nonempty, unique, and ascending")
    if any(type(day) is not date for day in dates):
        raise ValueError("Historical calendar requires datetime.date values")
    if not isinstance(base_currency, str) or not base_currency.strip():
        raise ValueError("base_currency is required")
    currency = base_currency.upper()
    if not initial_cash:
        raise ValueError("An explicit base-currency cash balance is required")
    if set(initial_cash) != {currency}:
        raise ValueError("Historical cash must be denominated only in base_currency")
    balance = _money(value=initial_cash[currency], label="cash")
    positions: dict[int, int] = {}
    for asset_id, quantity in holdings.items():
        if type(asset_id) is not int or asset_id < 0:
            raise ValueError(f"Invalid asset ID: {asset_id!r}")
        if type(quantity) is not int:
            raise ValueError(f"Noninteger holding in asset {asset_id}")
        if quantity:
            kind = str(instrument_classes.get(asset_id, "")).lower()
            if kind not in {"stock", "index"}:
                raise ValueError(
                    f"Unsupported historical holding type for asset {asset_id}: {kind!r}"
                )
            positions[asset_id] = quantity
    result: list[PortfolioSnapshot] = []
    for offset, session in enumerate(dates):
        if session not in closes:
            raise ValueError(f"No closes for historical session {session}")
        prices = closes[session]
        equity = balance
        for asset_id, quantity in positions.items():
            if asset_id not in prices:
                raise ValueError(f"Missing close for asset {asset_id} on {session}")
            price = _money(value=prices[asset_id], label=f"close {asset_id} on {session}")
            if price <= 0:
                raise ValueError(f"Non-positive close for asset {asset_id} on {session}")
            equity += quantity * price
        result.append(
            PortfolioSnapshot(
                agent_id=agent_id,
                day=offset - len(dates),
                as_of_date=session,
                base_currency=currency,
                positions=positions,
                futures_positions={},
                cash={currency: balance},
                portfolio_value=equity,
            )
        )
    return result
