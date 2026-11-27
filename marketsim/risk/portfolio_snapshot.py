"""Immutable point-in-time portfolio holdings for risk and backtesting.

This is a domain object, not another database table. Existing eod_agent_*
relations remain the persistence representation. Prices and risk calibrations
are separate point-in-time inputs, never read from mutable market objects here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from marketsim.agent.agent import Agent


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a decimal number") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _asset_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid asset_id: {value!r}")
    return value


def _quantity(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Quantity must be an integer: {value!r}")
    return value


@dataclass(frozen=True, kw_only=True)
class FuturesHolding:
    """One futures holding with settlement information (no market objects)."""

    quantity: int
    avg_initial_price: Decimal
    settlement_price: Decimal
    collateral: Decimal
    unsettled_pnl: Decimal

    def __post_init__(self) -> None:
        _quantity(self.quantity)
        for field in ("avg_initial_price", "settlement_price", "collateral", "unsettled_pnl"):
            object.__setattr__(self, field, _decimal(getattr(self, field), field=field))
        if self.collateral < 0:
            raise ValueError("Futures collateral cannot be negative")


@dataclass(frozen=True, kw_only=True)
class PortfolioSnapshot:
    """A closed portfolio for one agent on one (possibly historical) day.

    The mappings are defensively copied and read-only; holdings and cash
    cannot change when the originating live Agent changes later.
    `as_of_date` is optional to support legacy runs without a calendar.
    Historical producers must pass it explicitly.
    """

    agent_id: int
    day: int
    base_currency: str
    positions: Mapping[int, int]
    futures_positions: Mapping[int, FuturesHolding]
    cash: Mapping[str, Decimal]
    portfolio_value: Decimal
    as_of_date: date | None = None

    def __post_init__(self) -> None:
        if isinstance(self.agent_id, bool) or not isinstance(self.agent_id, int):
            raise ValueError("agent_id must be an integer")
        if isinstance(self.day, bool) or not isinstance(self.day, int):
            raise ValueError("day must be an integer")
        if self.as_of_date is not None and (
            not isinstance(self.as_of_date, date) or isinstance(self.as_of_date, datetime)
        ):
            raise ValueError("as_of_date must be a date, not a datetime")
        if not isinstance(self.base_currency, str) or not self.base_currency.strip():
            raise ValueError("base_currency must be nonempty")
        object.__setattr__(self, "base_currency", self.base_currency.upper())

        spots: dict[int, int] = {}
        for asset_id, quantity in self.positions.items():
            spots[_asset_id(asset_id)] = _quantity(quantity)
        object.__setattr__(self, "positions", MappingProxyType(spots))

        futures: dict[int, FuturesHolding] = {}
        for asset_id, holding in self.futures_positions.items():
            asset_id = _asset_id(asset_id)
            if not isinstance(holding, FuturesHolding):
                raise TypeError("futures_positions values must be FuturesHolding")
            futures[asset_id] = holding
        object.__setattr__(self, "futures_positions", MappingProxyType(futures))

        balances: dict[str, Decimal] = {}
        for currency, balance in self.cash.items():
            if not isinstance(currency, str) or not currency.strip():
                raise ValueError("Invalid cash currency")
            code = currency.upper()
            if code in balances:
                raise ValueError(f"Duplicate cash currency: {code}")
            balances[code] = _decimal(balance, field=f"cash[{code}]")
        object.__setattr__(self, "cash", MappingProxyType(balances))
        object.__setattr__(
            self,
            "portfolio_value",
            _decimal(self.portfolio_value, field="portfolio_value"),
        )

    @classmethod
    def from_agent(
        cls, *, agent: Agent, day: int, as_of_date: date | None = None
    ) -> PortfolioSnapshot:
        """Capture already-settled live state; never call eod or mutate Agent."""
        futures = {}
        for asset_id, quantity in agent.futures_position.items():
            values = agent.futures_value.get(asset_id)
            if values is None:
                if quantity:
                    raise ValueError(f"Missing futures accounting for asset {asset_id}")
                values = {}
            futures[asset_id] = FuturesHolding(
                quantity=quantity,
                avg_initial_price=values.get("avg_initial_price", 0),
                settlement_price=values.get("settlement_price", 0),
                collateral=values.get("collateral", 0),
                unsettled_pnl=values.get("unsettled_pnl", 0),
            )
        return cls(
            agent_id=agent.agent_id,
            day=day,
            as_of_date=as_of_date,
            base_currency=agent.base_currency,
            positions=agent.position,
            futures_positions=futures,
            cash=agent.cash,
            portfolio_value=agent.portfolio_value,
        )
