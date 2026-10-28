from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from marketsim.utils.id_generator import id_generator

if TYPE_CHECKING:
    from marketsim.fourheap import Order
    from marketsim.market import Price


def validate_price(price: Price) -> None:
    if price <= 0:
        raise ValueError("Price must be greater than 0")


@dataclass
class Order:
    price: Price
    order_type: int  # -1 for a sell order, +1 for a buy order
    quantity: int
    agent_id: int
    time: int
    order_id: int
    asset_id: int
    executed_price: Price | None = None
    executed_mode: str | None = (
        None  # arrived - executed immediately after placing/waited - placed in the LOB and later crossed with an order which arrived later
    )
    parent_id: int | None = None
    matched_with: int | None = None  # order_id of the order that matched with this one
    valid_until: int | None = (
        None  # order will be cancelled when time tick reaches this value. None means no cancelling by time
    )

    def __init__(
        self,
        price: Price,
        order_type: int,
        quantity: int,
        agent_id: int,
        time: int,
        asset_id: int,
        parent_id: int | None = None,
        matched_with: int | None = None,
        valid_until: int | None = None,
    ) -> None:
        validate_price(price)
        self.price = price
        self.order_type = order_type
        self.quantity = quantity
        self.agent_id = agent_id
        self.time = time
        self.asset_id = asset_id
        self.order_id = id_generator.next()
        self.parent_id = parent_id  # order_id of the original order when this one is created after partial execution
        self.matched_with = matched_with
        self.valid_until = valid_until

    def update_quantity_filled(self, quantity: int) -> None:
        self.quantity -= quantity

    def merge_order(self, q_additional: int) -> None:
        self.quantity += q_additional

    def copy_and_decrease(self, transact_quantity: int) -> "Order":
        new_order = Order(
            price=self.price,
            order_type=self.order_type,
            quantity=self.quantity - transact_quantity,
            agent_id=self.agent_id,
            time=self.time,
            parent_id=self.order_id,
            asset_id=self.asset_id,
            valid_until=self.valid_until,
        )
        self.update_quantity_filled(self.quantity - transact_quantity)
        # self.update_quantity_filled(transact_quantity)
        return new_order

    def __eq__(self, other: Order | None) -> bool:
        if other is None:
            return False
        return self.order_id == other.order_id

    def __gt__(self, other: "Order") -> bool:
        # TODO: we should check asset_id not to compare orders for different instruments
        if self.order_type == -1 and other.order_type == -1:
            return (self.price, self.time) < (other.price, other.time)
        elif self.order_type == 1 and other.order_type == 1:
            return (self.price, -self.time) > (other.price, -other.time)
        # I don't think these are needed
        elif self.order_type == -1:
            return self.price < other.price
        elif self.order_type == 1:
            return self.price > other.price


@dataclass
class MatchedOrder:
    price: Price
    time: int
    order: Order
    volume: int
    cash: Price
    phase: str | None = None  # continuous / fixing / otc - yet not used
