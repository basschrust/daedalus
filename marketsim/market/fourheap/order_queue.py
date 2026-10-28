from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

from marketsim.market.fourheap.order import MatchedOrder

if TYPE_CHECKING:
    from marketsim.market.fourheap.order import Order
    from marketsim.market.price import Price


class OrderQueue:
    def __init__(self, is_max_heap: bool = False, is_matched: bool = False, logger=None):
        self.is_max_heap = is_max_heap
        self.is_matched = is_matched

        self.size = 0
        self.heap = []
        self.order_dict = {}
        self.deleted_ids = set()
        self.logger = logger

    def add_order(
        self,
        order: Order,
        executed_price: Price | None = None,
        executed_mode: str | None = None,
        matched_with: int | None = None,
    ) -> None:
        # self.logger.debug(
        #     f"order_queue.add_order {order} with executed_price: {executed_price}, executed_mode: {executed_mode}, matched_with: {matched_with}"
        # )
        self.logger.debug(
            "QUEUE ADD | matched_queue={} "
            "object_id={} order_id={} parent_id={} "
            "asset_id={} agent_id={} "
            "price={} qty={} valid_until={} "
            "previous_match={} new_match={} "
            "previous_exec_price={} new_exec_price={} "
            "previous_mode={} new_mode={}",
            self.is_matched,
            id(order),
            order.order_id,
            order.parent_id,
            order.asset_id,
            order.agent_id,
            order.price,
            order.quantity,
            order.valid_until,
            order.matched_with,
            matched_with,
            order.executed_price,
            executed_price,
            order.executed_mode,
            executed_mode,
        )
        price = order.price if not self.is_max_heap else -order.price
        order_id = order.order_id
        if self.contains(order_id):
            self.order_dict[order_id].merge_order(order.quantity)
            # this may be a problem if we want to merge with order already executed
            # so - in unmatched heaps - no problem
            # - in matched heaps - well, in continuous mode there should not be such a situation?
            if self.is_matched:
                raise  # to confirm that for matched it never happens, but somehow for not matched it doesn't happen either
        else:
            heapq.heappush(self.heap, (price, order.order_id))
            self.order_dict[order.order_id] = order
            if self.is_matched:
                assert order.executed_price is None, (
                    f"Order {order.order_id} already executed at {order.executed_price}"
                )
                assert order.matched_with is None, (
                    f"Order {order.order_id} already matched with {order.matched_with}"
                )
            if self.is_matched:
                order.executed_price = (
                    executed_price  # hmm, the price from the other side peek which is not
                )
                # available here... so we pass it here :)
                order.executed_mode = executed_mode
                order.matched_with = matched_with
        self.size += order.quantity

    def peek(self) -> Price | float:
        # float is only because of +/- math.inf  Price type covers all finite values
        c = -1 if self.is_max_heap else 1
        # Return infinity if empty
        if self.is_empty() or len(self.heap) == 0:
            return c * math.inf

        # Clean up deleted orders
        try:
            while len(self.heap) > 0 and self.heap[0][1] in self.deleted_ids:
                heapq.heappop(self.heap)
                if len(self.heap) == 0:
                    return c * math.inf
        except IndexError:
            return c * math.inf

        # If we get here, heap should have at least one valid element
        if len(self.heap) == 0:
            return c * math.inf

        # AK - for matched heaps this should always be empty, always return inf
        if self.is_matched:
            self.logger.info(f"Peeking matched heap: {self.heap[0]}")
            # raise
        return c * self.heap[0][0]

    def peek_order(self) -> Order | None:
        if self.is_empty() or len(self.heap) == 0:
            return None
        # Safely clean up deleted orders
        try:
            while len(self.heap) > 0 and self.heap[0][1] in self.deleted_ids:
                heapq.heappop(self.heap)
            # Check if heap is now empty
            if len(self.heap) == 0:
                return None
            order_id = self.heap[0][1]
            return self.order_dict[order_id]
        except (IndexError, KeyError):
            return None

    def peek_order_id(self) -> float | None:
        if self.is_empty() or len(self.heap) == 0:
            return None
        try:
            return self.heap[0][1]
        except IndexError:
            return None

    def clear(self) -> None:
        self.heap = []
        self.order_dict = {}
        self.deleted_ids = set()
        self.size = 0

    def market_clear(self, *, current_time: int, price: Price | None = None) -> list[MatchedOrder]:
        # AK TODO: rename to match_orders ?
        if price is not None:
            raise  # to ensure we never pass the price from external var
        if self.is_matched:
            matched_orders = []
            for _, order_id in self.heap:
                if order_id not in self.deleted_ids:
                    order = self.order_dict[order_id]
                    price = (
                        price if price is not None else order.executed_price
                    )  # somehow it didn't work properly before?
                    volume = order.quantity
                    cash = volume * price * order.order_type
                    matched_orders.append(
                        MatchedOrder(
                            price=order.executed_price,
                            cash=cash,
                            volume=volume,
                            time=current_time,
                            order=order,
                        )
                    )
            self.clear()
            return matched_orders
        return []

    def is_empty(self) -> bool:
        return self.size <= 0 or len(self.heap) == 0

    def count(self) -> int:
        return self.size

    def remove(self, order_id: int) -> None:
        self.logger.info(f"Removing order {order_id}")
        if self.contains(order_id):
            self.deleted_ids.add(order_id)
            self.size -= self.order_dict[
                order_id
            ].quantity  # ?? AK - so what does the size contain?
            # is it the size of the queue or the size of the order?
            # the size is the sum of quantities on one side of LOB?

            # Clean up top of heap if needed
            try:
                # Only try to clean up if there are items in the heap
                if len(self.heap) > 0:
                    # Use direct access to heap instead of peek_order which could cause recursion
                    while len(self.heap) > 0 and self.heap[0][1] in self.deleted_ids:
                        heapq.heappop(self.heap)
            except (KeyError, AttributeError, IndexError):
                pass

            # Remove from order dict
            del self.order_dict[order_id]

    def contains(self, order_id: int) -> bool:
        return order_id in self.order_dict

    # TODO: mark as deprecated, let's use the pop_best_order only
    def push_to(self) -> Order | None:
        return self.pop_best_order()

    def pop_best_order(self) -> Order | None:
        while self.heap:
            price, order_id = heapq.heappop(self.heap)
            if order_id not in self.deleted_ids:
                order = self.order_dict[order_id]
                self.size -= order.quantity
                del self.order_dict[order.order_id]

                # Make sure the new top of heap is not a removed order
                try:
                    while len(self.heap) > 0 and self.heap[0][1] in self.deleted_ids:
                        heapq.heappop(self.heap)
                except IndexError:
                    pass

                return order
        return None

    def __str__(self) -> str:
        s = ""
        for _, order_id in self.heap:
            if order_id not in self.deleted_ids:
                order = self.order_dict[order_id]
                s += str(order) + "\n"

        return s
