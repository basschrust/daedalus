from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from marketsim.fourheap import constants
from marketsim.fourheap.order_queue import OrderQueue
from marketsim.market.price import Price

if TYPE_CHECKING:
    from marketsim.fourheap import MatchedOrder, Order
    from marketsim.market import Security


class FourHeap:
    """
    This class reimplements the four-heap data structure described in "Flexible double auctions for electronic commerce:
    theory and implementation" (Wurman, 98)
    """

    def __init__(self, plus_one=False, market: Security | None = None):
        self.plus_one = (
            plus_one  # AK - wtf is that? if True gets ask price in clearing, bid otherwise
        )

        self.market = market
        self.logger = logger.bind(market_id=market.asset_id)

        self.buy_matched = OrderQueue(is_max_heap=False, is_matched=True, logger=self.logger)
        self.buy_unmatched = OrderQueue(is_max_heap=True, is_matched=False, logger=self.logger)
        self.sell_matched = OrderQueue(is_max_heap=True, is_matched=True, logger=self.logger)
        self.sell_unmatched = OrderQueue(is_max_heap=False, is_matched=False, logger=self.logger)

        self.heaps = [self.buy_matched, self.buy_unmatched, self.sell_matched, self.sell_unmatched]
        self.agent_id_map = defaultdict(list)

        self.midprices = defaultdict(Price)  # [] # AK: well, it should be tied to the time slots

    def handle_new_order(self, order: Order) -> None:
        self.logger.debug("handle_new_order order_id={}", order.order_id)

        orders_matched = (
            self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        )
        counter_matched = (
            self.sell_matched if order.order_type == constants.BUY else self.buy_matched
        )
        counter_unmatched = (
            self.buy_unmatched if order.order_type == constants.SELL else self.sell_unmatched
        )
        orders_unmatched = (
            self.sell_unmatched if order.order_type == constants.SELL else self.buy_unmatched
        )

        while order.quantity > 0:
            best_price = counter_unmatched.peek()

            # Stop if the remaining order no longer crosses the book.
            if (order.order_type == constants.SELL and order.price > best_price) or (
                order.order_type == constants.BUY and order.price < best_price
            ):
                orders_unmatched.add_order(order)
                break

            to_match = counter_unmatched.pop_best_order()

            if to_match is None:
                orders_unmatched.add_order(order)
                break

            executed_price = to_match.price
            executed_quantity = min(order.quantity, to_match.quantity)

            # Preserve your existing copy_and_decrease semantics:
            # the current object becomes the executed quantity;
            # the returned object contains the remainder.
            if order.quantity > executed_quantity:
                remaining_order = order.copy_and_decrease(executed_quantity)
            else:
                remaining_order = None

            if to_match.quantity > executed_quantity:
                excess_order = to_match.copy_and_decrease(executed_quantity)
                counter_unmatched.add_order(excess_order)

            orders_matched.add_order(
                order,
                executed_price=executed_price,
                executed_mode="arrived",
                matched_with=to_match.order_id,
            )
            counter_matched.add_order(
                to_match,
                executed_price=executed_price,
                executed_mode="waited",
                matched_with=order.order_id,
            )

            if remaining_order is None:
                break

            order = remaining_order

    def handle_replace(self, order: Order) -> None:
        raise  # is it ever used in coninuous mode? yes, but no after the fix on L55 above on 29.7.2026
        # now developing this for the opening/closing phase
        self.logger.info(f"handle_replace {order}")
        matched = self.sell_matched if order.order_type == constants.SELL else self.buy_matched
        unmatched = (
            self.sell_unmatched if order.order_type == constants.SELL else self.buy_unmatched
        )
        q_order = order.quantity
        replaced = matched.pop_best_order()
        if replaced is not None:
            replaced_quantity = replaced.quantity
            if replaced_quantity == q_order:
                matched.add_order(order, executed_mode="replaced")
                unmatched.add_order(replaced)
            elif replaced_quantity > q_order:
                matched.add_order(order, executed_mode="replaced2")
                matched_s = replaced.copy_and_decrease(q_order)
                matched.add_order(matched_s, executed_mode="replaced3")
                unmatched.add_order(replaced)
            elif replaced_quantity < q_order:
                new_order = order.copy_and_decrease(replaced_quantity)
                matched.add_order(order, executed_mode="replaced4")
                unmatched.add_order(replaced)
                self.insert(new_order)

    def insert(self, order: Order, trading_phase: str = "continuous") -> None:
        # very important method in continuous market - determines the price in CDA (Cont.Double Auction)
        self.logger.debug(f"fourheap.insert {order}, trading_phase: {trading_phase}")
        if trading_phase == "continuous":
            self.agent_id_map[order.agent_id].append(order.order_id)
            if order.order_type == constants.SELL:
                # Cache peek values to avoid redundant heap cleanup operations
                buy_unmatched_peek = self.buy_unmatched.peek()
                # sell_matched_peek = self.sell_matched.peek() # does not matter here
                if order.price <= buy_unmatched_peek:
                    self.handle_new_order(order)
                else:
                    # no matching, at best the spread will get shrunk
                    self.sell_unmatched.add_order(order)
            elif order.order_type == constants.BUY:
                # Cache peek values to avoid redundant heap cleanup operations
                sell_unmatched_peek = self.sell_unmatched.peek()
                # buy_matched_peek = self.buy_matched.peek() # does not matter here
                if order.price >= sell_unmatched_peek:
                    # AK: crosses with existing orders, transaction will be made
                    self.handle_new_order(order)
                else:
                    # no transaction, but the spread becomes smaller
                    self.buy_unmatched.add_order(order)
        else:
            # opening, closing phases and other fixing phases
            self.agent_id_map[order.agent_id].append(order.order_id)
            if order.order_type == constants.SELL:
                # Cache peek values to avoid redundant heap cleanup operations
                buy_unmatched_peek = self.buy_unmatched.peek()
                sell_matched_peek = self.sell_matched.peek()
                if order.price <= buy_unmatched_peek and sell_matched_peek <= buy_unmatched_peek:
                    self.handle_new_order(order)
                elif order.price <= sell_matched_peek:
                    self.handle_replace(order)
                else:
                    self.sell_unmatched.add_order(order)
            elif order.order_type == constants.BUY:
                # Cache peek values to avoid redundant heap cleanup operations
                sell_unmatched_peek = self.sell_unmatched.peek()
                buy_matched_peek = self.buy_matched.peek()
                if order.price >= sell_unmatched_peek and buy_matched_peek >= sell_unmatched_peek:
                    # AK: crosses with existing orders, transaction will be made
                    self.handle_new_order(order)
                elif order.price >= buy_matched_peek:
                    # no transaction, but the spread becomes smaller
                    print(f"buy_matched_peek before handle_replace: {buy_matched_peek}")
                    self.handle_replace(order)
                else:
                    self.buy_unmatched.add_order(order)

    def remove(self, order_id: int) -> None:
        if self.buy_unmatched.contains(order_id):
            self.buy_unmatched.remove(order_id)
        elif self.sell_unmatched.contains(order_id):
            self.sell_unmatched.remove(order_id)
        elif self.buy_matched.contains(order_id):
            raise  # this should not happen - order already executed (in continuous, but in closing/fixing it may)
            order_q = self.buy_matched.order_dict[order_id].quantity
            self.buy_matched.remove(order_id)
            s = self.sell_matched.pop_best_order()
            s_quantity = s.quantity
            if s_quantity == order_q:
                self.insert(s)
            elif s_quantity > order_q:
                diff = s_quantity - order_q
                s.quantity -= diff
                self.insert(s)
                self.sell_matched.order_dict[s.order_id].quantity += diff
            elif s_quantity < order_q:
                while order_q > 0:
                    order_q -= s_quantity
                    self.insert(s)
                    s = self.sell_matched.pop_best_order()
                    s_quantity = s.quantity
        elif self.sell_matched.contains(order_id):
            raise  # this should not happen in continuous, but may happen in opening/closing/fixing
            order_q = self.sell_matched.order_dict[order_id].quantity
            self.sell_matched.remove(order_id)
            b = self.buy_matched.pop_best_order()
            b_quantity = b.quantity
            if b_quantity == order_q:
                self.insert(b)
            elif b_quantity > order_q:
                diff = b_quantity - order_q
                b.quantity -= diff
                self.insert(b)
                self.buy_matched.order_dict[b.order_id].quantity += diff
            elif b_quantity < order_q:
                while order_q > 0:
                    order_q -= b_quantity
                    self.insert(b)
                    b = self.buy_matched.pop_best_order()
                    b_quantity = b.quantity

    def withdraw_all(self, agent_id: int) -> None:
        # Check if agent has any orders before trying to remove them
        if agent_id in self.agent_id_map and self.agent_id_map[agent_id]:
            for order_id in self.agent_id_map[agent_id]:
                self.remove(order_id)
            self.agent_id_map[agent_id] = []

    def market_clear(
        self, current_time: int, trading_phase: str = "continuous"
    ) -> list[MatchedOrder]:
        # AK TODO: rename to "match_orders"?
        if trading_phase == "continuous":
            # in this mode the orders arrive one by one and are cleared. So when handling this queue of matched orders
            # they are treated as placed in sequential time points (later we will work out with the fractal time structure)
            # let's check if it is gonna even work properly - will the set of matched orders be exactly the same as in the
            # opening (standard) mode?
            # let's assume that this method is called after each new order placed
            # price = self.market.last_traded_price
            # price = self.market.traded_prices[current_time]["Close"]

            # well, those will create matched orders with not always proper price?
            # buy_matched = self.buy_matched.market_clear(price=price, current_time=current_time)
            # sell_matched = self.sell_matched.market_clear(price=price, current_time=current_time)
            buy_matched = self.buy_matched.market_clear(current_time=current_time)
            sell_matched = self.sell_matched.market_clear(current_time=current_time)

            matched_orders = buy_matched + sell_matched
            return matched_orders

        elif trading_phase == "fixed":
            raise
            price = (
                self.get_ask_quote() if self.plus_one else self.get_bid_quote()
            )  # AK - ohoh why not midprice?

            buy_matched = self.buy_matched.market_clear(price=price, current_time=current_time)
            sell_matched = self.sell_matched.market_clear(price=price, current_time=current_time)

            matched_orders = buy_matched + sell_matched
            return matched_orders
        else:
            raise ValueError(f"Invalid phase: {trading_phase}")

    def get_bid_quote(self) -> Price:
        return max(self.buy_unmatched.peek(), self.sell_matched.peek())

    def get_ask_quote(self) -> Price:
        # should be min, but maybe prices are reversed to negative here?
        return min(self.sell_unmatched.peek(), self.buy_matched.peek())

    def get_best_bid(self) -> float:
        return self.buy_unmatched.peek()

    def get_best_ask(self) -> float:
        return self.sell_unmatched.peek()

    # methods useful for WashTraders - get the liquidity info:
    def get_ask_at_volume(self, volume: int) -> Price | float:
        if self.sell_unmatched.size < volume:
            return math.inf

        cum_volume = 0

        for price, order_id in sorted(self.sell_unmatched.heap):
            try:
                order = self.sell_unmatched.order_dict[order_id]

                if order_id in self.sell_unmatched.deleted_ids:
                    continue

                cum_volume += order.quantity
            except KeyError:
                self.logger.warning(
                    f"No order with id {order_id} in sell_unmatched - get_ask_at_volume"
                )

            if cum_volume >= volume:
                self.logger.info(
                    f"Price: {price}, Cumulated volume for ask: {cum_volume}, required  {volume}"
                )
                return price

        raise ValueError(f"Invalid volume for ask: {volume}")

    def get_bid_at_volume(self, volume: int) -> Price | float:
        # but if inf /- inf then not Price, but float
        if self.buy_unmatched.size < volume:
            return -math.inf

        cum_volume = 0
        for price, order_id in sorted(self.buy_unmatched.heap):
            try:
                order = self.buy_unmatched.order_dict[order_id]

                if order_id in self.buy_unmatched.deleted_ids:
                    continue

                cum_volume += order.quantity
            except KeyError:
                self.logger.warning(
                    f"No order with id {order_id} in buy_unmatched - get_bid_at_volume"
                )

            if cum_volume >= volume:
                self.logger.info(
                    f"Price: {price}, Cumulated volume for bid: {cum_volume}, required {volume}"
                )
                return -price

        raise ValueError(f"Invalid volume for bid: {volume}")

    def update_midprice(self, current_time: int, lookback=14) -> None:
        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()

        if math.isinf(best_ask) or math.isinf(best_bid):
            if len(self.midprices) < lookback and len(self.midprices) > 0:
                self.midprices[current_time] = np.mean(list(self.midprices.values()))
            elif len(self.midprices) >= lookback:
                self.midprices[current_time] = np.mean(list(self.midprices.values())[-lookback:])
        else:
            self.midprices[current_time] = (best_ask + best_bid) / 2

    def observe(self) -> str:
        s = "--------------\n"
        names = ["buy_matched", "buy_unmatched", "sell_matched", "sell_unmatched"]
        for i, heap in enumerate(self.heaps):
            s += names[i]
            s += "\n"
            # s += f'Top order_id: {heap.peek_order().order_id}\n'
            s += f"Top price: {abs(heap.peek())}\n"
            s += f"Number of orders: {heap.count()}\n\n\n"

        return s

    def cancel_outdated_orders(self, current_time: int) -> None:
        # removes orders which are not yet matched and their allowed time for matching has passed
        # TODO: check for performance as this might be heavy. Well, better than expected :D
        sell_to_remove = []
        for ord_id, order in self.sell_unmatched.order_dict.items():
            if order.valid_until is not None and order.valid_until < current_time:
                sell_to_remove.append(ord_id)
        for ord_id in sell_to_remove:
            self.sell_unmatched.remove(order_id=ord_id)
        buy_to_remove = []
        for ord_id, order in self.buy_unmatched.order_dict.items():
            if order.valid_until is not None and order.valid_until < current_time:
                buy_to_remove.append(ord_id)
        for ord_id in buy_to_remove:
            self.buy_unmatched.remove(order_id=ord_id)
