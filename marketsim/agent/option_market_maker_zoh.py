from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.agent.agent import Agent
from marketsim.agent.helpers.mm_instrument_limits import InstrumentLimitMixin, historical_volume
from marketsim.market import FXRateProvider, Price, Security
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order
from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository


class OptionMMZOHAgent(InstrumentLimitMixin, Agent):
    ### Market Maker Zero Order Hold Agent for Options! -
    # A MM which just takes into account last traded price and sets new order ladder
    # symmetrically on both sides of this last traded price in each rebalance period
    ###
    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        configuration: dict | None = None,
        group: str = "OptionsMMZOH",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        default_configuration = {
            "xi": 0.1,
            "K": 3,
            "omega": 0.1,
            "rebalance_period": 5,
            "volume": 7,
            "q_max": 3000,
            "rebalance_by": "time",
            "rebalance_volume": 70,
        }
        final_configuration = default_configuration | (
            configuration if configuration is not None else {}
        )
        super().__init__(
            markets=markets,
            repository=repository,
            configuration=final_configuration,
            group=group,
            output_dir=output_dir,
            fx_rate_provider=fx_rate_provider,
        )

        self.option_markets = {
            market.asset_id: market for market in markets if market.instrument_class == "option"
        }

        self.underlying_markets = {
            market.underlying.asset_id: market.underlying
            for m_id, market in self.option_markets.items()
        }

        self.derivatives_map = {
            m_id: market.underlying.asset_id for m_id, market in self.option_markets.items()
        }
        self.hedge_futures = {
            market.asset_id: market
            for market in markets
            if market.instrument_class == "future"
            and market.underlying.asset_id in self.underlying_markets
        }
        self.markets = self.option_markets | self.underlying_markets | self.hedge_futures
        self.greeks = defaultdict(dict)  # per option, per unit,
        self.greeks_agg = defaultdict(dict)  # per underlying, per position

        self.position = {m_id: 0 for m_id in self.markets}

        #  TODO: Market Making parameters - per each market:
        self.xi = Decimal(final_configuration["xi"])  # step of the order ladder
        self.K = final_configuration["K"]  # number of orders in the ladder
        self.omega = Decimal(
            final_configuration["omega"]
        )  # bid ask spread between two closest MM quotations
        self.rebalance_period = final_configuration["rebalance_period"]
        self.rebalance_by = final_configuration["rebalance_by"]  # time or volume
        self.rebalance_volume = final_configuration["rebalance_volume"]
        self.last_rebalance_time = 0
        self.cum_volume = 0

        self.volume = final_configuration["volume"]

        self.initialize_instrument_limits(self.option_markets, final_configuration)

        self.volume_by_asset = {}
        for asset_id, market in self.option_markets.items():
            daily = historical_volume(market)
            if daily is None or daily <= 0:
                quantity = int(final_configuration["volume"])
            else:
                fraction = float(final_configuration.get("volume_fraction", 0.10))
                if not 0 < fraction <= 1:
                    raise ValueError("Invalid option MM volume_fraction")
                rebalances = math.ceil(40 / self.rebalance_period)
                quantity = max(
                    1,
                    round(daily * fraction / (rebalances * 2 * self.K)),
                )
            self.volume_by_asset[asset_id] = quantity

    def get_id(self) -> int:
        return self.agent_id

    def should_rebalance(self, *, current_time: int, market: Security) -> bool:
        if current_time == 0:
            return True
        if self.rebalance_by == "time":
            if current_time % self.rebalance_period == 0:
                return True
        elif self.rebalance_by == "volume":
            # in case of low trading volume period rebalance by time anyway:
            if current_time - self.last_rebalance_time >= self.rebalance_period:
                # TODO: prepare a more sophisticated time comparison function
                self.last_rebalance_time = current_time
                self.cum_volume = 0
                return True
            # TODO: make the calculation, but what about methods - own, global, side, cash?
            self.cum_volume += market.traded_prices.get(current_time - 1, {}).get("volume", 0)
            if self.cum_volume >= self.rebalance_volume:
                self.last_rebalance_time = current_time
                self.cum_volume = 0
                return True
        return False

    def take_action(self, current_time: int):
        for option_id, option_market in self.option_markets.items():
            orders = []
            # add orders only in rebalance periods:
            if self.should_rebalance(current_time=current_time, market=option_market):
                # AK - clear previous orders (should we?)
                self.logger.info(
                    "Withdrawing previous orders ()"
                )  # how to check number of orders of this agent?
                option_market.withdraw_all(agent_id=self.agent_id)
                # AK - don't withdraw, but also don't blindly add new orders - just ensure they are balanced
                # that's basically the same to just withdraw all and create new, the problem might be with timing
                # - we could loose the slot in the queue of waiting orders

                # Get the best bid and best ask
                best_ask = option_market.order_book.get_best_ask()
                best_bid = option_market.order_book.get_best_bid()

                self.logger.info(f"Best bid {best_bid}, best ask: {best_ask}")

                # estimate = self.market.last_traded_price
                # TODO: get the theoretical price
                estimate = Price(option_market.get_theoretical_price())

                self.logger.info(f"Last traded price: {estimate}")
                HALF = Decimal("0.5")
                st = estimate + HALF * self.omega
                bt = estimate - HALF * self.omega
                if best_bid is not None and math.isfinite(float(best_bid)):
                    st = max(st, Price(best_bid))
                if best_ask is not None and math.isfinite(float(best_ask)):
                    bt = min(bt, Price(best_ask))

                self.logger.info(f"Setting basic spread to: {bt}, {st}")
                buy_volume, sell_volume = self.limited_quote_sizes(
                    option_id, self.volume_by_asset[option_id], self.K
                )
                # TODO: adjust the spread for position rebalancing
                # if abs(self.position) > self.q_max/2:
                #     if self.position > 0:
                #         # the MM position is very long - needs to sell, so move the prices up
                #         st = st + HALF * self.omega
                #         bt = bt + HALF * self.omega
                #         if self.position > 3/4 * self.q_max:
                #             # if getting close to max we also limit the volume of buy orders:
                #             buy_volume = int(buy_volume /2)
                #             if self.position > self.q_max:
                #                 # if we exceeded the q_max then volume should be only symbolic
                #                 buy_volume = 1
                #     else:
                #         # the MM position is very short - has to buy more, lower the prices
                #         st = st - HALF * self.omega
                #         bt = bt - HALF * self.omega
                #         if self.position < -3/4 * self.q_max:
                #             sell_volume = int(sell_volume /2)
                #             if self.position < - self.q_max:
                #                 sell_volume = 1

                self.logger.info(f"Basic spread (option market) adjusted to: {bt}, {st}")

                for k in range(self.K):
                    price_bid = Price(bt - (k + 1) * self.xi)
                    if price_bid > 0 and buy_volume > 0:
                        orders.append(
                            Order(
                                price=price_bid,
                                quantity=buy_volume,  # 7,#1, # we ćould raise the quantity in each ladder step...
                                agent_id=self.agent_id,
                                day=self.current_day,
                                time=current_time,
                                order_type=BUY,
                                asset_id=option_id,
                            )
                        )
                    if sell_volume > 0:
                        orders.append(
                            Order(
                                price=Price(st + (k + 1) * self.xi),
                                quantity=sell_volume,  # 7,#1,
                                agent_id=self.agent_id,
                                day=self.current_day,
                                time=current_time,
                                order_type=SELL,
                                asset_id=option_id,
                            )
                        )

            # adding orders to the option market:
            option_market.add_orders(orders)

        # calculate Greeks
        self.calculate_greeks()
        # delta / gamma hedge
        self.hedge(current_time=current_time)

        # Index exposure is hedged with futures; basis arbitrage is handled by arbitrageurs.

    def get_pos_value(self) -> float:
        raise
        return 0

    def __str__(self):
        return f"Opt_MM_ZOH{self.agent_id}"

    ### the Derivative agent typical methods:
    def calculate_greeks(self, *, as_of_day: int | None = None):
        as_of_day = as_of_day if as_of_day is not None else self.current_day
        for asset_id, market in self.underlying_markets.items():
            self.greeks_agg[asset_id] = {
                "delta": self.position[asset_id],
                "gamma": 0,
                "theta": 0,
                "vega": 0,
                "rho": 0,
            }

        for option_id, option_market in self.option_markets.items():
            underlying_market = self.underlying_markets[self.derivatives_map[option_id]]
            underlying_id = underlying_market.asset_id
            # TODO: the volatility should be taken calculated/estimated from underlying
            if option_market.option_side == "CALL":
                self.greeks[option_id] = BSCall(
                    S=underlying_market.last_traded_price,
                    K=option_market.strike,
                    r=option_market.r,
                    volatility=option_market.volatility,
                    Time=(option_market.expiration - as_of_day) / 252,
                    d=0.0,
                )
            elif option_market.option_side == "PUT":
                self.greeks[option_id] = BSPut(
                    S=underlying_market.last_traded_price,
                    K=option_market.strike,
                    r=option_market.r,
                    volatility=option_market.volatility,
                    Time=(option_market.expiration - as_of_day) / 252,
                    d=0.0,
                )

            # now add and aggregate for the underlying
            for greek_letter in self.greeks_agg[underlying_id]:
                self.greeks_agg[underlying_id][greek_letter] += (
                    self.greeks[option_id][greek_letter]
                    * self.position[option_id]
                    * option_market.multiplier
                )

        self.logger.info(f"Greeks: {self.greeks}")
        self.logger.info(f"Position: {self.position}")
        self.logger.info(f"Greeks aggregated: {self.greeks_agg}")

    def _pending_hedge_orders(self, market: Security, current_time: int) -> list[Order]:
        """Own unfilled orders, both waiting in the book and awaiting this tick."""
        pending: dict[int, Order] = {}
        book = market.order_book
        if book is not None:
            for queue in (book.buy_unmatched, book.sell_unmatched):
                for order in queue.order_dict.values():
                    if order.agent_id == self.agent_id and (
                        order.valid_until is None or order.valid_until >= current_time
                    ):
                        pending[order.order_id] = order

        for tick, activities in market.event_queue.scheduled_activities.items():
            if tick < current_time:
                continue
            for order in activities:
                if order.agent_id == self.agent_id and (
                    order.valid_until is None or order.valid_until >= current_time
                ):
                    pending[order.order_id] = order
        return list(pending.values())

    def _cancel_pending_hedges(self, market: Security) -> None:
        """Cancel unfilled hedge orders, including orders not yet in the book."""
        activities = market.event_queue.scheduled_activities
        for tick, orders in list(activities.items()):
            retained = [order for order in orders if order.agent_id != self.agent_id]
            if retained:
                activities[tick] = retained
            else:
                activities.pop(tick, None)
        if market.order_book is not None:
            # Inspect the actual unmatched queues: a partially filled order
            # can acquire a new order ID absent from FourHeap.agent_id_map.
            for queue in (
                market.order_book.buy_unmatched,
                market.order_book.sell_unmatched,
            ):
                for order_id, order in list(queue.order_dict.items()):
                    if order.agent_id == self.agent_id:
                        market.order_book.remove(order_id)

    def _submit_hedge_adjustment(
        self, *, market: Security, desired: int, current_time: int
    ) -> None:
        """Reconcile target adjustment against open orders before quoting again."""
        pending = self._pending_hedge_orders(market, current_time)
        pending_signed = sum(order.order_type * order.quantity for order in pending)
        pending_gross = sum(order.quantity for order in pending)
        # A reversal, over-sized order or opposite-side pair is stale. Replace
        # it rather than entering an offsetting order and risking both fills.
        if pending and (
            desired == 0
            or (desired > 0) != (pending_signed > 0)
            or pending_gross != abs(pending_signed)
            or abs(pending_signed) > abs(desired)
        ):
            self._cancel_pending_hedges(market)
            pending_signed = 0

        adjustment = desired - pending_signed
        if adjustment == 0:
            return
        order = Order(
            price=market.last_traded_price,
            quantity=abs(adjustment),
            agent_id=self.agent_id,
            day=market.current_day,
            time=current_time,
            order_type=BUY if adjustment > 0 else SELL,
            asset_id=market.asset_id,
            valid_until=current_time + 10,
        )
        self.logger.info(f"Submitting delta-hedge order: {order}")
        market.add_orders([order])

    def hedge(self, *, current_time: int) -> None:
        """Hedge stocks directly and non-tradable indices via matching futures."""
        for index_id, underlying in self.underlying_markets.items():
            delta = float(self.greeks_agg[index_id]["delta"])
            if underlying.is_tradable:
                # Stock units are indivisible; preserve the existing truncation.
                self._submit_hedge_adjustment(
                    market=underlying, desired=-int(delta), current_time=current_time
                )
                continue

            matching = [
                future
                for future in self.hedge_futures.values()
                if future.underlying.asset_id == index_id
            ]
            active = [
                future
                for future in matching
                if future.is_tradable
                and future.status == "active"
                and int(future.expiration) > self.current_day
            ]
            if not active:
                for future in matching:
                    if future.order_book is not None:
                        self._cancel_pending_hedges(future)
                if abs(delta) > 0.5:
                    self.logger.warning(
                        f"No active future for index {underlying.short_name}; "
                        f"unhedged option delta={delta:.2f}"
                    )
                continue

            # Prefer the earliest expiration; other futures positions still
            # contribute to the total delta exposure, including across rolls.
            selected = min(active, key=lambda f: (int(f.expiration), f.asset_id))
            for future in matching:
                if future.asset_id != selected.asset_id and future.order_book is not None:
                    self._cancel_pending_hedges(future)
            futures_delta = sum(
                self.futures_position.get(future.asset_id, 0) * float(future.multiplier)
                for future in matching
            )
            multiplier = float(selected.multiplier)
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError(f"Invalid hedge futures multiplier: {selected.short_name}")
            desired = round((-delta - futures_delta) / multiplier)
            self._submit_hedge_adjustment(
                market=selected, desired=desired, current_time=current_time
            )

    def sod(self):
        expired_options = [
            m_id for m_id, market in self.option_markets.items() if market.status == "expired"
        ]
        for option_id in expired_options:
            self.option_markets.pop(option_id)
        super().sod()
