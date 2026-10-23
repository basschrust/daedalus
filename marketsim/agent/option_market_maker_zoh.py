from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.agent.agent import Agent
from marketsim.market import Security, Price, FXRateProvider
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price, Security
from marketsim.market.valuation_libs.BlackScholes import BSCall, BSPut

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository


class OptionMMZOHAgent(Agent):
    ### Market Maker Zero Order Hold Agent for Options! -
    # A MM which just takes into account last traded price and sets new order ladder
    # symmetrically on both sides of this last traded price in each rebalance period
    ###
    def __init__(self, *, markets: list[Security], repository: Repository, configuration: dict|None=None,
                 group: str = "OptionsMMZOH", output_dir: Path|str = "tmp",
                 fx_rate_provider: FXRateProvider|None=None) -> None:
        default_configuration = {"xi": 0.1,
                                 "K": 3,
                                 "omega": 0.1,
                                 "rebalance_period": 5,
                                 "volume": 7,
                                 "q_max": 3000,
                                 "rebalance_by": "time",
                                 "rebalance_volume": 70}
        final_configuration = default_configuration | (configuration if configuration is not None else {})
        super().__init__(markets=markets, repository=repository, configuration=final_configuration,
                         group=group, output_dir=output_dir, fx_rate_provider=fx_rate_provider)

        self.option_markets = { market.asset_id: market for market in markets if
                               market.instrument_class == "option" }

        self.underlying_markets = { market.underlying.asset_id: market.underlying for m_id, market in self.option_markets.items() }

        self.derivatives_map  = { m_id: market.underlying.asset_id for m_id, market in self.option_markets.items() }
        self.markets = self.option_markets | self.underlying_markets
        self.greeks = defaultdict(dict) # per option, per unit,
        self.greeks_agg = defaultdict(dict) # per underlying, per position

        self.position = { m_id: 0 for m_id in self.markets }

        #  TODO: Market Making parameters - per each market:
        self.xi = Decimal(final_configuration["xi"])  # step of the order ladder
        self.K = final_configuration["K"]  # number of orders in the ladder
        self.omega = Decimal(final_configuration["omega"])  # bid ask spread between two closest MM quotations
        self.rebalance_period = final_configuration["rebalance_period"]
        self.rebalance_by = final_configuration["rebalance_by"]  # time or volume
        self.rebalance_volume = final_configuration["rebalance_volume"]
        self.last_rebalance_time = 0
        self.cum_volume = 0

        self.volume = final_configuration["volume"]
        self.q_max = final_configuration["q_max"]


    def get_id(self) -> int:
        return self.agent_id

    def should_rebalance(self, *, current_time:int, market: Security) -> bool:
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
            self.cum_volume += market.traded_prices.get(current_time-1, {}).get("Volume", 0)
            if self.cum_volume >= self.rebalance_volume:
                self.last_rebalance_time = current_time
                self.cum_volume = 0
                return True
        return False

    def take_action(self, current_time: int):
        for option_id, option_market in self.option_markets.items():
            underlying_market = self.underlying_markets[self.derivatives_map[option_id]]
            orders = []
            # add orders only in rebalance periods:
            if self.should_rebalance(current_time=current_time, market=option_market):
                # AK - clear previous orders (should we?)
                self.logger.info("Withdrawing previous orders ()") # how to check number of orders of this agent?
                option_market.withdraw_all(agent_id=self.agent_id)
                # AK - don't withdraw, but also don't blindly add new orders - just ensure they are balanced
                # that's basically the same to just withdraw all and create new, the problem might be with timing
                # - we could loose the slot in the queue of waiting orders

                # Get the best bid and best ask
                best_ask = option_market.order_book.get_best_ask()
                best_bid = option_market.order_book.get_best_bid()

                self.logger.info(f"Best bid {best_bid}, best ask: {best_ask}")

                #estimate = self.market.last_traded_price
                # TODO: get the theoretical price
                estimate = Price(option_market.get_theoretical_price())

                self.logger.info(f"Last traded price: {estimate}")
                HALF = Decimal("0.5")
                st = max(estimate + HALF * self.omega, best_bid)
                bt = min(estimate - HALF * self.omega, best_ask)
                self.logger.info(f"Setting basic spread to: {bt}, {st}")
                buy_volume = self.volume
                sell_volume = self.volume
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
                    if price_bid > 0:
                        orders.append(
                            Order(
                                price=price_bid,
                                quantity=buy_volume, #7,#1, # we ćould raise the quantity in each ladder step...
                                agent_id=self.agent_id,
                                time=current_time,
                                order_type=BUY,
                                asset_id=option_id,
                            )
                        )
                        orders.append(
                            Order(
                                price= Price(st + (k + 1)*self.xi),
                                quantity=sell_volume, # 7,#1,
                                agent_id=self.agent_id,
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

            # TODO: add orders to the underlying market
            # TODO: if the underlying is an index then hedge with futures, other options, stocks, etc.



    def get_pos_value(self) -> float:
        raise
        return 0

    def __str__(self):
        return f'Opt_MM_ZOH{self.agent_id}'

    ### the Derivative agent typical methods:
    def calculate_greeks(self, as_of_day: int|None=None):
        as_of_day = as_of_day if as_of_day is not None else self.current_day
        for asset_id, market in self.underlying_markets.items():
            self.greeks_agg[asset_id] = {"delta": self.position[asset_id], "gamma": 0,
                                         "theta": 0, "vega": 0, "rho": 0}

        for option_id, option_market in self.option_markets.items():
            underlying_market = self.underlying_markets[self.derivatives_map[option_id]]
            underlying_id = underlying_market.asset_id
            # TODO: the volatility should be taken calculated/estimated from underlying
            if option_market.option_side == "CALL":
                self.greeks[option_id] = BSCall(S=underlying_market.last_traded_price,
                                                K=option_market.strike,
                            r=option_market.r, volatility=option_market.volatility,
                                    Time=(option_market.expiration-as_of_day)/248, d=0.0)
            elif option_market.option_side == "PUT":
                self.greeks[option_id] = BSPut(S=underlying_market.last_traded_price,
                                               K=option_market.strike,
                                r=option_market.r, volatility=option_market.volatility,
                                Time=(option_market.expiration-as_of_day)/248, d=0.0)

            # now add and aggregate for the underlying
            for greek_letter in self.greeks_agg[underlying_id]:
                self.greeks_agg[underlying_id][greek_letter] += (self.greeks[option_id][greek_letter]
                                                                 * self.position[option_id])

        self.logger.info(f"Greeks: {self.greeks}")
        self.logger.info(f"Position: {self.position}")
        self.logger.info(f"Greeks aggregated: {self.greeks_agg}")


    def hedge(self, *, current_time: int) -> None:
        # either as adjusting MM orders or by paying spread
        # simple delta hedge
        for asset_id, underlying_market in self.underlying_markets.items():
            delta_pos = self.greeks_agg[asset_id].get("delta")
            # check position and align, check minimum difference which leads to rebalance

            required_adjustment = - int(delta_pos)

            if required_adjustment >= 1:
                order = Order(price=underlying_market.last_traded_price,
                              quantity=required_adjustment,
                              agent_id=self.agent_id,
                              time=current_time,
                              order_type=BUY,
                              asset_id=underlying_market.asset_id,
                              valid_until=current_time + 10,
                              )
                self.logger.info(f"Adding buy order to underlying market: {order}")
                underlying_market.add_orders([order])
            elif required_adjustment <= -1:
                order = Order(price=underlying_market.last_traded_price,
                              quantity=abs(required_adjustment),
                              agent_id=self.agent_id,
                              time=current_time,
                              order_type=SELL,
                              asset_id=underlying_market.asset_id,
                              valid_until=current_time + 10,
                              )
                self.logger.info(f"Adding sell order to underlying market: {order}")
                underlying_market.add_orders([order])

    def sod(self):
        expired_options = [m_id for m_id, market in self.option_markets.items() if market.status == "expired"]
        for option_id in expired_options:
            self.option_markets.pop(option_id)
        super().sod()
