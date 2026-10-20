from __future__ import annotations

import random
from typing import TYPE_CHECKING
from pathlib import Path

from marketsim.agent.agent import Agent
from marketsim.fourheap.order import Order
from marketsim.fourheap.constants import BUY, SELL
from marketsim.market import Price

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository
    from marketsim.market.security import Security


class MomentumAgent(Agent):
    ### Momentum Agent -
    # Momentum Agent trades using moving average to catch the market trend
    ###
    def __init__(self, *, markets: list[Security], repository: Repository,
                 configuration: dict| None=None, group: str= "MOMENTUM", output_dir: Path|str = "tmp") -> None:
        default_configuration = { "period": 7,
                                  "lam": 0.5,
                                  "q_max": 100,
                                  "threshold": 0.01 }
        final_configuration = default_configuration | (configuration if configuration is not None else {})
        super().__init__(markets=markets, repository=repository, configuration=final_configuration
                         , group=group, output_dir=output_dir)

        self.period = final_configuration["period"] # the period for trend analyzing
        self.lam = final_configuration["lam"] # lambda, the activity parameter
        self.q_max = final_configuration["q_max"] # maximum agent position on each side
        self.threshold = final_configuration["threshold"] # threshold above which we consider the trend to exist


    def get_id(self) -> int:
        return self.agent_id

    def take_action(self, current_time: int):
        for asset_id, market in self.markets.items():
            orders = []
            # compute the momentum
            # compare last traded price (in t-1 - or maybe current last traded?) with price period ticks ago (why not MA?)
            # if price[t-1] > (1+threshold) * price[t-period] -> buy
            # if price[t-1] < (1-threshold) * price[t-period] -> sell
            # amounts? and lambda? yet ignore, take into account in next iteration, price limit?
            if current_time >= self.period:
                market.withdraw_all(agent_id=self.agent_id)
                previous_price = market.traded_prices[current_time-self.period]["close"]
                limit = Price(float(market.last_traded_price) * (0.95 + 0.1*random.uniform(0, 1)))
                # asymptotic approaching the q_max - but let it also reverse the trend when position is high...
                if market.last_traded_price > float(previous_price) * (1+self.threshold):
                    if self.position[asset_id] > 0:
                        # asymptotic approach
                        quantity = int(self.q_max - abs(self.position[asset_id]) / 10)
                    else:
                        # reversing the trend
                        quantity = int(self.q_max / 10)
                    if quantity > 0:
                        orders.append(
                            Order(
                                price=limit,
                                quantity=quantity,
                                agent_id=self.agent_id,
                                time=current_time,
                                order_type=BUY,
                                asset_id=asset_id,
                            )
                        )
                elif market.last_traded_price < float(previous_price) * (1-self.threshold):
                    if self.position[asset_id] < 0:
                        # asymptotic approach
                        quantity = int(self.q_max - abs(self.position[asset_id]) / 10)
                    else:
                        # reversing the trend
                        quantity = int(self.q_max / 10)
                    if quantity > 0:
                        orders.append(
                            Order(
                                price=limit,
                                quantity=quantity,
                                agent_id=self.agent_id,
                                time=current_time,
                                order_type=SELL,
                                asset_id=asset_id,
                            )
                        )

            market.add_orders(orders)


    def get_pos_value(self) -> float:
        # TODO: wtf is this function for?
        return 0

    def __str__(self):
        return f'Momentum_{self.agent_id}'

