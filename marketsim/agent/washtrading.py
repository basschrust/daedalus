from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price
from marketsim.market import Price, FXRateProvider

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.market.security import Security


class WashTradingAgent(Agent):
    def __init__(self, *, markets: list[Security], repository: Repository, configuration: dict| None=None,
                 group: str = "WashTraders", output_dir: Path|str="tmp",
                 fx_rate_provider: FXRateProvider|None=None):
        default_configuration = { "q_max": 1000,
                                  "lam": 0.5,
                                  "pool_id": 0,
                                  "mean_volume":  5.0,
                                  "manipulation_period_start": 10,
                                  "manipulation_period_end": 30,
                                  "manipulation_type": "PUSH_DOWN",
                                  "manipulation_side": "BUY",
                                  "spread": Price(0.8),
                                }
        final_configuration = default_configuration | (configuration if configuration is not None else {})
        super().__init__(markets=markets,
                         repository=repository,
                         configuration=final_configuration,
                         group=group,
                         output_dir=output_dir,
                         fx_rate_provider=fx_rate_provider)

        self.q_max = final_configuration["q_max"]
        self.lam = final_configuration["lam"] # yet not used - probably used in the non-manipulation period
        self.pool_id = final_configuration["pool_id"]
        # self.manipulation_boundaries =
        # configuration["manipulation_boundaries # what if several such periods? maybe list of dicts?
        self.mean_volume = final_configuration["mean_volume"]
        self.manipulation_period_start = final_configuration["manipulation_period_start"]
        self.manipulation_period_end = final_configuration["manipulation_period_end"]
        self.manipulation_type = final_configuration["manipulation_type"]
        self.manipulation_side = final_configuration["manipulation_side"]
        self.spread = final_configuration["spread"]

        # specific for this (WashTrading) type:
        # TODO: set it for all markets
        self.price_to_reach = markets[0].last_traded_price
        self.quantity = 0

        # info about partner in crime:   - Wash Trading counterparty
        self.wt_counterparty = None
        self.wt_pool = None # TODO: or better use the WashTradingPool object?


    def get_id(self) -> int:
        return self.agent_id


    def take_action(self, current_time: int):
        # TODO: parameters per each market
        # period = self.manipulation_boundaries["manipulation_period"]
        orders = []
        length = self.manipulation_period_end - self.manipulation_period_start  # let' make it kiss (keep it silly simple)
        for asset_id, market in self.markets.items():
            # TODO: they should not manipulate in the same day times every day :)
            if self.manipulation_period_start <= current_time <= self.manipulation_period_end:
                # so act as designed

                # if q_max almost reached we could try to push more with spread?
                if self.lam > random.random(): # let's see what happens when we push always
                    # but then this method is easy to find out
                    # TODO: if the position is heavily unbalanced set more aggressive price, too
                    if current_time % 3 == 1:
                        # in odd time ticks calculate volume & price and make order of one side
                        market.withdraw_all(agent_id=self.agent_id) # cancel also only in odd ticks

                        if self.manipulation_type == "PULL_UP":
                            self.quantity = int(
                                (self.q_max - abs(self.position[asset_id])) / (length * self.lam))
                            # TODO: check if this liquidity check gives the reached or exceeded volumes (what happens on boundaries)
                            self.price_to_reach = market.order_book.get_ask_at_volume(
                                self.quantity / 4)  # + Price(0.01)

                            if not math.isfinite(self.price_to_reach):
                                self.price_to_reach = market.last_traded_price + Price(0.5)

                            if self.manipulation_side == "BUY":
                                if self.price_to_reach > 0 and self.quantity > 0:
                                    order = Order(
                                        price=self.price_to_reach,
                                        quantity=self.quantity,
                                        agent_id=self.agent_id,
                                        asset_id=asset_id,
                                        time=current_time,
                                        order_type=1 if self.manipulation_side == 'BUY' else -1,
                                    )
                                    orders.append(order)
                                    market.add_orders(orders)
                                    return
                            else:
                                self.price_to_reach = max(self.price_to_reach - Price(0.01), Price(0.01))

                        elif self.manipulation_type == "PUSH_DOWN":
                            self.quantity = int(
                                (self.q_max - abs(self.position[asset_id])) / (length * self.lam))
                            self.price_to_reach = market.order_book.get_bid_at_volume(self.quantity / 4)
                            # TODO: add also a memory what price we set in the previous step (and was the order executed?)
                            # TODO: and matched with the other side of the WT or just MM or Noise?
                            if math.isfinite(self.price_to_reach):
                                self.price_to_reach = self.price_to_reach  # - Price(0.01)
                            else:
                                self.price_to_reach = max(market.last_traded_price - Price(0.5), Price(0.01))

                            if self.manipulation_side == "BUY":
                                self.price_to_reach = self.price_to_reach+Price(0.01)
                            else:
                                if self.price_to_reach > 0 and self.quantity > 0:
                                    order = Order(
                                        price=self.price_to_reach,
                                        quantity=self.quantity,
                                        agent_id=self.agent_id,
                                        asset_id=asset_id,
                                        time=current_time,
                                        order_type=1 if self.manipulation_side == 'BUY' else -1,
                                    )
                                    orders.append(order)
                                    market.add_orders(orders)
                                    return

                        else:
                            raise ValueError(f"Invalid manipulation type {self.manipulation_type}.")
                    elif current_time % 3 == 2:
                        # in even time ticks (2/3) place the other side order, use price and volume calculated in previous step
                        if self.manipulation_type == "PULL_UP":
                            if self.manipulation_side == "BUY":
                                self.quantity = 0
                            else:
                                if self.price_to_reach > 0 and self.quantity > 0:
                                    order = Order(
                                        price=self.price_to_reach,
                                        quantity=self.quantity,
                                        agent_id=self.agent_id,
                                        asset_id=asset_id,
                                        time=current_time,
                                        order_type=1 if self.manipulation_side == 'BUY' else -1,
                                    )
                                    orders.append(order)

                        elif self.manipulation_type == "PUSH_DOWN":
                            if self.manipulation_side == "SELL":
                                self.quantity = 0
                            else:
                                if self.price_to_reach > 0 and self.quantity > 0:
                                    order = Order(
                                        price=self.price_to_reach,
                                        quantity=self.quantity,
                                        agent_id=self.agent_id,
                                        asset_id=asset_id,
                                        time=current_time,
                                        order_type=1 if self.manipulation_side == 'BUY' else -1,
                                    )
                                    orders.append(order)

                        else:
                            raise ValueError(
                                f"Invalid manipulation type {self.manipulation_type}")
                    else:
                        self.logger.info(f"WT waiting turn in time {current_time}.")

            else:
                # TODO: be a normal ZI agent :)  (sometimes smoothly align position using PVs)
                if random.random() < self.lam:
                    # but if we are after washtrading then let's try to rebalance as much as we can
                    # so let only one side of the orders
                    if current_time < self.manipulation_period_start:
                        side = random.choice([BUY, SELL])
                        quantity = np.random.poisson(lam=self.mean_volume) if abs(self.position[asset_id]) < self.q_max else 1
                    else:
                        # so we are after the manipulation period - let's just rebalance here
                        side = 1 if self.manipulation_side == 'BUY' else -1
                        # but how not to exceed the q_max? - like this:   # but we don't know how many steps are left
                            # till the end of the simulation, it should depend on the momentary liquidity

                        # taking into account the total position of the WT pool
                        pool_position = self.wt_pool.get_position(asset_id=asset_id)
                        quantity = 0
                        if pool_position > 10:
                            if side == SELL:
                                quantity = int((self.q_max - abs(self.position[asset_id])) * (0.5 + 0.5 *random.random()) / length)
                        if pool_position < 10:
                            if side == BUY:
                                quantity = int((self.q_max - abs(self.position[asset_id])) * (0.5 + 0.5 * random.random()) / length)

                    spread = self.spread # maybe some other spread should be put here
                    # TODO: some rebalance spread parameter?
                    price = market.last_traded_price + Price(0.05 * spread * (random.random() - 0.5))

                    if price > 0 and quantity > 0:
                        order = Order(
                            price=price,
                            quantity=quantity,
                            agent_id=self.agent_id,
                            asset_id=asset_id,
                            time=current_time,
                            order_type=side,
                        )
                        orders.append(order)
            market.add_orders(orders)

    def __str__(self):
        return f'WT_{self.pool_id}_{self.agent_id}'

    def reset(self):
        self.position = 0
        # self.cash = 0

    def get_pos_value(self) -> float:
        return 0

    def set_wt_pool(self, wt_pool: WashTradingPool):
        self.wt_pool = wt_pool


class WashTradingPool:
    def __init__(self, *, buy_pool: list, sell_pool: list):
                 # manipulation_type: str, manipulation_start: int, manipulation_end: int):
        # TODO: maybe we could store a reciprocal hook in each of those agents in the pool so that they can
        # check the balance of each other and push their position towards equilibrium?
        # self.market = market
        # self.id = pool_id
        # self.type = manipulation_type # 'PULL_UP' or 'PUSH_DOWN'
        # self.manipulation_start = manipulation_start # tau_1, manipulation starts here
        # self.manipulation_end = manipulation_end # tau_2, manipulation ends here
        self.buy_pool = buy_pool
        self.sell_pool = sell_pool

    def get_id(self) -> int:
        return self.id

    def manipulation_start(self):
        # send signal to all agents in the pool
        pass

    def get_position(self, *, asset_id: int) -> int:
        position = 0
        for agent in self.buy_pool:
            position += agent.position[asset_id]
        for agent in self.sell_pool:
            position += agent.position[asset_id]

        return position

