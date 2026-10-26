from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price
from marketsim.market.instruments.security import Security
from marketsim.private_values.private_values import PrivateValues

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository


class ZIAgentNotInformed(Agent):
    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        group: str = "ZINI",
        configuration: dict = None,
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        # q_max: int, shade: List, pv_var: float, eta: float = 1.0
        # , lam=1.0, mean_volume: float = 5.0):
        default_configuration = {
            "q_max": 1000,
            "shade": [0.1, 0.2],
            "pv_var": 0.1,
            "lam": 0.5,
            "eta": 1.0,
            "mean_volume": 5.0,
        }
        final_configuration = default_configuration | (
            configuration if configuration is not None else {}
        )
        super().__init__(
            markets=markets,
            repository=repository,
            group=group,
            configuration=final_configuration,
            output_dir=output_dir,
            fx_rate_provider=fx_rate_provider,
        )
        # self.group = "ZINI"
        self.q_max = final_configuration.get("q_max")
        self.pv_var = final_configuration.get("pv_var")
        # print(f"q_max: {self.q_max}, pv_var: {self.pv_var}")
        self.pv = PrivateValues(self.q_max, float(self.pv_var))
        self.shade = final_configuration.get("shade")
        # print(f"shade: {self.shade}")
        self.eta = final_configuration.get("eta")
        self.lam = final_configuration.get("lam")  # activity parameter
        self.mean_volume = final_configuration["mean_volume"]

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self, current_time: int) -> Price:
        estimate = self.market.last_traded_price
        print(
            f"ZINI - It is time {current_time} with final I observed last traded price {estimate}, so my estimate is {estimate}"
        )
        return estimate

    def take_action(self, current_time: int):
        for asset_id, market in self.markets.items():
            orders = []
            if random.random() < self.lam:
                side = random.choice([BUY, SELL])
                quantity = np.random.poisson(lam=self.mean_volume)  # AK why not volume?
                # quantity = 3 if side == BUY else 5 # just for tests - use prime numbers to check splittings properly

                # if estimate is None:
                #     estimate = Price(self.estimate_fundamental(current_time=current_time))
                # print(f"The estimate: {estimate}")
                # print(f"Private values: {self.pv.values}")
                estimate = market.last_traded_price
                spread = Decimal(self.shade[1] - self.shade[0])
                valuation_offset = Price(spread * Decimal(random.random()) + Decimal(self.shade[0]))

                # Cache private value lookup (avoid duplicate computation when eta != 1.0)
                pv_value = Price(self.pv.value_for_exchange(self.position[asset_id], side))

                if side == BUY:
                    price = estimate + pv_value - valuation_offset
                    # AK: print(f"price: {price}, estimate: {estimate}, pv_value: {pv_value},  valuation_offset: {valuation_offset}")
                else:
                    price = estimate + pv_value + valuation_offset

                if self.eta != 1.0:
                    base_price = estimate + pv_value
                    if side == BUY:
                        best_price = market.order_book.get_best_ask()
                        if (
                            base_price - best_price
                        ) > self.eta * valuation_offset and best_price != np.inf:
                            price = best_price
                    else:
                        best_price = market.order_book.get_best_bid()
                        if (
                            best_price - base_price
                        ) > self.eta * valuation_offset and best_price != np.inf:
                            price = best_price

                if price > 0:
                    order = Order(
                        price=Price(price),
                        quantity=quantity,
                        agent_id=self.agent_id,
                        time=current_time,
                        order_type=side,
                        asset_id=asset_id,
                    )
                    orders.append(order)
                else:
                    print(
                        f"Order not placed as calculated price was negative: {price}, q: {quantity}"
                    )

            market.add_orders(orders)

    def __str__(self) -> str:
        return f"ZI_non_informed{self.agent_id}"
        # TODO: AK to info func: with PVs: {self.pv.values}'

    def get_pos_value(self) -> float:
        return self.pv.value_at_position(self.position)

    def reset(self) -> None:
        self.position = 0
        # self.cash = 0
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self._order_counter = 0
