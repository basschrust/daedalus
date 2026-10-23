from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.agent.agent import Agent
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price
from marketsim.private_values.private_values import PrivateValues

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.market.security import Security


class SpoofingAgent(Agent):
    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        configuration: dict | None = None,
        group: str = "Spoofers",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ):
        default_configuration = {
            "q_max": 1000,
            "pv_var": 0.2,
            "order_size": 10,
            "spoofing_size": 15000,
            # "normalizers": dict,
            "normalizers": {"fundamental": 100.0, "invt": 12, "cash": 1000000},
            # spoofing_times: list[int] | None
            "spoofing_times": [23, 67],
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

        if final_configuration["pv_var"] is not None:
            self.pv = final_configuration["pv_var"]
        else:
            self.pv = PrivateValues(
                final_configuration["q_max"], float(final_configuration["pv_var"])
            )
        self.spoofing_size = final_configuration["spoofing_size"]
        self.order_size = final_configuration["order_size"]
        self.last_value = 0  # value at last time step (liquidate all inventory)
        self.normalizers = final_configuration[
            "normalizers"
        ]  # A dictionary {"fundamental": float, "invt": float, "cash": float}
        self.spoofing_size = final_configuration["spoofing_size"]
        self.regular_order_size = final_configuration["order_size"]

        self.q_max = final_configuration["q_max"]
        self.pv_var = final_configuration["pv_var"]

        if final_configuration["spoofing_times"] is None:
            self.spoofing_times = [50, 1300, 2345, 3709]
        else:
            self.spoofing_times = final_configuration["spoofing_times"]

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self, asset_id: int):
        mean, r, T = self.markets[asset_id].get_info()
        t = self.markets[asset_id].get_time()
        val = self.markets[asset_id].get_fundamental_value()

        rho = (1 - r) ** (T - t)

        estimate = (1 - rho) * mean + rho * val
        # print(f'It is time {t} with final time {T} and I observed {val} and my estimate is {rho, estimate}')
        return estimate

    def take_action(self, current_time: int):
        for asset_id, market in self.markets.items():
            orders = []
            if current_time in self.spoofing_times:
                market.withdraw_all(agent_id=self.get_id())
                spoof_side = random.choice([-1, 1])

                # TODO - calculate them - we can inspect LOB (just like WTs)!
                if spoof_side == 1:
                    regular_order_price = market.last_traded_price + Price(0.4)
                    spoofing_order_price = market.last_traded_price - Price(0.01)
                else:
                    regular_order_price = market.last_traded_price - Price(0.4)
                    spoofing_order_price = market.last_traded_price + Price(0.01)
                    # TODO: make those prices configurable

                # TODO: we can make it a little more sophisticated! lol
                # 1. toss sides
                # 2. take action in some special time ticks (high or low volume? or defined tick numer)
                # 3. add valid_until field to the Order class and cancel each order which reaches this value
                # (cancel at the beginning of the step)

                # Regular order.
                # self.logger.info(f"Normalizers fundamental: {self.normalizers['fundamental'].__class__}")
                regular_order = Order(
                    price=Price(
                        float(regular_order_price)
                    ),  # * float(self.normalizers["fundamental"])),
                    quantity=self.order_size,
                    agent_id=self.get_id(),
                    time=current_time,
                    order_type=-1 * spoof_side,
                    asset_id=asset_id,
                    valid_until=current_time + 100,
                )
                orders.append(regular_order)

                # Spoofing Order
                # TODO: we have to cancel the order very quickly, before it is executed!
                # TODO: but the traders to be tricked are HBL - so they have to trade some reasonable volume!
                spoofing_order = Order(
                    price=Price(
                        float(spoofing_order_price)
                    ),  # * float(self.normalizers["fundamental"])),
                    quantity=self.spoofing_size,
                    agent_id=self.get_id(),
                    time=current_time,
                    order_type=spoof_side,
                    asset_id=asset_id,
                    valid_until=current_time + 2,  # TODO: check, maybe +1 would be enough
                )
                orders.append(spoofing_order)

            market.add_orders(orders)

    def __str__(self):
        return f"SP{self.agent_id}"

    def get_pos_value(self) -> float:
        return 0  # TODO: check self.pv.value_at_position(self.position)

    def reset(self):
        self.pv = PrivateValues(self.q_max, self.pv_var)
        self.position = 0
        # self.cash = 0
        self.last_value = 0
