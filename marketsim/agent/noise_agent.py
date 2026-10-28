from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.market import FXRateProvider, Price
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order
from marketsim.market.instruments.security import Security

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository


class NoiseAgent(Agent):
    """
    Noise agent - aware only of last traded price and his own position (but this also only roughly)
    """

    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        configuration: dict | None = None,
        group: str = "Noise",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        default_configuration = {
            "q_max": 10000,
            "lam": 0.2,
            "mean_volume": 50.0,
            "mean_spread": Price(0.8),
            "withdraw_old": False,
            "validity_period": 3,
            "use_historical_volatility": False,
            "historical_volatility_scale": 0.1,
        }
        self.configuration = default_configuration | (
            configuration if configuration is not None else {}
        )
        super().__init__(
            markets=markets,
            repository=repository,
            configuration=self.configuration,
            group=group,
            output_dir=output_dir,
            fx_rate_provider=fx_rate_provider,
        )

        self.q_max = self.configuration["q_max"]  # check if doesn't collide with mean_volume
        self.lam = self.configuration["lam"]  # activity parameter
        self.mean_volume = self.configuration["mean_volume"]
        self.mean_spread = self.configuration["mean_spread"]
        # withdrawing old oders when placing new one:
        self.withdraw_old = self.configuration["withdraw_old"]
        self.validity_period = self.configuration["validity_period"]  # validity period for orders

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self, current_time: int) -> Price:
        raise  # should not be used for noise agent

    def take_action(self, current_time: int) -> None:
        for asset_id, market in self.markets.items():
            if market.instrument_class == "index":
                continue
            orders = []
            if random.random() < self.lam:
                if self.withdraw_old:
                    market.withdraw_all(
                        agent_id=self.agent_id
                    )  # TODO: check the impact on resistance/support
                side = random.choice([BUY, SELL])
                # side chosen randomly and stick to that, but later this agent may place many orders on chosen side
                # quantity = np.random.poisson(lam=self.mean_volume)  # AK why not volume?
                mean_volume = self.get_order_mean_volume(
                    asset_id,
                    fallback=self.mean_volume,
                )
                quantity = max(1, int(np.random.poisson(mean_volume)))

                spread_side = random.choice([-1, 1])
                mean_spread = float(self.mean_spread)

                knowledge = self.get_market_knowledge(asset_id)

                if knowledge is not None and self.configuration["use_historical_volatility"]:
                    mean_spread = (
                        float(market.last_traded_price)
                        * knowledge.expected_volatility()
                        * float(self.configuration["historical_volatility_scale"])
                    )

                best_bid = market.order_book.get_best_bid()
                best_ask = market.order_book.get_best_ask()

                best_bid = (
                    float(best_bid)
                    if best_bid is not None and math.isfinite(float(best_bid))
                    else None
                )
                best_ask = (
                    float(best_ask)
                    if best_ask is not None and math.isfinite(float(best_ask))
                    else None
                )

                last_price = float(market.last_traded_price)

                # Maximum price deviation for noise orders.
                max_deviation = float(self.configuration.get("max_price_deviation", 0.02))

                if side == BUY:
                    reference = best_ask if best_ask is not None else last_price
                else:
                    reference = best_bid if best_bid is not None else last_price

                # Preserve random positive/negative deviations,
                # but impose a hard bound.
                raw_offset = spread_side * np.random.poisson(lam=max(0.0, mean_spread))

                max_offset = reference * max_deviation

                offset = max(
                    -max_offset,
                    min(max_offset, raw_offset),
                )

                price = reference + offset

                if price > 0:
                    order = Order(
                        price=Price(price),
                        quantity=quantity,
                        agent_id=self.agent_id,
                        time=current_time,
                        order_type=side,
                        asset_id=market.asset_id,
                        valid_until=current_time + self.validity_period,
                    )
                    orders.append(order)
                else:
                    print(
                        f"Order not placed as calculated price was negative: {price}, q: {quantity}"
                    )

            market.add_orders(orders)

    def __str__(self) -> str:
        return f"Noise_{self.agent_id}"

    def get_pos_value(self) -> Price:
        return self.cash + sum(
            [
                market.last_traded_price * self.position[asset_id]
                for asset_id, market in self.markets.items()
            ]
        )
