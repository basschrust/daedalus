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
            "use_historical_volatility": True,
            "historical_volatility_scale": 0.1,
            "max_price_deviation": 0.02,
            "volume_lookback": 20,
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
            # Indices are not directly tradable.
            if market.instrument_class == "index":
                continue

            orders = []

            # Decide whether the agent acts at this time step.
            if random.random() >= self.lam:
                continue

            if self.withdraw_old:
                market.withdraw_all(agent_id=self.agent_id)

            side = random.choice([BUY, SELL])

            # --------------------------------------------------
            # 1. ORDER QUANTITY - HISTORICAL VOLUME
            # --------------------------------------------------
            quantity = self.get_order_quantity(
                asset_id,
                fallback=self.mean_volume,
                max_quantity=self.q_max,
            )

            # --------------------------------------------------
            # 2. REFERENCE PRICE - CURRENT ORDER BOOK
            # --------------------------------------------------
            best_bid = market.order_book.get_best_bid()
            best_ask = market.order_book.get_best_ask()

            best_bid = (
                float(best_bid) if best_bid is not None and math.isfinite(float(best_bid)) else None
            )
            best_ask = (
                float(best_ask) if best_ask is not None and math.isfinite(float(best_ask)) else None
            )

            last_price = float(market.last_traded_price)

            if not math.isfinite(last_price) or last_price <= 0:
                continue

            if side == BUY:
                reference = best_ask if best_ask is not None else last_price
            else:
                reference = best_bid if best_bid is not None else last_price

            if not math.isfinite(reference) or reference <= 0:
                continue

            # --------------------------------------------------
            # 3. PRICE DISPERSION - HISTORICAL VOLATILITY
            # --------------------------------------------------
            offset_std = self.get_order_price_std(
                asset_id,
                fallback=float(self.mean_spread),
            )

            # Draw a random price offset around the reference.
            raw_offset = float(
                np.random.normal(
                    loc=0.0,
                    scale=max(0.0, offset_std),
                )
            )

            # --------------------------------------------------
            # 4. LIMIT EXTREME PRICE DEVIATIONS
            # --------------------------------------------------
            max_deviation = float(self.configuration.get("max_price_deviation", 0.02))

            if not math.isfinite(max_deviation) or max_deviation < 0:
                raise ValueError("max_price_deviation must be finite and non-negative")

            max_offset = reference * max_deviation

            offset = max(
                -max_offset,
                min(max_offset, raw_offset),
            )

            price = reference + offset

            if not math.isfinite(price) or price <= 0:
                continue

            # --------------------------------------------------
            # 5. CREATE AND SUBMIT ORDER
            # --------------------------------------------------
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
