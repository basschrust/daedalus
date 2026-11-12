from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.agent.helpers.fundamental_signal import fundamental_reference
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
            "future_lam_multiplier": 0.25,
            "future_mean_contracts": 1.5,
            "future_max_order_contracts": 5,
            "future_max_position_contracts": 20,
            "future_max_daily_contracts": 10,
            "volume_activity_sensitivity": 0.40,
            "volume_activity_min_multiplier": 0.35,
            "volume_activity_max_multiplier": 2.5,
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

        self.q_max = self.configuration["q_max"]
        self.lam = self.configuration["lam"]
        self.mean_volume = self.configuration["mean_volume"]
        self.mean_spread = self.configuration["mean_spread"]
        self.withdraw_old = self.configuration["withdraw_old"]
        self.validity_period = self.configuration["validity_period"]
        self.future_daily_volume: dict[int, int] = {}

    def get_id(self) -> int:
        return self.agent_id

    def estimate_fundamental(self, current_time: int) -> Price:
        raise  # should not be used for noise agent

    def take_action(self, current_time: int) -> None:
        for asset_id, market in self.markets.items():
            if market.instrument_class == "index":
                continue

            orders = []
            activity = self.lam
            if market.instrument_class == "future":
                activity *= float(self.configuration["future_lam_multiplier"])
            # The shared daily volume regime changes participation as well
            # as order size. Order-size scaling remains in Agent.get_order_mean_volume.
            regime = getattr(self, "volume_regime", None)
            if regime is not None and market.instrument_class == "stock":
                factor = regime.factor(self.current_day, asset_id)
                sensitivity = float(self.configuration["volume_activity_sensitivity"])
                lower = float(self.configuration["volume_activity_min_multiplier"])
                upper = float(self.configuration["volume_activity_max_multiplier"])
                if not math.isfinite(sensitivity) or sensitivity < 0:
                    raise ValueError("volume_activity_sensitivity must be nonnegative")
                if not (0 < lower <= upper and math.isfinite(upper)):
                    raise ValueError("Invalid volume activity multiplier bounds")
                activity *= min(upper, max(lower, factor**sensitivity))
            activity = min(1.0, max(0.0, activity))
            if random.random() >= activity:
                continue

            if self.withdraw_old:
                market.withdraw_all(agent_id=self.agent_id)

            side = random.choice([BUY, SELL])

            if market.instrument_class == "future":
                mean_contracts = float(self.configuration["future_mean_contracts"])
                max_order = int(self.configuration["future_max_order_contracts"])
                quantity = min(int(np.random.poisson(mean_contracts)), max_order)
                position = int(self.futures_position.get(asset_id, 0))
                max_position = int(self.configuration["future_max_position_contracts"])
                if side == BUY:
                    quantity = min(quantity, max(0, max_position - position))
                else:
                    quantity = min(quantity, max(0, max_position + position))
                daily_limit = int(self.configuration["future_max_daily_contracts"])
                used = self.future_daily_volume.get(asset_id, 0)
                quantity = min(quantity, max(0, daily_limit - used))
            else:
                quantity = self.get_order_quantity(
                    asset_id,
                    fallback=self.mean_volume,
                    max_quantity=self.q_max,
                )

            if quantity <= 0:
                continue

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

            if market.instrument_class == "future":
                reference = float(market.calculate_theoretical_value())
            else:
                reference = fundamental_reference(self, market, reference)

            if not math.isfinite(reference) or reference <= 0:
                continue

            offset_std = self.get_order_price_std(
                asset_id,
                fallback=float(self.mean_spread),
            )
            raw_offset = float(np.random.normal(loc=0.0, scale=max(0.0, offset_std)))
            max_deviation = float(self.configuration.get("max_price_deviation", 0.02))
            if not math.isfinite(max_deviation) or max_deviation < 0:
                raise ValueError("max_price_deviation must be finite and non-negative")
            max_offset = reference * max_deviation
            offset = max(-max_offset, min(max_offset, raw_offset))
            price = reference + offset
            if not math.isfinite(price) or price <= 0:
                continue

            order = Order(
                price=Price(price),
                quantity=quantity,
                agent_id=self.agent_id,
                day=self.current_day,
                time=current_time,
                order_type=side,
                asset_id=market.asset_id,
                valid_until=current_time + self.validity_period,
            )
            orders.append(order)
            market.add_orders(orders)
            if market.instrument_class == "future":
                self.future_daily_volume[asset_id] = (
                    self.future_daily_volume.get(asset_id, 0) + quantity
                )

    def __str__(self) -> str:
        return f"Noise_{self.agent_id}"

    def get_pos_value(self) -> Price:
        return self.cash + sum(
            [
                market.last_traded_price * self.position[asset_id]
                for asset_id, market in self.markets.items()
            ]
        )

    def sod(self):
        super().sod()
        self.future_daily_volume.clear()
