from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from marketsim.agent.agent import Agent
from marketsim.agent.mm_instrument_limits import InstrumentLimitMixin
from marketsim.market import FXRateProvider, Price
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.market.instruments.security import Security


class MMZOHAgent(InstrumentLimitMixin, Agent):
    ### Market Maker Zero Order Hold Agent -
    # A MM which just takes into account last traded price and sets new order ladder
    # symmetrically on both sides of this last traded price in each rebalance period
    ###
    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        configuration: dict | None = None,
        group: str = "MMZOH",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
        sim_time: int = 40,
    ) -> None:
        default_configuration = {
            "xi": 0.1,
            "K": 3,
            "omega": 0.1,
            "rebalance_period": 7,
            "volume": 5,
            "q_max": 1000,
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

        ## TODO: MM parameters - should be defined per market
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

        # self.volume = final_configuration["volume"]

        # calculating the volume for each asset by fraction of historical data:
        self.volume_fraction = Decimal(str(final_configuration.get("volume_fraction", "0.10")))

        if not 0 < self.volume_fraction <= 1:
            raise ValueError("volume_fraction must be in (0, 1]")

        self.volume = final_configuration.get("volume", 5)

        if sim_time <= 0:
            raise ValueError("sim_time must be positive")

        self.volume_by_asset = {}

        for asset_id, market in self.markets.items():
            # Indices are not tradable and do not need volume calibration.
            if not getattr(market, "is_tradable", True):
                continue

            history = getattr(market, "historical_ohlcv", None)

            # No usable historical DataFrame: fixed-volume fallback.
            if not isinstance(history, pd.DataFrame) or history.empty:
                self.volume_by_asset[asset_id] = (
                    1 if market.instrument_class == "future" else int(self.volume)
                )
                continue

            if "volume" not in history.columns:
                raise ValueError(
                    f"Invalid historical volume for {market.name}: missing volume column"
                )

            volumes = pd.to_numeric(history["volume"], errors="coerce")
            if (
                volumes.isna().any()
                or not (volumes >= 0).all()
                or not np.isfinite(volumes.to_numpy(dtype=float)).all()
            ):
                raise ValueError(f"Invalid historical volume for {market.name}")
            daily_volume = float(volumes.mean())
            if not math.isfinite(daily_volume):
                raise ValueError(f"Invalid historical volume for {market.name}")
            if daily_volume == 0:
                self.volume_by_asset[asset_id] = 0
                self.logger.warning(
                    f"MM volume calibration: {market.name} has zero "
                    "historical volume; no liquidity will be quoted"
                )
                continue

            rebalance_count = math.ceil(sim_time / self.rebalance_period)

            quantity = round(
                daily_volume * float(self.volume_fraction) / (rebalance_count * 2 * self.K)
            )

            self.volume_by_asset[asset_id] = max(1, quantity)

            self.logger.info(
                f"MM volume calibration: {market.name}, "
                f"average_daily_volume={daily_volume:.0f}, "
                f"fraction={self.volume_fraction}, "
                f"order_quantity={self.volume_by_asset[asset_id]}"
            )

        self.initialize_instrument_limits(self.markets, final_configuration)

    def get_id(self) -> int:
        return self.agent_id

    def is_market_maker(self) -> bool:
        return True

    def should_rebalance(self, current_time: int) -> bool:
        # TODO: add market dimension?
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
            # TODO: per market!
            for asset_id, market in self.markets.items():
                self.cum_volume += market.traded_prices.get(current_time - 1, {}).get("Volume", 0)
            if self.cum_volume >= self.rebalance_volume:
                self.last_rebalance_time = current_time
                self.cum_volume = 0
                return True
        return False

    def take_action(self, current_time: int):
        # TODO: the current_time argument should be replaced to take the self.current_time
        for asset_id, market in self.markets.items():
            if not getattr(market, "is_tradable", True):
                continue
            # Do not quote markets with zero historical volume.
            if self.volume_by_asset.get(asset_id, 0) <= 0:
                continue
            orders = []
            # add orders only in rebalance periods:
            if self.should_rebalance(current_time):
                # AK - clear previous orders (should we?)
                self.logger.info(
                    "Withdrawing previous orders ()"
                )  # how to check number of orders of this agent?
                market.withdraw_all(agent_id=self.agent_id)
                # AK - don't withdraw, but also don't blindly add new orders - just ensure they are balanced
                # that's basically the same to just withdraw all and create new, the problem might be with timing
                # - we could loose the slot in the queue of waiting orders

                # Get the best bid and best ask
                best_ask = market.order_book.get_best_ask()
                best_bid = market.order_book.get_best_bid()

                self.logger.info(f"Best bid {best_bid}, best ask: {best_ask}")

                estimate = market.last_traded_price
                self.logger.info(f"Last traded price: {estimate}")
                HALF = Decimal("0.5")
                if market.instrument_class == "future":
                    underlying = market.underlying
                    if underlying.instrument_class == "index":
                        estimate = market.calculate_theoretical_value()
                    else:
                        estimate = Price(underlying.last_traded_price)
                    # Anchor futures quotes to their theoretical price.
                    st = estimate + HALF * self.omega
                    bt = estimate - HALF * self.omega
                else:
                    estimate = market.last_traded_price
                    st = max(estimate + HALF * self.omega, best_bid)
                    bt = min(estimate - HALF * self.omega, best_ask)
                self.logger.info(f"Setting basic spread to: {bt}, {st}")
                buy_volume, sell_volume = self.limited_quote_sizes(
                    asset_id, self.volume_by_asset[asset_id], self.K
                )
                self.logger.info(f"Basic spread adjusted to: {bt}, {st}")

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
                                asset_id=market.asset_id,
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
                                asset_id=market.asset_id,
                            )
                        )

            # return orders
            market.add_orders(orders)

    def get_pos_value(self) -> float:
        return 0

    def __str__(self):
        return f"MM_ZOH{self.agent_id}"
