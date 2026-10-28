from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING
import math

from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.market.security import Security


class MMZOHAgent(Agent):
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
        self.q_max = final_configuration["q_max"]

        # calculating the volume for each asset by fraction of historical data:
        self.volume_fraction = Decimal(
            str(final_configuration.get("volume_fraction", "0.10"))
        )

        if not 0 < self.volume_fraction <= 1:
            raise ValueError("volume_fraction must be in (0, 1]")

        self.volume = final_configuration.get("volume", 5)

        if sim_time <= 0:
            raise ValueError("sim_time must be positive")

        self.volume_by_asset = {}

        for asset_id, market in self.markets.items():
            history = getattr(market, "historical_ohlcv", None)

            if history is None or history.empty:
                # Preserve existing fixed-volume behavior.
                self.volume_by_asset[asset_id] = int(self.volume)
                continue

            volumes = history["volume"]

            if (
                    volumes.isna().any()
                    or not (volumes >= 0).all()
                    or not (volumes < float("inf")).all()
            ):
                raise ValueError(
                    f"Invalid historical volume for {market.name}"
                )

            daily_volume = float(volumes.mean())

            if daily_volume <= 0:
                raise ValueError(
                    f"Invalid historical volume for {market.name}"
                )

            daily_volume = float(history["volume"].mean())

            if not 0 < daily_volume < float("inf"):
                raise ValueError(
                    f"Invalid historical volume for {market.name}"
                )

            rebalance_count = math.ceil(
                sim_time / self.rebalance_period
            )

            quantity = round(
                daily_volume
                * float(self.volume_fraction)
                / (rebalance_count * 2 * self.K)
            )

            self.volume_by_asset[asset_id] = max(1, quantity)

            self.logger.info(
                f"MM volume calibration: {market.name}, "
                f"average_daily_volume={daily_volume:.0f}, "
                f"fraction={self.volume_fraction}, "
                f"order_quantity={self.volume_by_asset[asset_id]}"
            )


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
            if market.instrument_class == "index":
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
                st = max(estimate + HALF * self.omega, best_bid)
                bt = min(estimate - HALF * self.omega, best_ask)
                self.logger.info(f"Setting basic spread to: {bt}, {st}")
                # volumes for this market:
                buy_volume = self.volume_by_asset[asset_id]
                sell_volume = self.volume_by_asset[asset_id]
                # TODO: adjust the spread for position rebalancing
                if abs(self.position[asset_id]) > self.q_max / 2:
                    if self.position[asset_id] > 0:
                        # the MM position is very long - needs to sell, so move the prices up
                        st = st + HALF * self.omega
                        bt = bt + HALF * self.omega
                        if self.position[asset_id] > 3 / 4 * self.q_max:
                            # if getting close to max we also limit the volume of buy orders:
                            buy_volume = int(buy_volume / 2)
                            if self.position[asset_id] > self.q_max:
                                # if we exceeded the q_max then volume should be only symbolic
                                buy_volume = 1
                    else:
                        # the MM position is very short - has to buy more, lower the prices
                        st = st - HALF * self.omega
                        bt = bt - HALF * self.omega
                        if self.position[asset_id] < -3 / 4 * self.q_max:
                            sell_volume = int(sell_volume / 2)
                            if self.position[asset_id] < -self.q_max:
                                sell_volume = 1

                self.logger.info(f"Basic spread adjusted to: {bt}, {st}")

                for k in range(self.K):
                    price_bid = Price(bt - (k + 1) * self.xi)
                    if price_bid > 0:
                        orders.append(
                            Order(
                                price=price_bid,
                                quantity=buy_volume,  # 7,#1, # we ćould raise the quantity in each ladder step...
                                agent_id=self.agent_id,
                                time=current_time,
                                order_type=BUY,
                                asset_id=market.asset_id,
                            )
                        )
                        orders.append(
                            Order(
                                price=Price(st + (k + 1) * self.xi),
                                quantity=sell_volume,  # 7,#1,
                                agent_id=self.agent_id,
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
