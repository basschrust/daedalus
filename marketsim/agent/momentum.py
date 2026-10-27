from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from marketsim.agent.agent import Agent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fourheap.order import Order
from marketsim.market import FXRateProvider, Price

if TYPE_CHECKING:
    from database.connectors.duckdb_storage import Repository

    from marketsim.market.security import Security


class MomentumAgent(Agent):
    ### Momentum Agent -
    # Momentum Agent trades using moving average to catch the market trend
    ###
    def __init__(
        self,
        *,
        markets: list[Security],
        repository: Repository,
        configuration: dict | None = None,
        group: str = "MOMENTUM",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        default_configuration = {"period": 7, "lam": 0.5, "q_max": 100, "threshold": 0.01}
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

        self.period = final_configuration["period"]  # the period for trend analyzing
        self.lam = final_configuration["lam"]  # lambda, the activity parameter
        self.q_max = final_configuration["q_max"]  # maximum agent position on each side
        self.threshold = final_configuration[
            "threshold"
        ]  # threshold above which we consider the trend to exist

    def get_id(self) -> int:
        return self.agent_id

    def get_daily_momentum(self, market, lookback: int = 10) -> float:
        historical = market.historical_ohlcv

        if historical is None or historical.empty:
            return 0.0

        history = historical.copy()

        # Historical data should already end at the simulation cutoff.
        # Sort defensively to ensure correct return calculations.
        if "date" in history.columns:
            history = history.sort_values("date")
            if market.simulation_dates:
                first_sim_date = market.simulation_dates[0]
                history = history.loc[pd.to_datetime(history["date"]).dt.date < first_sim_date]
        else:
            history = history.sort_index()
            if market.simulation_dates and isinstance(history.index, pd.DatetimeIndex):
                history = history.loc[history.index.date < market.simulation_dates[0]]

        closes = history["close"].astype(float).tolist()
        closes.extend(market.completed_daily_closes)

        if len(closes) <= lookback:
            return 0.0

        return closes[-1] / closes[-1 - lookback] - 1.0

    def take_action(self, current_time: int) -> None:
        for asset_id, market in self.markets.items():
            if market.instrument_class == "index":
                # TODO: it just should check whether the instrument is tradable
                continue

            if random.random() >= self.lam:
                continue

            current_price = float(market.last_traded_price)
            mode = self.configuration.get("momentum_mode", "intraday")

            if mode == "daily":
                lookback = int(self.configuration.get("daily_lookback", 10))

                momentum = self.get_daily_momentum(
                    market,
                    lookback=lookback,
                )

            else:
                if current_time < self.period:
                    continue

                previous = market.traded_prices.get(current_time - self.period)

                if previous is None:
                    continue

                previous_price = float(previous["close"])

                if previous_price <= 0:
                    continue

                momentum = current_price / previous_price - 1.0

            if momentum > self.threshold:
                side = BUY
            elif momentum < -self.threshold:
                side = SELL
            else:
                continue

            position = self.position[asset_id]

            # Genuine maximum net position, allowing short positions.
            capacity = self.q_max - position if side == BUY else self.q_max + position

            if capacity <= 0:
                continue

            mean_volume = float(self.configuration.get("mean_volume", 300))

            quantity = min(
                max(1, int(np.random.poisson(mean_volume))),
                capacity,
            )

            if side == BUY:
                opposing = market.order_book.get_best_ask()
            else:
                opposing = market.order_book.get_best_bid()

            if opposing is None or not math.isfinite(float(opposing)):
                continue

            aggression = float(self.configuration.get("aggression", 0.003))

            if side == BUY:
                limit = float(opposing) + current_price * aggression
            else:
                limit = max(0.01, float(opposing) - current_price * aggression)

            if not math.isfinite(limit) or Price(limit) <= 0:
                self.logger.warning(
                    f"Momentum invalid price: "
                    f"asset={asset_id}, "
                    f"time={current_time}, "
                    f"side={side}, "
                    f"last={current_price}, "
                    f"opposing={opposing}, "
                    f"aggression={aggression}, "
                    f"limit={limit}"
                )
                continue

            if not math.isfinite(limit) or limit <= 0:
                continue

            order_price = Price(limit)

            # Validate after tick-size rounding.
            if order_price <= 0:
                continue

            market.withdraw_all(agent_id=self.agent_id)

            market.add_orders(
                [
                    Order(
                        price=order_price,
                        quantity=quantity,
                        agent_id=self.agent_id,
                        time=current_time,
                        order_type=side,
                        asset_id=asset_id,
                        valid_until=current_time + 3,
                    )
                ]
            )

    def get_pos_value(self) -> float:
        # TODO: wtf is this function for?
        return 0

    def __str__(self):
        return f"Momentum_{self.agent_id}"
