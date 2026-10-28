"""Regime-switching directional trader with a shared, per-market sentiment signal."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order
from marketsim.market import Price

if TYPE_CHECKING:
    from marketsim.market import FXRateProvider
    from marketsim.market.instruments.security import Security


class RegimeTraderAgent(Agent):
    """Submit directional, crossing limit orders following persistent sentiment.

    All agents referencing the same market share a sentiment process via
    ``market.regime_sentiment``. The first acting agent updates it once per tick.
    """

    def __init__(
        self,
        *,
        markets: list[Security],
        repository,
        configuration: dict | None = None,
        group: str = "Regime",
        output_dir: Path | str = "tmp",
        fx_rate_provider: FXRateProvider | None = None,
    ) -> None:
        defaults = {
            "lam": 0.20,
            "mean_volume": 120,
            "regime_switch_probability": 0.002,
            "bullish_probability": 0.45,
            "bearish_probability": 0.45,
            "aggression_ticks": 5,
            "tick_size": 0.01,
            "validity_period": 2,
            "max_position": 5000,
            "initial_cash": {"PLN": 2_000_000},
        }
        cfg = defaults | (configuration or {})
        if not 0 <= cfg["lam"] <= 1:
            raise ValueError("lam must be between 0 and 1")
        if not 0 <= cfg["regime_switch_probability"] <= 1:
            raise ValueError("regime_switch_probability must be between 0 and 1")
        if (
            cfg["bullish_probability"] < 0
            or cfg["bearish_probability"] < 0
            or cfg["bullish_probability"] + cfg["bearish_probability"] > 1
        ):
            raise ValueError("Invalid regime probabilities")
        if cfg["mean_volume"] <= 0 or cfg["tick_size"] <= 0 or cfg["aggression_ticks"] < 0:
            raise ValueError("Order parameters must be positive")
        if cfg["validity_period"] < 1 or cfg["max_position"] < 1:
            raise ValueError("Invalid order lifetime or position limit")
        super().__init__(
            markets=markets,
            repository=repository,
            configuration=cfg,
            group=group,
            output_dir=output_dir,
            fx_rate_provider=fx_rate_provider,
        )

    def get_id(self) -> int:
        return self.agent_id

    def _draw_regime(self) -> int:
        u = random.random()
        if u < self.configuration["bullish_probability"]:
            return BUY
        if (
            u
            < self.configuration["bullish_probability"] + self.configuration["bearish_probability"]
        ):
            return SELL
        return 0

    def _market_regime(self, market, current_time: int) -> int:
        # Store sentiment on the market, not the agent: agents share it.
        # A tick guard ensures switching probability does not scale with agent count.
        state = getattr(market, "regime_sentiment", None)
        if state is None:
            state = {"tick": None, "regime": self._draw_regime()}
            market.regime_sentiment = state
        if state["tick"] != current_time:
            if random.random() < self.configuration["regime_switch_probability"]:
                state["regime"] = self._draw_regime()
            state["tick"] = current_time
        return state["regime"]

    def take_action(self, current_time: int) -> None:
        cfg = self.configuration
        for asset_id, market in self.markets.items():
            if market.instrument_class == "index":
                continue
            regime = self._market_regime(market, current_time)
            if regime == 0 or random.random() >= cfg["lam"]:
                continue
            position = self.position[asset_id]
            max_position = int(cfg["max_position"])
            capacity = max_position - position if regime == BUY else max_position + position
            if capacity <= 0:
                continue

            mean_volume = self.get_order_mean_volume(
                asset_id,
                fallback=float(cfg["mean_volume"]),
            )
            quantity = min(
                max(1, int(np.random.poisson(mean_volume))),
                capacity,
            )

            opposing = (
                market.order_book.get_best_ask()
                if regime == BUY
                else market.order_book.get_best_bid()
            )
            if opposing is None or not math.isfinite(float(opposing)):
                continue
                # No opposite liquidity: do not manufacture an artificial print.
            offset = float(cfg["aggression_ticks"]) * float(cfg["tick_size"])
            limit = float(opposing) + offset if regime == BUY else float(opposing) - offset
            if limit <= 0:
                continue
            self.logger.warning(
                "REGIME ORDER DIAGNOSTIC | "
                f"asset={asset_id} "
                f"day={market.current_day} "
                f"tick={current_time} "
                f"agent={self.agent_id} "
                f"side={regime} "
                f"bid={market.order_book.get_best_bid()} "
                f"ask={market.order_book.get_best_ask()} "
                f"opposing={opposing} "
                f"offset={offset} "
                f"limit={limit} "
                f"last={market.last_traded_price}"
            )
            if regime == BUY and Price(limit) < Price(opposing):
                raise AssertionError("Regime BUY limit below opposing ask")

            if regime == SELL and Price(limit) > Price(opposing):
                raise AssertionError("Regime SELL limit above opposing bid")

            market.add_orders(
                [
                    Order(
                        price=Price(limit),
                        quantity=quantity,
                        agent_id=self.agent_id,
                        time=current_time,
                        order_type=regime,
                        asset_id=asset_id,
                        valid_until=current_time + int(cfg["validity_period"]),
                    )
                ]
            )

    def get_pos_value(self):
        return (
            sum(
                (
                    market.last_traded_price * self.position[asset_id]
                    for asset_id, market in self.markets.items()
                ),
                Price(0),
            )
            + self.get_cash_balance()
        )

    def __str__(self) -> str:
        return f"Regime_{self.agent_id}"
