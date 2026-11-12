"""Regime-switching trader with optional fundamental-gap price discovery."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from marketsim.agent.agent import Agent
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order

if TYPE_CHECKING:
    from marketsim.market import FXRateProvider
    from marketsim.market.instruments.security import Security


class RegimeTraderAgent(Agent):
    """Cross the spread using shared sentiment or optional fundamental gaps.

    Default configuration retains the original shared-sentiment strategy.
    With fundamental_price_discovery enabled, stocks with a calibrated
    fundamental process use the log gap to choose direction and intensity.
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
            "fundamental_price_discovery": False,
            "fundamental_gap_threshold": 0.002,
            "fundamental_gap_sensitivity": 25.0,
            "fundamental_max_activity_multiplier": 3.0,
            "fundamental_max_size_multiplier": 3.0,
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
        if (
            not math.isfinite(float(cfg["fundamental_gap_threshold"]))
            or cfg["fundamental_gap_threshold"] < 0
            or not math.isfinite(float(cfg["fundamental_gap_sensitivity"]))
            or cfg["fundamental_gap_sensitivity"] <= 0
            or not math.isfinite(float(cfg["fundamental_max_activity_multiplier"]))
            or cfg["fundamental_max_activity_multiplier"] < 1
            or not math.isfinite(float(cfg["fundamental_max_size_multiplier"]))
            or cfg["fundamental_max_size_multiplier"] < 1
        ):
            raise ValueError("Invalid fundamental price-discovery parameters")
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
        state = getattr(market, "regime_sentiment", None)
        if state is None:
            state = {"tick": None, "regime": self._draw_regime()}
            market.regime_sentiment = state
        if state["tick"] != current_time:
            if random.random() < self.configuration["regime_switch_probability"]:
                state["regime"] = self._draw_regime()
            state["tick"] = current_time
        return state["regime"]

    def _fundamental_signal(self, market) -> tuple[int, float] | None:
        """Return (direction, normalized gap strength), or None for legacy mode."""
        if not self.configuration["fundamental_price_discovery"]:
            return None
        regime = getattr(self, "fundamental_return_regime", None)
        if regime is None or market.instrument_class != "stock":
            return None
        symbol = str(market.symbol).upper()
        if symbol not in regime.symbols:
            return None
        initial = getattr(self, "fundamental_initial_prices", {}).get(symbol)
        if initial is None:
            return None
        traded = float(market.last_traded_price)
        fundamental = float(regime.value(self.current_day, symbol, float(initial)))
        if not (
            math.isfinite(traded) and traded > 0 and math.isfinite(fundamental) and fundamental > 0
        ):
            return (0, 0.0)
        gap = math.log(fundamental / traded)
        excess = max(0.0, abs(gap) - float(self.configuration["fundamental_gap_threshold"]))
        if excess <= 0:
            return (0, 0.0)
        strength = excess * float(self.configuration["fundamental_gap_sensitivity"])
        return (BUY if gap > 0 else SELL, strength)

    def take_action(self, current_time: int) -> None:
        cfg = self.configuration
        for asset_id, market in self.markets.items():
            if market.instrument_class == "index":
                continue
            signal = self._fundamental_signal(market)
            if signal is None:
                regime = self._market_regime(market, current_time)
                activity = float(cfg["lam"])
                size_multiplier = 1.0
            else:
                regime, strength = signal
                activity = min(
                    1.0,
                    float(cfg["lam"])
                    * min(float(cfg["fundamental_max_activity_multiplier"]), 1.0 + strength),
                )
                size_multiplier = min(float(cfg["fundamental_max_size_multiplier"]), 1.0 + strength)
            if regime == 0 or random.random() >= activity:
                continue
            position = self.position[asset_id]
            max_position = int(cfg["max_position"])
            capacity = max_position - position if regime == BUY else max_position + position
            if capacity <= 0:
                continue
            mean_volume = self.get_order_mean_volume(asset_id, fallback=float(cfg["mean_volume"]))
            quantity = min(max(1, int(np.random.poisson(mean_volume * size_multiplier))), capacity)
            opposing = (
                market.order_book.get_best_ask()
                if regime == BUY
                else market.order_book.get_best_bid()
            )
            if opposing is None or not math.isfinite(float(opposing)):
                continue
            offset = float(cfg["aggression_ticks"]) * float(cfg["tick_size"])
            limit = float(opposing) + offset if regime == BUY else float(opposing) - offset
            if limit <= 0:
                continue
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
                        day=self.current_day,
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
