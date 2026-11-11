"""Historical-volume-based limits, tracked independently for every traded asset."""

from __future__ import annotations

import math

import pandas as pd


def historical_volume(market):
    history = getattr(market, "historical_ohlcv", None)
    if not isinstance(history, pd.DataFrame) or history.empty:
        return None
    if "volume" not in history:
        raise ValueError(f"{market.name}: historical volume column missing")
    values = pd.to_numeric(history["volume"], errors="coerce")
    if (
        values.isna().any()
        or (values < 0).any()
        or not all(math.isfinite(float(x)) for x in values)
    ):
        raise ValueError(f"{market.name}: invalid historical volume")
    return float(values.mean())


class InstrumentLimitMixin:
    def initialize_instrument_limits(self, markets, config):
        self.q_max_by_asset = {}
        self.daily_volume_limit_by_asset = {}
        self.daily_executed_volume_by_asset = {}
        for asset_id, market in markets.items():
            if not getattr(market, "is_tradable", True):
                continue
            kind = market.instrument_class
            history = historical_volume(market)
            # q_max is only a backward-compatible global ceiling. The effective
            # inventory limit is always stored per instrument in q_max_by_asset.
            # Overrides accept market.name, market.symbol or string asset_id.
            overrides = config.get("q_max_by_asset", {})
            if not isinstance(overrides, dict):
                raise ValueError("q_max_by_asset must be a mapping")
            candidates = (
                str(asset_id),
                getattr(market, "symbol", None),
                getattr(market, "name", None),
            )
            override = next(
                (overrides[key] for key in candidates if key is not None and key in overrides), None
            )
            max_inventory = int(
                override
                if override is not None
                else config.get(f"{kind}_q_max", config.get("q_max", 1000))
            )
            inventory_fraction = float(
                config.get(f"{kind}_inventory_fraction", config.get("inventory_fraction", 0.10))
            )
            daily_fraction = float(
                config.get(
                    f"{kind}_daily_volume_fraction", config.get("daily_volume_fraction", 0.15)
                )
            )
            if max_inventory < 1 or not 0 < inventory_fraction <= 1 or not 0 < daily_fraction <= 1:
                raise ValueError("Invalid market-maker inventory or volume limit configuration")
            if history is None:
                inventory = max_inventory
                daily = int(config.get(f"{kind}_daily_volume_limit", max_inventory))
            elif history == 0:
                inventory = daily = 0
            else:
                # Explicit instrument overrides are hard limits, not merely caps
                # on the historical-volume estimate.
                inventory = (
                    max_inventory
                    if override is not None
                    else min(max_inventory, max(1, int(history * inventory_fraction)))
                )
                daily = max(2 * int(config.get("K", 3)), int(history * daily_fraction))
            self.q_max_by_asset[asset_id] = inventory
            self.daily_volume_limit_by_asset[asset_id] = daily
            self.daily_executed_volume_by_asset[asset_id] = 0

    def limited_quote_sizes(self, asset_id, desired, levels):
        """Conservative capacity: every level on either side could execute."""
        limit = self.q_max_by_asset.get(asset_id, 0)
        if levels <= 0 or limit <= 0:
            return 0, 0
        position = (
            self.futures_position.get(asset_id, 0)
            if asset_id in self.futures_position
            else self.position.get(asset_id, 0)
        )
        available = max(
            0,
            self.daily_volume_limit_by_asset[asset_id]
            - self.daily_executed_volume_by_asset[asset_id],
        )
        # Buy and sell sides share the same remaining daily turnover allowance.
        side_budget = available // 2
        buy = min(int(desired), max(0, (limit - position) // levels), side_budget // levels)
        sell = min(int(desired), max(0, (limit + position) // levels), side_budget // levels)
        return max(0, buy), max(0, sell)

    def record_trade(self, matched_order):
        super().record_trade(matched_order)
        asset_id = matched_order.order.asset_id
        if asset_id in self.daily_executed_volume_by_asset:
            self.daily_executed_volume_by_asset[asset_id] += int(matched_order.order.quantity)

    def record_futures_trade(self, matched_order):
        super().record_futures_trade(matched_order)
        asset_id = matched_order.order.asset_id
        if asset_id in self.daily_executed_volume_by_asset:
            self.daily_executed_volume_by_asset[asset_id] += int(matched_order.order.quantity)

    def sod(self):
        super().sod()
        for asset_id in self.daily_executed_volume_by_asset:
            self.daily_executed_volume_by_asset[asset_id] = 0
