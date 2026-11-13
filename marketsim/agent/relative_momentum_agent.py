"""Long-only, cash-funded relative momentum portfolio agent.

Signals use only completed daily bars. The strategy is exchange independent;
this first execution adapter supports cash equities in the agent's base currency.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from marketsim.agent.agent import Agent
from marketsim.agent.strategies.relative_momentum import (
    MomentumSettings,
    equal_weight_targets,
    rank_momentum,
)
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order


class RelativeMomentumAgent(Agent):
    def __init__(
        self,
        *,
        markets,
        repository,
        configuration=None,
        group="RELATIVE_MOMENTUM",
        output_dir: Path | str = "tmp",
        fx_rate_provider=None,
    ):
        config = {
            "lookback_sessions": 252,
            "turnover_sessions": 30,
            "min_median_turnover": 250_000.0,
            "portfolio_size": 10,
            "rebalance_sessions": 20,
            "rebalance_offset": 0,
            "aggression": 0.003,
            "order_validity_ticks": 3,
            **(configuration or {}),
        }
        super().__init__(
            markets=markets,
            repository=repository,
            configuration=config,
            group=group,
            output_dir=output_dir,
            fx_rate_provider=fx_rate_provider,
        )
        self.settings = MomentumSettings(
            lookback_sessions=int(config["lookback_sessions"]),
            turnover_sessions=int(config["turnover_sessions"]),
            min_median_turnover=float(config["min_median_turnover"]),
            portfolio_size=int(config["portfolio_size"]),
        )
        self.rebalance_sessions = int(config["rebalance_sessions"])
        if self.rebalance_sessions < 1:
            raise ValueError("rebalance_sessions must be positive")
        # dispersing offset across all agents in the group:
        self.rebalance_offset = (
            int(config["rebalance_offset"]) + self.agent_id
        ) % self.rebalance_sessions
        if not 0 <= self.rebalance_offset < self.rebalance_sessions:
            raise ValueError("rebalance_offset must be between 0 and rebalance_sessions - 1")
        self._simulated_bars: dict[int, list[dict]] = {}
        self._last_rebalance_day: int | None = None

    def get_id(self) -> int:
        return self.agent_id

    def get_pos_value(self) -> float:
        return float(
            sum(
                self.position.get(asset_id, 0) * float(market.last_traded_price)
                for asset_id, market in self.markets.items()
                if market.instrument_class == "stock"
            )
        )

    def _histories(self) -> tuple[dict[str, pd.DataFrame], dict[str, object], date]:
        histories = {}
        symbols = {}
        as_of = None
        for asset_id, market in self.markets.items():
            if market.instrument_class != "stock" or not market.is_tradable:
                continue
            if market.currency != self.base_currency:
                continue  # Cross-currency portfolio sizing needs an FX-aware adapter.
            symbol = getattr(market, "symbol", None)
            if not symbol or symbol in symbols:
                continue
            raw = market.historical_ohlcv
            if raw is None or raw.empty:
                continue
            history = raw.copy()
            if "date" not in history.columns:
                history = history.reset_index().rename(
                    columns={history.index.name or "index": "date"}
                )
            if "date" not in history.columns:
                raise ValueError(f"{symbol}: history has no date")
            history["date"] = pd.to_datetime(history["date"])
            sim_dates = market.simulation_dates
            if not sim_dates:
                continue
            first = sim_dates[0]
            history = history.loc[history["date"].dt.date < first]
            extra = self._simulated_bars.get(asset_id, [])
            if extra:
                history = pd.concat([history, pd.DataFrame(extra)], ignore_index=True)
            histories[symbol] = history
            symbols[symbol] = market
            cutoff = sim_dates[self.current_day] - timedelta(days=1)
            if as_of is None:
                as_of = cutoff
        return histories, symbols, as_of or date.min

    def take_action(self, current_time: int) -> None:
        if current_time != 0:
            return
        if (self.current_day - self.rebalance_offset) % self.rebalance_sessions != 0:
            return
        if self._last_rebalance_day == self.current_day:
            return
        self._last_rebalance_day = self.current_day
        histories, symbols, as_of = self._histories()
        if not histories:
            return
        ranking = rank_momentum(histories, as_of=as_of, settings=self.settings)
        weights = equal_weight_targets(ranking)
        if not weights:
            return  # No qualified instruments: do not liquidate on missing data.

        # Reserve cash conservatively. Sell orders are submitted first, but their
        # proceeds are NOT counted until actually matched.
        cash = max(Decimal(0), self.get_cash_balance(currency=self.base_currency))
        equity = cash + sum(
            Decimal(self.position.get(m.asset_id, 0)) * Decimal(str(m.last_traded_price))
            for m in symbols.values()
        )
        intents = []
        for symbol, market in symbols.items():
            price = Decimal(str(market.last_traded_price))
            if price <= 0:
                continue
            target = int(equity * Decimal(str(weights.get(symbol, 0))) / price)
            delta = target - self.position.get(market.asset_id, 0)
            if delta:
                intents.append((market, delta))
        intents.sort(key=lambda item: item[1])
        for market, delta in intents:
            side = BUY if delta > 0 else SELL
            opposing = (
                market.order_book.get_best_ask()
                if side == BUY
                else market.order_book.get_best_bid()
            )
            last_price = float(market.last_traded_price)
            if not math.isfinite(last_price) or last_price <= 0:
                self.logger.warning(
                    f"Relative momentum: invalid last price for {market.symbol}: {last_price}"
                )
                continue
            reference = float(opposing) if opposing is not None else last_price
            # An empty order book can return +inf or -inf.
            if not math.isfinite(reference) or reference <= 0:
                reference = last_price
            aggression = float(self.configuration["aggression"])

            limit_value = reference * (1 + aggression if side == BUY else 1 - aggression)
            if not math.isfinite(limit_value) or limit_value <= 0:
                self.logger.warning(
                    f"Relative momentum: invalid order limit for {market.symbol}: "
                    f"reference={reference}, aggression={aggression}"
                )
                continue
            limit = Price(max(0.01, limit_value))

            if side == BUY:
                quantity = min(delta, int(cash / Decimal(str(limit))))
                cash -= Decimal(quantity) * Decimal(str(limit))
            else:
                quantity = min(-delta, max(0, self.position.get(market.asset_id, 0)))
            if quantity <= 0:
                continue
            market.withdraw_all(agent_id=self.agent_id)
            market.add_orders(
                [
                    Order(
                        price=limit,
                        quantity=quantity,
                        agent_id=self.agent_id,
                        day=self.current_day,
                        time=current_time,
                        order_type=side,
                        asset_id=market.asset_id,
                        valid_until=current_time + int(self.configuration["order_validity_ticks"]),
                    )
                ]
            )

    def eod(self) -> None:
        # Market.eod() runs before Agent.eod() in Simulator.eod().
        for asset_id, market in self.markets.items():
            if market.instrument_class != "stock" or not market.simulation_dates:
                continue
            df = getattr(market, "traded_prices_df", None)
            if df is None or df.empty:
                continue
            day = self.current_day
            if day >= len(market.simulation_dates):
                continue
            row = {
                "date": market.simulation_dates[day],
                "open": float(df.iloc[0]["open"]),
                "high": float(df["high"].max()),
                "low": float(df["low"].min()),
                "close": float(df.loc[df["time_tick"].idxmax(), "close"]),
                "volume": float(df["volume"].sum()),
            }
            bars = self._simulated_bars.setdefault(asset_id, [])
            if not bars or bars[-1]["date"] != row["date"]:
                bars.append(row)
        super().eod()

    def __str__(self):
        return f"RelativeMomentum_{self.agent_id}"
