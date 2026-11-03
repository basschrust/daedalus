from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from marketsim.agent.agent import Agent
from marketsim.agent.helpers.index_arbitrage_portfolio import index_hedge_quantities
from marketsim.market import Price
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order

if TYPE_CHECKING:
    from marketsim.market.instruments.security import Security


class IndexFuturesArbitrageAgent(Agent):
    """Trade futures deviations from their underlying index value.

    Version 1: unhedged basis-convergence strategy.
    No direct modification of market prices or index values.
    """

    def __init__(
        self,
        *,
        markets: list[Security],
        repository,
        configuration: dict | None = None,
        group: str = "ARBITRAGE",
        output_dir: Path | str = "tmp",
    ) -> None:
        defaults = {
            "entry_threshold": 0.003,
            "exit_threshold": 0.0005,
            "transaction_cost": 0.0002,
            "order_quantity": 2,
            "max_position": 20,
            "validity_period": 1,
            "min_time_between_orders": 1,
            "future_max_daily_contracts": 10,
        }
        cfg = defaults | (configuration or {})

        super().__init__(
            markets=markets,
            repository=repository,
            configuration=cfg,
            group=group,
            output_dir=output_dir,
        )

        self.entry_threshold = float(cfg["entry_threshold"])
        self.exit_threshold = float(cfg["exit_threshold"])
        self.transaction_cost = float(cfg["transaction_cost"])
        self.order_quantity = int(cfg["order_quantity"])
        self.max_position = int(cfg["max_position"])
        self.validity_period = int(cfg["validity_period"])
        self.min_time_between_orders = int(cfg["min_time_between_orders"])
        self.max_order_quantity = int(cfg.get("max_order_quantity", self.order_quantity))
        self.volume_aggression = float(cfg.get("volume_aggression", 0.0))

        if self.max_order_quantity < self.order_quantity:
            raise ValueError("max_order_quantity must be >= order_quantity")
        if not math.isfinite(self.volume_aggression) or self.volume_aggression < 0:
            raise ValueError("volume_aggression must be finite and non-negative")

        if not (
            0 <= self.exit_threshold < self.entry_threshold
            and self.transaction_cost >= 0
            and self.order_quantity > 0
            and self.max_position > 0
            and self.validity_period > 0
            and self.min_time_between_orders > 0
        ):
            raise ValueError("Invalid arbitrage configuration")

        self.last_order_time: dict[int, int] = {}

    def get_id(self) -> int:
        return self.agent_id

    def get_pos_value(self) -> float:
        return 0.0

    def take_action(self, current_time: int) -> None:
        # Reconcile the stock portfolio against actual futures fills.
        self._rebalance_all_stock_hedges(
            current_time=current_time,
        )
        for asset_id, future in self.markets.items():
            if future.instrument_class != "future":
                continue

            if future.status != "active":
                continue

            previous = self.last_order_time.get(asset_id)

            if previous is not None and current_time - previous < self.min_time_between_orders:
                continue

            fair_value = float(future.underlying.calculate_value())

            if not math.isfinite(fair_value) or fair_value <= 0:
                continue

            book = future.order_book

            best_bid = book.get_best_bid()
            best_ask = book.get_best_ask()

            bid = float(best_bid) if best_bid is not None else None
            ask = float(best_ask) if best_ask is not None else None

            if bid is not None and (not math.isfinite(bid) or bid <= 0):
                bid = None

            if ask is not None and (not math.isfinite(ask) or ask <= 0):
                ask = None

            position = int(self.futures_position.get(asset_id, 0))

            # Positive basis: futures are expensive.
            # Negative basis: futures are cheap.
            sell_basis = (bid - fair_value) / fair_value if bid is not None else None
            buy_basis = (ask - fair_value) / fair_value if ask is not None else None

            entry = self.entry_threshold + self.transaction_cost
            exit_band = self.exit_threshold

            side = None
            quantity = 0
            limit_price = None
            is_closing = False

            # Close an existing long position when basis converges.
            if position > 0 and bid is not None:
                if sell_basis >= -exit_band:
                    side = SELL
                    quantity = min(self.order_quantity, position)
                    limit_price = bid
                    is_closing = True

            # Close an existing short position when basis converges.
            elif position < 0 and ask is not None:
                if buy_basis <= exit_band:
                    side = BUY
                    quantity = min(self.order_quantity, -position)
                    limit_price = ask
                    is_closing = True

            # Open or extend a short position.
            if side is None and sell_basis is not None:
                if sell_basis > entry and position > -self.max_position:
                    side = SELL
                    quantity = min(
                        self.order_quantity,
                        self.max_position + position,
                    )
                    limit_price = bid

            # Open or extend a long position.
            if side is None and buy_basis is not None:
                if buy_basis < -entry and position < self.max_position:
                    side = BUY
                    quantity = min(
                        self.order_quantity,
                        self.max_position - position,
                    )
                    limit_price = ask

            if side is None or quantity <= 0:
                continue

            basis = sell_basis if side == SELL else buy_basis

            excess_basis = max(
                0.0,
                abs(basis) - (self.entry_threshold + self.transaction_cost),
            )

            scale = 1.0 + self.volume_aggression * (excess_basis / max(self.entry_threshold, 1e-9))

            desired_quantity = min(
                self.max_order_quantity,
                max(self.order_quantity, int(self.order_quantity * scale)),
            )

            # Respect the position limit in both directions.
            capacity = self.max_position - position if side == BUY else self.max_position + position

            if is_closing:
                quantity = min(
                    self.order_quantity,
                    abs(position),
                )
            else:
                quantity = min(
                    desired_quantity,
                    max(0, capacity),
                )

            if quantity <= 0:
                continue

            # Remove outstanding orders before replacing them.
            future.withdraw_all(agent_id=self.agent_id)

            price = Price(limit_price)

            if price <= 0:
                continue

            order = Order(
                price=price,
                quantity=quantity,
                agent_id=self.agent_id,
                time=current_time,
                order_type=side,
                asset_id=asset_id,
                valid_until=current_time + self.validity_period,
            )

            future.add_orders([order])
            self.last_order_time[asset_id] = current_time

    def __str__(self) -> str:
        return f"IndexFuturesArbitrage_{self.agent_id}"

    def get_stock_hedge_deltas(self, future):

        contracts = self.futures_position.get(future.asset_id, 0)

        targets = index_hedge_quantities(
            future,
            contracts,
        )

        return {
            asset_id: target - self.position.get(asset_id, 0)
            for asset_id, target in targets.items()
        }

    def _submit_stock_hedge(
        self,
        *,
        future,
        current_time: int,
    ) -> None:
        """Rebalance stock holdings against executed futures positions.

        Stock orders are submitted to the ordinary order books.
        Actual positions are updated by the existing matching engine.
        """
        deltas = self.get_stock_hedge_deltas(future)

        max_stock_order = int(self.configuration.get("max_stock_order_quantity", 5000))

        for asset_id, delta in deltas.items():
            stock = self.markets.get(asset_id)

            if stock is None or stock.status != "active":
                continue

            # Cancel outstanding hedge orders before calculating
            # a replacement against the latest filled positions.
            stock.withdraw_all(agent_id=self.agent_id)

            if delta == 0:
                continue

            side = BUY if delta > 0 else SELL

            book = stock.order_book

            quote = book.get_best_ask() if side == BUY else book.get_best_bid()

            if quote is None:
                continue

            price = float(quote)

            if not math.isfinite(price) or price <= 0:
                continue

            quantity = min(
                abs(int(delta)),
                max_stock_order,
            )

            if quantity <= 0:
                continue

            order = Order(
                price=Price(price),
                quantity=quantity,
                agent_id=self.agent_id,
                time=current_time,
                order_type=side,
                asset_id=asset_id,
                valid_until=current_time + self.validity_period,
            )

            stock.add_orders([order])

    def _rebalance_all_stock_hedges(
        self,
        *,
        current_time: int,
    ) -> None:
        """Rebalance all stock hedges once per simulation tick."""
        for future in self.markets.values():
            if future.instrument_class != "future":
                continue

            self._submit_stock_hedge(
                future=future,
                current_time=current_time,
            )
