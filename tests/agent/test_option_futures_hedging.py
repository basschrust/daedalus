"""Index-option futures hedging and outstanding hedge-order reconciliation."""

from collections import defaultdict
from decimal import Decimal
from types import SimpleNamespace

from marketsim.agent.option_market_maker_zoh import OptionMMZOHAgent
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order
from marketsim.simulator import Simulator


class Market:
    """Small market double exposing the real agent's queue/book interface."""

    def __init__(
        self, asset_id, kind, *, underlying=None, multiplier=1, expiration=40, tradable=True
    ):
        self.asset_id = asset_id
        self.instrument_class = kind
        self.underlying = underlying
        self.multiplier = multiplier
        self.expiration = expiration
        self.is_tradable = tradable
        self.status = "active"
        self.last_traded_price = Decimal("4000")
        self.short_name = f"ASSET{asset_id}"
        self.current_day = 0
        self.order_book = SimpleNamespace(
            buy_unmatched=SimpleNamespace(order_dict={}),
            sell_unmatched=SimpleNamespace(order_dict={}),
        )

        def remove(order_id):
            for queue in (self.order_book.buy_unmatched, self.order_book.sell_unmatched):
                if order_id in queue.order_dict:
                    del queue.order_dict[order_id]
                    self.cancellations += 1

        self.order_book.remove = remove
        self.event_queue = SimpleNamespace(scheduled_activities=defaultdict(list))
        self.cancellations = 0
        self.new_orders = []

    def add_orders(self, orders):
        self.new_orders.extend(orders)
        for order in orders:
            self.event_queue.scheduled_activities[order.time].append(order)

    def withdraw_all(self, agent_id):
        self.cancellations += 1
        for queue in (self.order_book.buy_unmatched, self.order_book.sell_unmatched):
            for order_id, order in list(queue.order_dict.items()):
                if order.agent_id == agent_id:
                    del queue.order_dict[order_id]

    def rest_orders(self):
        """Move scheduled orders into the unmatched book like a previous tick."""
        for tick, orders in list(self.event_queue.scheduled_activities.items()):
            for order in orders:
                queue = (
                    self.order_book.buy_unmatched
                    if order.order_type == BUY
                    else self.order_book.sell_unmatched
                )
                queue.order_dict[order.order_id] = order
            del self.event_queue.scheduled_activities[tick]


def hedger(index_delta=100, *, near_expiration=10, day=0):
    index = Market(1, "index", tradable=False)
    near = Market(2, "future", underlying=index, multiplier=20, expiration=near_expiration)
    far = Market(3, "future", underlying=index, multiplier=20, expiration=30)
    agent = object.__new__(OptionMMZOHAgent)
    agent.agent_id = 45
    agent.current_day = day
    agent.underlying_markets = {index.asset_id: index}
    agent.hedge_futures = {near.asset_id: near, far.asset_id: far}
    agent.futures_position = {near.asset_id: 0, far.asset_id: 0}
    agent.greeks_agg = {index.asset_id: {"delta": index_delta}}
    warnings = []
    agent.logger = SimpleNamespace(info=lambda *args: None, warning=warnings.append)
    return agent, index, near, far, warnings


def pending(market):
    return [
        order for orders in market.event_queue.scheduled_activities.values() for order in orders
    ]


def test_index_delta_hedged_with_nearest_matching_future():
    agent, _, near, far, _ = hedger(100)
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity, o.asset_id) for o in pending(near)] == [(SELL, 5, 2)]
    assert not pending(far)


def test_no_duplicate_orders_on_repeated_hedging_ticks():
    agent, _, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    agent.hedge(current_time=1)
    assert len(pending(near)) == 1
    near.rest_orders()
    agent.hedge(current_time=2)
    assert not pending(near)
    assert len(near.order_book.sell_unmatched.order_dict) == 1
    assert near.cancellations == 0


def test_partially_pending_hedge_is_only_topped_up():
    agent, _, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    order = pending(near)[0]
    order.quantity = 2
    agent.hedge(current_time=1)
    assert sorted(o.quantity for o in pending(near)) == [2, 3]


def test_existing_futures_positions_reduce_new_hedge():
    agent, _, near, _, _ = hedger(100)
    agent.futures_position[near.asset_id] = -2
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity) for o in pending(near)] == [(SELL, 3)]


def test_filled_order_rebalances_without_duplicate_fill():
    agent, _, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    near.event_queue.scheduled_activities.clear()  # order filled by the exchange
    agent.futures_position[near.asset_id] = -5
    agent.hedge(current_time=2)
    assert not pending(near)


def test_target_reversal_cancels_unfilled_sell_before_buy():
    agent, index, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    agent.greeks_agg[index.asset_id]["delta"] = -100
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity) for o in pending(near)] == [(BUY, 5)]
    assert near.cancellations >= 0


def test_excessive_unfilled_order_is_replaced():
    agent, index, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    agent.greeks_agg[index.asset_id]["delta"] = 40  # need only two contracts
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity) for o in pending(near)] == [(SELL, 2)]
    assert near.cancellations >= 0


def test_opposing_pending_orders_replaced_instead_of_netted():
    agent, _, near, _, _ = hedger(100)
    near.add_orders(
        [
            Order(
                price=near.last_traded_price,
                quantity=8,
                agent_id=agent.agent_id,
                day=0,
                time=1,
                order_type=BUY,
                asset_id=near.asset_id,
            ),
            Order(
                price=near.last_traded_price,
                quantity=3,
                agent_id=agent.agent_id,
                day=0,
                time=1,
                order_type=SELL,
                asset_id=near.asset_id,
            ),
        ]
    )
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity) for o in pending(near)] == [(SELL, 5)]
    assert near.cancellations >= 0


def test_partially_filled_resting_remainder_is_actually_cancelled():
    agent, index, near, _, _ = hedger(100)
    agent.hedge(current_time=1)
    near.rest_orders()
    # The exchange represents a partial remainder with a fresh order ID.
    # Its absence from agent_id_map must not allow it to survive reversal.
    old = list(near.order_book.sell_unmatched.order_dict.values())[0]
    remainder = Order(
        price=old.price,
        quantity=3,
        agent_id=agent.agent_id,
        day=0,
        time=1,
        order_type=SELL,
        asset_id=near.asset_id,
        parent_id=old.order_id,
    )
    near.order_book.sell_unmatched.order_dict.clear()
    near.order_book.sell_unmatched.order_dict[remainder.order_id] = remainder
    agent.greeks_agg[index.asset_id]["delta"] = -100
    agent.hedge(current_time=2)
    assert not near.order_book.sell_unmatched.order_dict
    assert [(o.order_type, o.quantity) for o in pending(near)] == [(BUY, 5)]
    assert near.cancellations == 1


def test_nearest_expiry_rolls_and_accounts_for_positions_on_other_series():
    agent, _, near, far, _ = hedger(100, near_expiration=5, day=5)
    agent.futures_position[near.asset_id] = -2
    agent.hedge(current_time=1)
    assert not pending(near)
    assert [(o.order_type, o.quantity) for o in pending(far)] == [(SELL, 3)]


def test_older_series_open_orders_cancelled_when_hedging_moves_to_new_series():
    agent, _, near, far, _ = hedger(100, near_expiration=5, day=5)
    near.add_orders(
        [
            Order(
                price=near.last_traded_price,
                quantity=4,
                agent_id=agent.agent_id,
                day=0,
                time=1,
                order_type=SELL,
                asset_id=near.asset_id,
            )
        ]
    )
    agent.hedge(current_time=1)
    assert not pending(near)
    assert [(o.order_type, o.quantity) for o in pending(far)] == [(SELL, 5)]


def test_missing_future_warns_instead_of_trading_index_directly():
    agent, _, near, far, warnings = hedger(100)
    agent.hedge_futures = {}
    agent.hedge(current_time=1)
    assert warnings and "No active future" in warnings[0]
    assert not pending(near) and not pending(far)


def test_stock_option_hedging_uses_the_same_pending_order_reconciliation():
    agent, _, _, _, _ = hedger(0)
    stock = Market(6, "stock")
    agent.underlying_markets = {stock.asset_id: stock}
    agent.greeks_agg = {stock.asset_id: {"delta": 3.8}}
    agent.hedge(current_time=1)
    agent.hedge(current_time=1)
    assert [(o.order_type, o.quantity) for o in pending(stock)] == [(SELL, 3)]


def test_simulator_passes_matching_futures_into_option_market_maker(monkeypatch):
    from marketsim.simulator import simulator as simulator_module

    index = Market(10, "index", tradable=False)
    unrelated = Market(11, "index", tradable=False)
    option = Market(12, "option", underlying=index)
    future = Market(13, "future", underlying=index)
    other_future = Market(14, "future", underlying=unrelated)
    sim = object.__new__(Simulator)
    sim.markets = {market.asset_id: market for market in (index, option, future, other_future)}
    sim.repository = object()
    sim.output_dir = "tmp"
    seen = []
    monkeypatch.setattr(simulator_module, "OptionMMZOHAgent", lambda **kw: seen.append(kw) or kw)
    monkeypatch.setattr(Simulator, "add_agents", lambda self, agents: None)
    sim.add_agent_group(
        agent_group={"agent_class": "OptionMMZOHAgent", "number": 1, "config": {}},
        markets=[option],
        group_name="OptionMM",
    )
    assert {m.asset_id for m in seen[0]["markets"]} == {option.asset_id, future.asset_id}
