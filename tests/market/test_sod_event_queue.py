from unittest.mock import MagicMock

from marketsim.market.fourheap import FourHeap
from marketsim.market.fourheap.constants import BUY
from marketsim.market.fourheap.event_queue import EventQueue
from marketsim.market.fourheap.order import Order
from marketsim.market import Price
from marketsim.market.instruments.security import Security


def make_market(tmp_path):
    repository = MagicMock()
    market = Security(
        name="PKO.WA",
        reference_price=Price(109.54),
        market_type="continuous",
        repository=repository,
        output_dir=tmp_path,
    )
    return market


def make_order(market, tick=0):
    return Order(
        price=Price(109.54),
        order_type=BUY,
        quantity=100,
        agent_id=38,
        time=tick,
        asset_id=market.asset_id,
    )


def test_sod_discards_previous_day_events(tmp_path):
    market = make_market(tmp_path)
    market.sod()

    old_queue = market.event_queue
    old_order = make_order(market, tick=0)
    old_queue.schedule_activity(old_order)

    # Confirm the order is scheduled, without consuming it.
    assert old_queue.scheduled_activities[0] == [old_order]

    # Simulate transition to the next trading day.
    market.eod_status = "closed"
    market.sod()

    # SoD must create a new event queue.
    assert market.event_queue is not old_queue

    # The new trading day must not inherit old orders.
    assert market.event_queue.get_activities(0) == []

    # The old queue is independent and remains unchanged.
    assert old_queue.get_activities(0) == [old_order]

    # Reading an event consumes it exactly once.
    assert old_queue.get_activities(0) == []


def test_sod_reinitializes_order_book_and_event_queue(tmp_path):
    market = make_market(tmp_path)
    market.sod()

    old_book = market.order_book
    old_queue = market.event_queue

    market.event_queue.schedule_activity(make_order(market, tick=10))

    market.eod_status = "closed"
    market.sod()

    assert market.order_book is not old_book
    assert isinstance(market.order_book, FourHeap)

    assert market.event_queue is not old_queue
    assert isinstance(market.event_queue, EventQueue)
    assert market.event_queue.get_activities(10) == []


def test_order_book_initialized_on_security_creation(tmp_path):
    market = make_market(tmp_path)

    assert isinstance(market.order_book, FourHeap)
    assert market.order_book.market_id == market.asset_id
