from marketsim.market.fourheap.constants import BUY
from marketsim.market.fourheap.event_queue import EventQueue
from marketsim.market.fourheap.order import Order
from marketsim.market.price import Price


def test_event_queue_consumes_activities():
    queue = EventQueue(rand_seed=67)
    order = Order(
        price=Price(109.54),
        order_type=BUY,
        quantity=100,
        agent_id=38,
        time=10,
        asset_id=0,
    )

    queue.schedule_activity(order)

    assert queue.get_activities(10) == [order]
    assert queue.get_activities(10) == []
    assert 10 not in queue.scheduled_activities
