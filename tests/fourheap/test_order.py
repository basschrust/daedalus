import math

import pytest

from marketsim.fourheap import FourHeap
from marketsim.fourheap.order import Order
from marketsim.market import Price, Security

BUY = 1
SELL = -1


#############   test functions  #############################33


def test_order_creation(buy_order: Order):
    assert buy_order.price == Price(100)
    assert buy_order.order_type == BUY
    assert buy_order.quantity == 10
    assert buy_order.agent_id == 1
    assert buy_order.time == 1
    assert buy_order.asset_id == 1

    assert buy_order.order_id is not None
    assert buy_order.parent_id is None
    assert buy_order.matched_with is None
    assert buy_order.valid_until is None
    assert buy_order.executed_price is None
    assert buy_order.executed_mode is None


def test_order_ids_are_unique(order_factory):
    order1 = order_factory()
    order2 = order_factory()

    assert order1.order_id != order2.order_id


def test_price_must_be_positive(order_factory):
    with pytest.raises(ValueError, match="Price must be greater than 0"):
        order_factory(price=0)

    with pytest.raises(ValueError, match="Price must be greater than 0"):
        order_factory(price=-10)


def test_copy_and_decrease(order_factory):
    order = order_factory(quantity=10)

    remaining_order = order.copy_and_decrease(3)

    assert order.quantity == 3
    assert remaining_order.quantity == 7
    assert remaining_order.order_id != order.order_id
    assert remaining_order.parent_id == order.order_id


# old tests before fixtures were set:


def test_copy_and_decrease_preserves_order_properties():
    order = Order(
        price=Price(123.45),
        order_type=SELL,
        quantity=10,
        agent_id=42,
        time=17,
        asset_id=5,
        valid_until=100,
    )

    remaining_order = order.copy_and_decrease(3)

    assert remaining_order.price == Price(123.45)
    assert remaining_order.order_type == SELL
    assert remaining_order.quantity == 7
    assert remaining_order.agent_id == 42
    assert remaining_order.time == 17
    assert remaining_order.asset_id == 5
    assert remaining_order.valid_until == 100

    assert order.price == Price(123.45)
    assert order.order_type == SELL
    assert order.quantity == 3
    assert order.agent_id == 42
    assert order.time == 17
    assert order.asset_id == 5
    assert order.valid_until == 100

    # It must be a new order
    assert remaining_order.order_id != order.order_id

    # And it must remember its parent
    assert remaining_order.parent_id == order.order_id


# @pytest.mark.parametrize(
#     ("number_of_buy_orders", "incoming_quantity", "expected_remaining"),
#     [
#         (2493, 2493, 0),
#         (2000, 2493, 493),
#         (3000, 2493, 0),
#     ],
# )
# def test_large_sell_order_against_small_buy_orders(
#     number_of_buy_orders,
#     incoming_quantity,
#     expected_remaining,
# ):
#     ...


def test_large_sell_order_against_many_small_buy_orders(stock):
    fourheap = FourHeap(market=stock)

    number_of_orders = 2493

    # Rest 2493 BUY orders, each for one share.
    for i in range(number_of_orders):
        order = Order(
            price=Price(100.1),
            order_type=BUY,
            quantity=1,
            agent_id=i,
            time=i,
            asset_id=1,
        )
        fourheap.insert(order)

    assert fourheap.buy_unmatched.size == number_of_orders
    assert fourheap.sell_unmatched.size == 0

    # One large SELL order.
    sell_order = Order(
        price=Price(100),
        order_type=SELL,
        quantity=number_of_orders,
        agent_id=9999,
        time=number_of_orders + 1,
        asset_id=1,
    )

    fourheap.insert(sell_order)

    # The whole incoming order was executed.
    # AK - well, the sell should be split to 2493 orders here?
    # assert sell_order.quantity == number_of_orders

    # All resting BUY liquidity was consumed.
    assert fourheap.buy_unmatched.size == 0

    # Nothing remains on the SELL side.
    assert fourheap.sell_unmatched.size == 0

    # Every one-share BUY order became a matched order.
    assert fourheap.buy_matched.size == number_of_orders


def test_large_sell_order_leaves_remaining_quantity(stock):
    fourheap = FourHeap(market=stock)

    resting_quantity = 2000
    incoming_quantity = 2493

    for i in range(resting_quantity):
        order = Order(
            price=Price(100),
            order_type=BUY,
            quantity=1,
            agent_id=i,
            time=i,
            asset_id=1,
        )
        fourheap.insert(order)

    sell_order = Order(
        price=Price(99.8),
        order_type=SELL,
        quantity=incoming_quantity,
        agent_id=9999,
        time=resting_quantity + 1,
        asset_id=1,
    )

    fourheap.insert(sell_order)

    # The incoming Order object represents the executed portion.
    # AK - no, it is split to unit orders?
    # assert sell_order.quantity == resting_quantity

    # All resting BUY orders were consumed.
    assert fourheap.buy_unmatched.size == 0

    # 493 shares remain as a new SELL order.
    assert fourheap.sell_unmatched.size == (incoming_quantity - resting_quantity)


def test_order_price_match_99(stock_tnt: Security, buy_order_99: Order, sell_order_99: Order):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"


def test_order_price_match_99_2(stock_tnt: Security, buy_order_99: Order, sell_order_99: Order):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"


def test_order_price_no_match_99_101(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_101)

    assert fourheap.buy_unmatched.size == 10
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == Price(99.0)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None


def test_order_price_no_match_99_101_2(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)

    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 10
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == Price(99.0)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None


def test_order_price_match_101_99(stock_tnt: Security, buy_order_101: Order, sell_order_99: Order):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(buy_order_101)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_101.executed_price == Price(101.0)
    assert sell_order_99.executed_price == Price(101.0)
    assert buy_order_101.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"


def test_order_price_match_101_99_2(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_101)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_101.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_101.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"


###### crosses with different sizes


def test_order_price_match_99_q(stock_tnt: Security, buy_order_99: Order, sell_order_99: Order):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_99.quantity = 12
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 2
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(99.0)
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"


def test_order_price_match_99_2_q(stock_tnt: Security, buy_order_99: Order, sell_order_99: Order):
    fourheap = FourHeap(market=stock_tnt)
    sell_order_99.quantity = 14
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 4
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(99.0)
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"


def test_order_price_no_match_99_101_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_99.quantity = 17
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_101)

    assert fourheap.buy_unmatched.size == 17
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == Price(99.0)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None


def test_order_price_no_match_99_101_2_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)

    sell_order_101.quantity = 45
    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 10
    assert fourheap.sell_unmatched.size == 45
    assert fourheap.buy_unmatched.peek() == Price(99.00)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None


def test_order_price_match_101_99_q(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)
    sell_order_99.quantity = 88
    fourheap.insert(buy_order_101)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 78
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(99.0)
    assert buy_order_101.executed_price == Price(101.0)
    assert sell_order_99.executed_price == Price(101.0)
    assert buy_order_101.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"


def test_order_price_match_101_99_2_q(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_101.quantity = 81
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_101)

    assert fourheap.buy_unmatched.size == 71
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(101.00)
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_101.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_101.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"


## three orders matched or not:


def test_order_price_match_99_3(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(101.00)
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"
    assert sell_order_99.matched_with == buy_order_99.order_id
    assert buy_order_99.matched_with == sell_order_99.order_id


def test_order_price_match_99_3_2(
    stock_tnt: Security, buy_order_99: Order, sell_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(sell_order_101)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(101.00)
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"
    assert sell_order_99.matched_with == buy_order_99.order_id
    assert buy_order_99.matched_with == sell_order_99.order_id


def test_order_price_no_match_99_101_3(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order, buy_order: Order
):
    fourheap = FourHeap(market=stock_tnt)
    fourheap.insert(buy_order)  # has default limit 100.00
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_101)

    assert fourheap.buy_unmatched.size == 20
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == Price(100.0)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order.executed_price is None
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert buy_order.executed_mode is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None
    assert buy_order.matched_with is None
    assert buy_order_99.matched_with is None
    assert sell_order_101.matched_with is None


def test_order_price_no_match_99_101_3_2(
    stock_tnt: Security, buy_order_99: Order, sell_order: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)

    fourheap.insert(sell_order)  # 100
    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_99)

    assert fourheap.buy_unmatched.size == 10
    assert fourheap.sell_unmatched.size == 20
    assert fourheap.buy_unmatched.peek() == Price(99.0)
    assert fourheap.sell_unmatched.peek() == Price(100.0)
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price is None
    assert sell_order.executed_price is None
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode is None
    assert sell_order.executed_mode is None
    assert buy_order_99.matched_with is None
    assert sell_order_101.matched_with is None
    assert sell_order.matched_with is None


def test_order_price_match_101_99_partial(
    stock_tnt: Security, buy_order: Order, buy_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_101.quantity = 6
    fourheap.insert(buy_order)
    fourheap.insert(buy_order_101)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 6
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(100.00)
    assert fourheap.sell_unmatched.peek() == math.inf
    # partial execution
    assert buy_order_101.executed_price == Price(101.00)
    assert buy_order.executed_price == Price(100.0)
    assert sell_order_99.executed_price == Price(101.0)
    assert buy_order_101.executed_mode == "waited"
    assert buy_order.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"
    assert buy_order.matched_with is not None, "This matches with child order of max"
    assert buy_order_101.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_101.order_id


def test_order_price_match_101_99_partial_2(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order, sell_order: Order
):
    fourheap = FourHeap(market=stock_tnt)
    sell_order_99.quantity = 4
    fourheap.insert(sell_order)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_101)

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 4
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(100.00)
    # partial execution
    assert sell_order.executed_price == Price(100.00)
    assert buy_order_101.executed_price == Price(99.0)
    assert buy_order_101.quantity == 4, "excuted quantity with first matched order"
    assert sell_order_99.executed_price == Price(99.0)
    assert sell_order.executed_mode == "waited"
    assert buy_order_101.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"
    assert buy_order_101.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_101.order_id


###### crosses with different sizes


def test_order_price_match_99_3_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_99: Order, buy_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_99.quantity = 7
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_101)

    assert fourheap.buy_unmatched.size == 7
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(101.0)
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert buy_order_101.executed_price == Price(99.0)
    assert buy_order_99.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"
    assert buy_order_101.executed_mode == "arrived"
    assert buy_order_101.quantity == 3, "Remaining quantity from sell order 10-7"
    assert buy_order_99.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_99.order_id


def test_order_price_match_99_3_2_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    sell_order_99.quantity = 6
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_101)

    assert fourheap.buy_unmatched.size == 4
    assert fourheap.sell_unmatched.size == 10
    assert fourheap.buy_unmatched.peek() == Price(99.00)
    assert fourheap.sell_unmatched.peek() == Price(101.00)
    assert buy_order_99.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"
    assert buy_order_99.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_99.order_id
    assert sell_order_101.matched_with is None


def test_order_price_no_match_99_101_3_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order, buy_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_101.quantity = 17
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_101)

    assert fourheap.buy_unmatched.size == 17
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(101.0)
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_99.executed_price is None
    assert sell_order_101.executed_price == Price(101.00)
    assert buy_order_101.executed_price == Price(101.00)
    assert buy_order_99.executed_mode is None
    assert sell_order_101.executed_mode == "waited"
    assert buy_order_101.executed_mode == "arrived"
    assert buy_order_99.matched_with is None
    assert sell_order_101.matched_with == buy_order_101.order_id
    assert buy_order_101.matched_with == sell_order_101.order_id


def test_order_price_no_match_99_101_3_2_q(
    stock_tnt: Security, buy_order_99: Order, sell_order_101: Order, sell_order_99: Order
):
    fourheap = FourHeap(market=stock_tnt)

    sell_order_101.quantity = 45
    sell_order_99.quantity = 4
    fourheap.insert(sell_order_101)
    fourheap.insert(buy_order_99)
    fourheap.insert(sell_order_99)

    assert fourheap.buy_unmatched.size == 6
    assert fourheap.sell_unmatched.size == 45
    assert fourheap.buy_unmatched.peek() == Price(99.00)
    assert fourheap.sell_unmatched.peek() == Price(101.0)
    assert buy_order_99.executed_price == Price(99.0)
    assert buy_order_99.quantity == 4
    assert sell_order_101.executed_price is None
    assert buy_order_99.executed_mode == "waited"
    assert sell_order_101.executed_mode is None
    assert sell_order_99.executed_mode == "arrived"
    assert buy_order_99.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_99.order_id
    assert sell_order_101.matched_with is None


def test_order_price_match_101_99_3_q(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order, buy_order: Order
):
    fourheap = FourHeap(market=stock_tnt)
    sell_order_99.quantity = 88
    fourheap.insert(buy_order_101)
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order)  # price limit 100

    assert fourheap.buy_unmatched.size == 0
    assert fourheap.sell_unmatched.size == 68
    assert fourheap.buy_unmatched.peek() == -math.inf
    assert fourheap.sell_unmatched.peek() == Price(99.0)
    assert buy_order_101.executed_price == Price(101.0)
    assert sell_order_99.executed_price == Price(101.0)
    assert buy_order.executed_price == Price(99.0)
    assert buy_order_101.executed_mode == "waited"
    assert sell_order_99.executed_mode == "arrived"
    assert buy_order.executed_mode == "arrived"
    assert buy_order_101.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_101.order_id


def test_order_price_match_101_99_3_2_q(
    stock_tnt: Security, buy_order_101: Order, sell_order_99: Order, sell_order_101: Order
):
    fourheap = FourHeap(market=stock_tnt)
    buy_order_101.quantity = 81
    fourheap.insert(sell_order_99)
    fourheap.insert(buy_order_101)
    fourheap.insert(sell_order_101)

    assert fourheap.buy_unmatched.size == 61
    assert fourheap.sell_unmatched.size == 0
    assert fourheap.buy_unmatched.peek() == Price(101.00)
    assert fourheap.sell_unmatched.peek() == math.inf
    assert buy_order_101.executed_price == Price(99.0)
    assert sell_order_99.executed_price == Price(99.0)
    assert sell_order_101.executed_price == Price(101.0)
    assert buy_order_101.executed_mode == "arrived"
    assert sell_order_99.executed_mode == "waited"
    assert sell_order_101.executed_mode == "arrived"
    assert buy_order_101.matched_with == sell_order_99.order_id
    assert sell_order_99.matched_with == buy_order_101.order_id
