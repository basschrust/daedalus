import pytest

from marketsim.market.fourheap import FourHeap
from marketsim.market.fourheap.constants import BUY, SELL
from marketsim.market.fourheap.order import Order
from marketsim.market.price import Price

PKO_ID = 100
KGH_ID = 200


def make_order(
    asset_id,
    side=BUY,
    price="109.54",
    quantity=100,
):
    return Order(
        price=Price(price),
        order_type=side,
        quantity=quantity,
        agent_id=38,
        time=0,
        asset_id=asset_id,
    )


def test_same_asset_order_is_accepted():
    book = FourHeap(market_id=PKO_ID)

    order = make_order(asset_id=PKO_ID)
    book.insert(order)

    assert book.buy_unmatched.contains(order.order_id)
    assert book.buy_unmatched.size == 100


@pytest.mark.parametrize("side", [BUY, SELL])
def test_cross_asset_order_is_rejected(side):
    book = FourHeap(market_id=PKO_ID)

    foreign_order = make_order(
        asset_id=KGH_ID,
        side=side,
    )

    with pytest.raises(
        AssertionError,
        match="Cross-asset order detected",
    ):
        book.insert(foreign_order)

    assert book.buy_unmatched.size == 0
    assert book.sell_unmatched.size == 0
    assert book.buy_matched.size == 0
    assert book.sell_matched.size == 0


def test_cross_asset_order_cannot_match_existing_liquidity():
    book = FourHeap(market_id=PKO_ID)

    resting_sell = make_order(
        asset_id=PKO_ID,
        side=SELL,
        price="109.54",
        quantity=100,
    )
    book.insert(resting_sell)

    foreign_buy = make_order(
        asset_id=KGH_ID,
        side=BUY,
        price="200.00",
        quantity=100,
    )

    # This BUY would cross the resting SELL if not rejected.
    with pytest.raises(
        AssertionError,
        match="Cross-asset order detected",
    ):
        book.insert(foreign_buy)

    # The legitimate PKO order must remain untouched.
    assert book.sell_unmatched.contains(resting_sell.order_id)
    assert book.sell_unmatched.size == 100
    assert book.buy_matched.size == 0
    assert book.sell_matched.size == 0


def test_two_markets_keep_orders_separate():
    pko_book = FourHeap(market_id=PKO_ID)
    kgh_book = FourHeap(market_id=KGH_ID)

    pko_order = make_order(
        asset_id=PKO_ID,
        price="109.54",
    )
    kgh_order = make_order(
        asset_id=KGH_ID,
        price="293.20",
    )

    pko_book.insert(pko_order)
    kgh_book.insert(kgh_order)

    assert pko_book.buy_unmatched.contains(pko_order.order_id)
    assert not pko_book.buy_unmatched.contains(kgh_order.order_id)

    assert kgh_book.buy_unmatched.contains(kgh_order.order_id)
    assert not kgh_book.buy_unmatched.contains(pko_order.order_id)

    assert pko_book.buy_unmatched.size == 100
    assert kgh_book.buy_unmatched.size == 100
