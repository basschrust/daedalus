
from decimal import Decimal

from marketsim.agent import MMZOHAgent, NoiseAgent
from marketsim.fourheap import Order
from marketsim.market import Security

# from tests.conftest import stock_tnt




#####   test  functions ##########

def test_cf_1(stock_tnt: Security,
              agent_noise: NoiseAgent,
              buy_order_101_tnt: Order,
              agent_mm: MMZOHAgent,
              sell_order_99_tnt: Order):

    stock_tnt.add_agents([agent_mm, agent_noise])
    stock_tnt.sod()
    stock_tnt.add_orders([buy_order_101_tnt])
    stock_tnt.step(1)
    sell_order_99_tnt.time = 2
    stock_tnt.add_orders([sell_order_99_tnt])
    new_orders_matched = stock_tnt.step(2)

    for matched_order in new_orders_matched:
        stock_tnt.record_trade(matched_order=matched_order)

    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('-1010.00')}
    assert agent_noise.agent_id == buy_order_101_tnt.agent_id
    assert agent_noise.position.get(stock_tnt.asset_id) == 10, "Inventory should be 10 - equal to the order volume"
    assert agent_mm.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('1010.00')}
    assert agent_mm.agent_id == sell_order_99_tnt.agent_id
    assert agent_mm.position.get(stock_tnt.asset_id) == -10, "Inventory should be -10 - equal to the order volume but negative"
    # TODO: inspect the security stats, too



def test_cf_after_sod(stock_tnt: Security,
              agent_noise: NoiseAgent,
              agent_mm: MMZOHAgent,
              stock_mit: Security,
              ):

    stock_mit.sod()
    stock_tnt.sod()

    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('0.00')}
    assert agent_mm.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('0.00')}
    assert agent_mm.position.get(stock_tnt.asset_id) == 0
    assert agent_mm.position.get(stock_mit.asset_id) == 0
    assert agent_noise.position.get(stock_mit.asset_id) == 0
    assert agent_noise.position.get(stock_tnt.asset_id) == 0


def test_cf_after_two_trades(stock_tnt: Security,
              agent_noise: NoiseAgent,
              buy_order_101_tnt: Order,
              agent_mm: MMZOHAgent,
              sell_order_99_tnt: Order,
              stock_mit: Security,
              buy_order_102_mit_5: Order,
              sell_order_101_mit: Order,
              buy_order_101_mit: Order,
              ):

    stock_tnt.add_agents([agent_mm, agent_noise])
    stock_mit.add_agents([agent_mm, agent_noise])
    stock_mit.sod()
    stock_tnt.sod()

    stock_tnt.add_orders([buy_order_101_tnt])
    stock_mit.add_orders([buy_order_102_mit_5])
    stock_tnt.step(1)
    stock_mit.step(1)
    sell_order_99_tnt.time = 2
    sell_order_101_mit.time = 2
    buy_order_101_mit.time = 2
    stock_tnt.add_orders([sell_order_99_tnt])
    stock_mit.add_orders([sell_order_101_mit, buy_order_101_mit])

    new_orders_matched = stock_tnt.step(2)
    for matched_order in new_orders_matched:
        stock_tnt.record_trade(matched_order=matched_order)

    new_orders_matched = stock_mit.step(2)
    for matched_order in new_orders_matched:
        stock_mit.record_trade(matched_order=matched_order)

    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('-2025.00')}
    assert agent_mm.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('2025.00')}
    assert agent_mm.position.get(stock_tnt.asset_id) == -10
    assert agent_mm.position.get(stock_mit.asset_id) == -10
    assert agent_noise.position.get(stock_mit.asset_id) == 10
    assert agent_noise.position.get(stock_tnt.asset_id) == 10

