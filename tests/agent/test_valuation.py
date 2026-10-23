
from decimal import Decimal

from marketsim.agent import Agent, MMZOHAgent, NoiseAgent
from marketsim.fourheap import Order
from marketsim.market import Price, Security

# from tests.conftest import stock_tnt





def test_valuation_after_two_trades(stock_tnt: Security,
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

    for agent in [agent_mm, agent_noise]:
        agent.record_valuation(current_time=2)

    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('-2025.00')}
    assert agent_mm.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('2025.00')}
    assert agent_mm.position.get(stock_tnt.asset_id) == -10
    assert agent_mm.position.get(stock_mit.asset_id) == -10
    assert agent_noise.position.get(stock_mit.asset_id) == 10
    assert agent_noise.position.get(stock_tnt.asset_id) == 10
    assert agent_noise.portfolio_value == -5
    assert agent_mm.portfolio_value == 5


def test_valuation_price_change(agent_3k_mit_2k_tnt: Agent,
                                stock_mit: Security,
                                stock_tnt: Security):
    stock_tnt.last_traded_price = Price(165.00)
    stock_mit.last_traded_price = Price(18.67)
    agent_3k_mit_2k_tnt.record_valuation(current_time=1)

    assert agent_3k_mit_2k_tnt.portfolio_value == Price(386010.00)
