import pytest

from marketsim.agent import Agent
from marketsim.market import Option, Price

######## test functions  ######


def test_cf_after_expiry_call(agent_long_call_option_expiration_day: Agent,
                              option_call_at_expiration_itm: Option):
    option_call_at_expiration_itm.eod()
    agent_long_call_option_expiration_day.eod()
    # premium: 167 - 90 = 77, position 10

    assert agent_long_call_option_expiration_day.position[option_call_at_expiration_itm.asset_id] == 0, "Should be 0"
    assert option_call_at_expiration_itm.current_day == 1, "Date should be rolled"
    assert agent_long_call_option_expiration_day.current_day == 1, "Date should be rolled"
    assert agent_long_call_option_expiration_day.cash == Price(770.00), "Cash should be 10*77 position * premium"
    assert agent_long_call_option_expiration_day.portfolio_value == Price(770.00), "Portolio value equal to cash"

def test_cf_after_expiry_put(agent_long_put_option_expiration_day: Agent,
                              option_put_at_expiration_itm: Option):
    option_put_at_expiration_itm.eod()
    agent_long_put_option_expiration_day.eod()
    # premium: 170-167 = 3, position 10

    assert agent_long_put_option_expiration_day.position[option_put_at_expiration_itm.asset_id] == 0, "Should be 0"
    assert option_put_at_expiration_itm.current_day == 1, "Date should be rolled"
    assert agent_long_put_option_expiration_day.current_day == 1, "Date should be rolled"
    assert agent_long_put_option_expiration_day.cash == Price(30.00), "Cash should be 10*3 position * premium"
    assert agent_long_put_option_expiration_day.portfolio_value == Price(30.00), "Portolio value equal to cash"

def test_cf_after_expiry_call_otm(agent_long_call_option_expiration_day_otm: Agent,
                              option_call_at_expiration_otm: Option):
    option_call_at_expiration_otm.eod()
    agent_long_call_option_expiration_day_otm.eod()
    # Premium: 0, position 10

    assert agent_long_call_option_expiration_day_otm.position[option_call_at_expiration_otm.asset_id] == 0, "Should be 0"
    assert option_call_at_expiration_otm.current_day == 1, "Date should be rolled"
    assert agent_long_call_option_expiration_day_otm.current_day == 1, "Date should be rolled"
    assert agent_long_call_option_expiration_day_otm.cash == Price(0.00), "Cash should be 10*0 position * premium"
    assert agent_long_call_option_expiration_day_otm.portfolio_value == Price(0.00), "Portolio value equal to cash"

def test_cf_after_expiry_put_otm(agent_long_put_option_expiration_day_otm: Agent,
                              option_put_at_expiration_otm: Option):
    option_put_at_expiration_otm.eod()
    agent_long_put_option_expiration_day_otm.eod()
    # strike 165,  spot 167
    # Premium: 0

    assert agent_long_put_option_expiration_day_otm.position[option_put_at_expiration_otm.asset_id] == 0, "Should be 0"
    assert option_put_at_expiration_otm.current_day == 1, "Date should be rolled"
    assert agent_long_put_option_expiration_day_otm.current_day == 1, "Date should be rolled"
    assert agent_long_put_option_expiration_day_otm.cash == Price(0.00), "Cash should be 10*0 position * premium"
    assert agent_long_put_option_expiration_day_otm.portfolio_value == Price(0.00), "Portolio value equal to cash"

###### check next day sod() !!!!

def test_sod_1(agent_after_expiry_call: Agent,
               option_call_at_expiration_itm: Option):
    agent = agent_after_expiry_call
    option = option_call_at_expiration_itm

    option.sod()
    agent.sod()

    assert option.current_day == 1, "Date should be rolled"
    assert option.status == "expired", "We're after expiration"
    assert option.eod_status == "closed", "We're after expiration so it doesn't open"

    assert agent.current_day == 1, "Date should be rolled"
    assert agent.position[option.asset_id] == 0, "Should be 0 or not exist?"
    assert agent.portfolio_value == Price(770.00), "Should be 0"
    assert len(agent.portfolio_value_history) == 0, "Daily portfolio value history should be empty"
    assert agent.cash == Price(770.00), "Should be 10*77 position * premium from yesterday"
    assert len(agent.cash_history) == 0, "Cash history should be empty"
    assert agent.eod_status == "open", "After SoD opened for trading"
    assert option.asset_id in agent.abandoned_markets
    assert option.asset_id not in agent.markets

def test_sod_2(agent_after_expiry_put: Agent,
               option_put_at_expiration_itm: Option):
    agent = agent_after_expiry_put
    option = option_put_at_expiration_itm

    option.sod()
    agent.sod()

    assert option.current_day == 1, "Date should be rolled"
    assert option.status == "expired", "We're after expiration"
    assert option.eod_status == "closed", "We're after expiration so it doesn't open"

    assert agent.current_day == 1, "Date should be rolled"
    assert agent.position[option.asset_id] == 0, "Should be 0 or not exist?"
    assert agent.portfolio_value == Price(30.00), "Should be 0"
    assert len(agent.portfolio_value_history) == 0, "Daily portfolio value history should be empty"
    assert agent.cash == Price(30.00), "Should be 10*77 position * premium from yesterday"
    assert len(agent.cash_history) == 0, "Cash history should be empty"
    assert agent.eod_status == "open", "After SoD opened for trading"
    assert option.asset_id in agent.abandoned_markets
    assert option.asset_id not in agent.markets

def test_sod_3(agent_after_expiry_call_otm: Agent,
               option_call_at_expiration_otm: Option):
    agent = agent_after_expiry_call_otm
    option = option_call_at_expiration_otm

    option.sod()
    agent.sod()

    assert option.current_day == 1, "Date should be rolled"
    assert option.status == "expired", "We're after expiration"
    assert option.eod_status == "closed", "We're after expiration so it doesn't open"

    assert agent.current_day == 1, "Date should be rolled"
    assert agent.position[option.asset_id] == 0, "Should be 0 or not exist?"
    assert agent.portfolio_value == Price(0.00), "Should be 0"
    assert len(agent.portfolio_value_history) == 0, "Daily portfolio value history should be empty"
    assert agent.cash == Price(0.00), "Should be 10*77 position * premium from yesterday"
    assert len(agent.cash_history) == 0, "Cash history should be empty"
    assert agent.eod_status == "open", "After SoD opened for trading"
    assert option.asset_id in agent.abandoned_markets
    assert option.asset_id not in agent.markets

def test_sod_4(agent_after_expiry_put_otm: Agent,
               option_put_at_expiration_otm: Option):
    agent = agent_after_expiry_put_otm
    option = option_put_at_expiration_otm

    option.sod()
    agent.sod()

    assert option.current_day == 1, "Date should be rolled"
    assert option.status == "expired", "We're after expiration"
    assert option.eod_status == "closed", "We're after expiration so it doesn't open"

    assert agent.current_day == 1, "Date should be rolled"
    assert agent.position[option.asset_id] == 0, "Should be 0 or not exist?"
    assert agent.portfolio_value == Price(0.00), "Should be 0"
    assert len(agent.portfolio_value_history) == 0, "Daily portfolio value history should be empty"
    assert agent.cash == Price(0.00), "Should be 10*77 position * premium from yesterday"
    assert len(agent.cash_history) == 0, "Cash history should be empty"
    assert agent.eod_status == "open", "After SoD opened for trading"
    assert option.asset_id in agent.abandoned_markets
    assert option.asset_id not in agent.markets
