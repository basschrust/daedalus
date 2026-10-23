from decimal import Decimal

from marketsim.agent import (
    Agent,
    HBLAgent,
    MMZOHAgent,
    MomentumAgent,
    NoiseAgent,
    OptionMMZOHAgent,
    SpoofingAgent,
    WashTradingAgent,
    ZIAgentNotInformed,
)
from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.database.connectors.duckdb_storage import Repository

from marketsim.agent import (Agent, NoiseAgent, HBLAgent, MMZOHAgent, MomentumAgent, OptionMMZOHAgent, SpoofingAgent,
        WashTradingAgent, ZIAgentNotInformed)

from marketsim.market import Price, Security

#######   testing functions   #######


# [NoiseAgent, HBLAgent, MMZOHAgent, MomentumAgent, OptionMMZOHAgent, SpoofingAgent,
#         WashTradingAgent, ZIAgentNotInformed])
# def test_create_agent(stock, repository):
#     agent = (markets=[stock], repository=repository)
#
#     assert agent is not None


# empty agents (simple default arguments):

def test_create_agent_noise(stock, repository):
    agent = NoiseAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_hbl(stock, repository):
    agent = HBLAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_mm(stock, repository):
    agent = MMZOHAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_momentum(stock, repository):
    agent = MomentumAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_option_mm(stock: Security, repository: Repository):
    agent = OptionMMZOHAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_spoof(stock, repository):
    agent = SpoofingAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_wash(stock, repository):
    agent = WashTradingAgent(markets=[stock], repository=repository)

    assert agent is not None

def test_create_agent_zi(stock, repository):
    agent = ZIAgentNotInformed(markets=[stock], repository=repository)

    assert agent is not None


# sod for each type:
def test_sod_noise_agent(agent_noise: Agent):
    agent_noise.sod()

    assert agent_noise is not None
    assert type(agent_noise) is NoiseAgent
    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('0.00')}


# eod for each type:

def test_eod_noise_agent(agent_noise: Agent):
    agent_noise.eod()

    assert agent_noise is not None
    assert type(agent_noise) is NoiseAgent
    assert agent_noise.cash == {'USD': Decimal('0.00'), 'EUR': Decimal('0.00'), 'PLN': Decimal('0.00')}



# cash flows, trades



# valuations


def test_zero_value_future_without_market_is_ignored(agent_noise: Agent):
    agent = agent_noise

    agent.futures_position[999] = 0
    agent.futures_value[999] = {
        "avg_initial_price": Price(0),
        "settlement_price": Price(0),
        "collateral": Price(0),
        "unsettled_pnl": Price(0),
    }

    expected = agent.calculate_portfolio_value()

    assert agent.calculate_portfolio_value() == expected

def test_zero_position_without_market_is_ignored(agent_noise: Agent):
    agent = agent_noise

    expected = agent.calculate_portfolio_value()

    agent.position[999] = 0

    assert agent.calculate_portfolio_value() == expected

