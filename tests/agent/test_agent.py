import pytest
from typing import TYPE_CHECKING

from marketsim.agent import (Agent, NoiseAgent, HBLAgent, MMZOHAgent, MomentumAgent, OptionMMZOHAgent, SpoofingAgent,
        WashTradingAgent, ZIAgentNotInformed)

if TYPE_CHECKING:
    from marketsim.market import Price

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

def test_create_agent_option_mm(stock, repository):
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
    assert type(agent_noise) == NoiseAgent
    assert agent_noise.cash == 0


# eod for each type:

def test_eod_noise_agent(agent_noise: Agent):
    agent_noise.eod()

    assert agent_noise is not None
    assert type(agent_noise) == NoiseAgent
    assert agent_noise.cash == 0



# cash flows, trades



# valuations






