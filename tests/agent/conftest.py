from decimal import Decimal

import pytest

from marketsim.agent import (
    Agent,
    MomentumAgent,
    NoiseAgent,
    OptionMMZOHAgent,
    SpoofingAgent,
    WashTradingAgent,
    ZIAgentNotInformed,
)
from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import FXRateProvider, Option, Price, Security

BUY = 1
SELL = -1


@pytest.fixture
def fx_provider():
    fx = FXRateProvider()

    for day, usd, eur in [
        (0, "3.80", "4.25"),
        (1, "3.90", "4.20"),
    ]:
        fx.set_rate(day=day, base="USD", quote="PLN", rate=usd)
        fx.set_rate(day=day, base="EUR", quote="PLN", rate=eur)

        # triangled rate:
        fx.set_rate(
            day=day,
            base="EUR",
            quote="USD",
            rate=Decimal(eur) / Decimal(usd),
        )

    return fx


@pytest.fixture
def agent_momentum(security: Security, repository: Repository):
    return MomentumAgent(markets=[security], repository=repository)

@pytest.fixture
def agent_spoof(security: Security, repository: Repository):
    return SpoofingAgent(markets=[security], repository=repository)

@pytest.fixture
def agent_washtrader(security: Security, repository: Repository):
    return WashTradingAgent(markets=[security], repository=repository)

@pytest.fixture
def agent_zi(security: Security, repository: Repository):
    return ZIAgentNotInformed(markets=[security], repository=repository)

@pytest.fixture
def agent_option_mm(security: Security, option: Option, repository: Repository):
    return OptionMMZOHAgent(markets=[security], repository=repository)


@pytest.fixture
def multicurrency_agent(repository, fx_provider):
    markets = [
        Security(
            name="WIGCO",
            currency="PLN",
            repository=repository,
        ),
        Security(
            name="USCO",
            currency="USD",
            repository=repository,
        ),
        Security(
            name="EUROCO",
            currency="EUR",
            repository=repository,
        ),
    ]

    agent = NoiseAgent(
        markets=markets,
        repository=repository,
        fx_rate_provider=fx_provider,
    )

    agent.cash = {
        "PLN": Price(1000),
        "USD": Price(100),
        "EUR": Price(200),
    }

    for market, quantity in zip(markets, [10, 5, 3]):
        agent.position[market.asset_id] = quantity

    return agent

######### for cash flow check between agents:

@pytest.fixture
def buy_order_tnt(order_factory,
                     stock_tnt: Security,
                     agent_noise: Agent):
    return order_factory(agent_id=agent_noise.agent_id, asset_id=stock_tnt.asset_id)

@pytest.fixture
def buy_order_99_tnt(order_factory,
                     stock_tnt: Security,
                     agent_noise: Agent):
    return order_factory(price=99, order_type=BUY, agent_id=agent_noise.agent_id, asset_id=stock_tnt.asset_id)

@pytest.fixture
def buy_order_101_tnt(order_factory,
                     stock_tnt: Security,
                     agent_noise: Agent):
    return order_factory(price=101, order_type=BUY, agent_id=agent_noise.agent_id, asset_id=stock_tnt.asset_id)

@pytest.fixture
def sell_order_tnt(order_factory,
                     stock_tnt: Security,
                     agent_mm: Agent):
    return order_factory(order_type=SELL, agent_id=agent_mm.agent_id, asset_id=stock_tnt.asset_id)

@pytest.fixture
def sell_order_99_tnt(order_factory,
                     stock_tnt: Security,
                     agent_mm: Agent):
    return order_factory(price=99, order_type=SELL, agent_id=agent_mm.agent_id, asset_id=stock_tnt.asset_id)

@pytest.fixture
def sell_order_101_tnt(order_factory,
                     stock_tnt: Security,
                     agent_mm: Agent):
    return order_factory(price=101, order_type=SELL, agent_id=agent_mm.agent_id, asset_id=stock_tnt.asset_id)

# MIT orders

@pytest.fixture
def sell_order_101_mit(order_factory,
                     stock_mit: Security,
                     agent_mm: Agent):
    return order_factory(price=101, order_type=SELL, agent_id=agent_mm.agent_id, asset_id=stock_mit.asset_id)

@pytest.fixture
def buy_order_102_mit_5(order_factory,
                     stock_mit: Security,
                     agent_noise: Agent):
    return order_factory(price=102,
                         order_type=BUY,
                         agent_id=agent_noise.agent_id,
                         asset_id=stock_mit.asset_id,
                         quantity=5)

@pytest.fixture
def buy_order_101_mit(order_factory,
                     stock_mit: Security,
                     agent_noise: Agent):
    return order_factory(price=101, order_type=BUY, agent_id=agent_noise.agent_id, asset_id=stock_mit.asset_id)


##### agents with open positions



