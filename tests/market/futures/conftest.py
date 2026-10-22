import pytest

from market import FXRateProvider
from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import Index, Future, Security
from marketsim.agent import Agent, NoiseAgent



@pytest.fixture
def future_short(
    genius_index: Index,
    repository: Repository,
) -> Future:
    return Future(
        repository=repository,
        underlying=genius_index,
        expiration=3,
        multiplier=20,
        margin_rate=0.1,
        name="GeNiUS future - short maturity",
        short_name="FGeN3",
    )


@pytest.fixture
def future_long(
    genius_index: Index,
    repository: Repository,
) -> Future:
    return Future(
        repository=repository,
        underlying=genius_index,
        expiration=10,
        multiplier=20,
        margin_rate=0.1,
        name="GeNiUS future - long maturity",
        short_name="FGeN10",
    )


@pytest.fixture
def futures_markets(
    stock_tnt: Security,
    stock_mit: Security,
    genius_index: Index,
    future_short: Future,
    future_long: Future,
) -> list[Security]:
    return [
        stock_tnt,
        stock_mit,
        genius_index,
        future_short,
        future_long,
    ]


@pytest.fixture
def futures_agents(
    futures_markets: list[Security],
    future_short: Future,
    future_long: Future,
    repository: Repository,
    fx_rate_provider: FXRateProvider
) -> dict[str, Agent]:
    agents = {
        name: NoiseAgent(
            markets=futures_markets,
            repository=repository,
            fx_rate_provider=fx_rate_provider
        )
        for name in ("buyer", "seller", "hedger")
    }

    for contract in (future_short, future_long):
        contract.add_agents(list(agents.values()))

    return agents


@pytest.fixture
def futures_agents_with_positions(
    futures_agents: dict[str, Agent],
    future_short: Future,
    future_long: Future,
) -> dict[str, Agent]:
    buyer = futures_agents["buyer"]
    seller = futures_agents["seller"]
    hedger = futures_agents["hedger"]

    # Start the trading session
    for agent in futures_agents.values():
        agent.sod()

    # Short-maturity contract: +2 / -2
    buyer.update_futures_position(
        quantity=2, contract=future_short
    )
    seller.update_futures_position(
        quantity=-2, contract=future_short
    )

    # Long-maturity contract: +1 / -1
    buyer.update_futures_position(
        quantity=1, contract=future_long
    )
    hedger.update_futures_position(
        quantity=-1, contract=future_long
    )

    return futures_agents

