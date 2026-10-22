import pytest

from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import Security, Option, Price, Index
# from tests.conftest import repository, option_tnt_call
from marketsim.agent import Agent, NoiseAgent

@pytest.fixture
def genius_index(
    stock_tnt: Security,
    stock_mit: Security,
    repository: Repository,
) -> Index:
    return Index(
        repository=repository,
        name="GeNiUS index",
        short_name="GeNiUS",
        components={
            stock_tnt.asset_id: {
                "weight": 0.4,
                "security": stock_tnt,
            },
            stock_mit.asset_id: {
                "weight": 0.6,
                "security": stock_mit,
            },
        },
    )


@pytest.fixture
def derivatives_config1(stock: Security):
    config1 = { "underlying": stock,
                "strike": Price(90),
                "expiration": 30,
                "option_side": "CALL",
                "option_type": "European",
                }
    return config1



@pytest.fixture
def option_call_short(index_genius, repository):
    return Option(
        repository=repository,
        underlying=index_genius,
        short_name="GeNiUS-C",
        derivatives_config={
            "strike": 75,
            "expiration": 3,
            "option_side": "CALL",
            "option_type": "European",
        },
    )


@pytest.fixture
def option_put_short(index_genius, repository):
    return Option(
        repository=repository,
        underlying=index_genius,
        short_name="GeNiUS-P",
        derivatives_config={
            "strike": 75,
            "expiration": 3,
            "option_side": "PUT",
            "option_type": "European",
        },
    )


@pytest.fixture
def option_call_long(index_genius, repository):
    return Option(
        repository=repository,
        underlying=index_genius,
        short_name="GeNiUS-C",
        derivatives_config={
            "strike": 85,
            "expiration": 10,
            "option_side": "CALL",
            "option_type": "European",
        },
    )


@pytest.fixture
def option_put_long(index_genius, repository):
    return Option(
        repository=repository,
        underlying=index_genius,
        short_name="GeNiUS-P",
        derivatives_config={
            "strike": 85,
            "expiration": 10,
            "option_side": "PUT",
            "option_type": "European",
        },
    )

