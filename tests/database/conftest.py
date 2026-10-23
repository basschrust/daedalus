import pytest

from marketsim.agent import Agent
from marketsim.database.connectors.duckdb_storage import Repository

# for testing data extraction:

@pytest.fixture
def position_history_1(repository: Repository):
    repository.connection.execute("""
            INSERT INTO position_history
                (day, time_tick, agent_id, asset_id, position)
            VALUES
                (0, 10, 1, 0, 100),
                (0, 10, 1, 1,  50),
                (0, 10, 2, 0, 999)
        """)


###### for testing of data saving:
# most of those functions are called inside Agent eod()   Therefore we'll create 2 sophisticated agents
# and check if their data is properly injected

@pytest.fixture
def agent_sophisticated_1(repository: Repository,
                          agent_3k_mit_2k_tnt: Agent):

    return agent_3k_mit_2k_tnt






