import pytest

import pandas as pd

from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import Security, Price, Option
from marketsim.agent import Agent


EXPECTED_TABLES = {"securities",
                   "options",
                   "agents",
                    "position_history",
                    "eod_positions",
                    "portfolio_value_history",
                    "eod_portfolio_value",
                    "cash_history",
                    "eod_cash",
                    "traded_prices",
                    "eod_prices",
                    "orders",
                    "trades",
                    "option_expiration"}


###### test functions  #####


def test_database_creation(repository: Repository):
    assert repository is not None
    assert repository.localdb.name == "daedalus.duckdb"


def test_expected_tables_exist(repository):
    rows = repository.connection.execute("SHOW TABLES").fetchall()
    actual_tables = {row[0] for row in rows}

    assert EXPECTED_TABLES <= actual_tables


def test_tables_are_empty_at_initialization(repository):
    for table in EXPECTED_TABLES:
        count = repository.connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

        assert count == 0, f"Table {table!r} should be empty, got {count} rows"



####  getting data

@pytest.mark.skip(reason="TODO")
@pytest.mark.usefixtures("position_history_1")
def test_get_eod_positions_filters_by_agent(repository: Repository):
    actual = repository.get_eod_positions(agent_id=1)

    expected = pd.DataFrame({
        "day": [0, 0],
        "asset_id": [0, 1],
        "position": [100, 50],
    })

    actual = actual.sort_values("asset_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(actual, expected)

@pytest.mark.skip(reason="TODO")
def test_get_eod_prices(repository: Repository):
    actual = repository.get_eod_prices(asset_id=1)

    # actual = actual.sort_values("asset_id").reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 0],
        "asset_id": [0, 1],
        "position": [100, 50],
    })

    pd.testing.assert_frame_equal(actual, expected)

@pytest.mark.skip(reason="TODO")
def test_get_eod_cash(repository: Repository):
    actual = repository.get_eod_cash(agent_id=1)

    # actual = actual.sort_values("asset_id").reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 0],
        "asset_id": [0, 1],
        "position": [100, 50],
    })

    pd.testing.assert_frame_equal(actual, expected)

@pytest.mark.skip(reason="TODO")
def test_get_eod_portfolio_values(repository: Repository):
    actual = repository.get_eod_portfolio_values(agent_id=1)

    # actual = actual.sort_values("asset_id").reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 0],
        "asset_id": [0, 1],
        "position": [100, 50],
    })

    pd.testing.assert_frame_equal(actual, expected)



## saving data:

def test_save_security(repository: Repository,
                       stock_tnt: Security):
    #repository.save_security(stock_tnt)

    ret = repository.connection.execute("""SELECT s.asset_id,
                                                  s.instrument_class,
                                                  s.market_type,
                                                  s.name,
                                                  s.eod_status,
                                                  s.reference_price
                                          FROM securities s""").fetchdf()

    # TODO: duplicates?  there are currently 2 rows as Security.__init__  already saves the security
    assert ret.shape == (1, 6)
    row = ret.iloc[0]
    assert row["asset_id"] == stock_tnt.asset_id
    assert row["instrument_class"] == stock_tnt.instrument_class
    assert row["market_type"] == stock_tnt.market_type
    assert row["name"] == stock_tnt.name
    assert row["eod_status"] == stock_tnt.eod_status
    assert row["reference_price"] == stock_tnt.reference_price


def test_save_agent(repository: Repository,
                    agent_long_put_option_expiration_day: Agent):

    agent = agent_long_put_option_expiration_day

    ret = repository.connection.execute("""SELECT
                                                agent_id,
                                                name,
                                                group_name,
                                                configuration
                                            FROM agents""").fetchdf()
    row = ret.iloc[0]

    assert ret.shape == (1, 4)
    assert row["agent_id"] == agent.agent_id
    assert row["name"] == agent.name
    assert row["group_name"] == agent.group
    #assert row["configuration"] == agent.configuration # TODO: discrepancies due to casting str->Decimal


def test_save_position_history(repository: Repository,
                               agent_3k_mit_2k_tnt: Agent):
    agent = agent_3k_mit_2k_tnt
    agent.eod()

    ret = repository.connection.execute("""SELECT *
                        FROM position_history 
                        ORDER BY time_tick, asset_id""").fetchdf()

    assert ret.shape == (4, 5)

    # TODO: intraday history!
    assert ret.iloc[0]["day"] == 0



def test_save_eod_position(repository: Repository,
                           agent_sophisticated_1: Agent,
                           stock_tnt: Security,
                           stock_mit: Security,):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT day, asset_id, agent_id, position
                FROM eod_positions 
                ORDER BY day DESC, agent_id, asset_id""").fetchdf()

    assert ret.shape == (4, 4)
    assert ret.iloc[0]["day"] == 1
    assert ret.iloc[0]["agent_id"] == agent.agent_id
    assert ret.iloc[0]["asset_id"] == stock_tnt.asset_id
    assert ret.iloc[0]["position"] == 2000

    assert ret.iloc[1]["day"] == 1
    assert ret.iloc[1]["agent_id"] == agent.agent_id
    assert ret.iloc[1]["asset_id"] == stock_mit.asset_id
    assert ret.iloc[1]["position"] == 3000

def test_save_portfolio_value_history(repository: Repository,
                                      agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT *
                FROM portfolio_value_history""").fetchdf()

    assert ret.shape == (0,4)


def test_save_eod_portfolio_value(repository: Repository,
                                  agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT *
                FROM eod_portfolio_value 
                ORDER BY day DESC""").fetchdf()

    row = ret.iloc[0]

    assert ret.shape == (2, 3)
    assert row["day"] == agent.current_day - 1
    assert row["agent_id"] == agent.agent_id
    assert row["portfolio_value"] == agent.portfolio_value

@pytest.mark.skip(reason="intraday!")
def test_save_cash_history(repository: Repository,
                           agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()
    ret = repository.connection.execute("""SELECT *
            FROM cash_history""").fetchdf()

    row = ret.iloc[0]
    assert ret.shape == (1, 4)
    assert row["day"] == agent.current_day - 1
    assert row["time_tick"] == 0
    assert row["agent_id"] == agent.agent_id
    assert row["cash"] == agent.cash


def test_save_eod_cash(repository: Repository,
                       agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT *
            FROM eod_cash""").fetch_df()

    assert ret.shape == (2, 3)

def test_save_eod_cash_2(repository: Repository,
                       agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()
    # two days so two records are saved
    agent.sod()
    agent.eod()

    ret = repository.connection.execute("""SELECT *
            FROM eod_cash""").fetch_df()

    assert ret.shape == (3, 3)


def test_save_traded_prices(repository: Repository):
    ret = repository.connection.execute("""SELECT *
                FROM traded_prices""").fetchdf()

    assert ret.shape == (0, 9)

def test_save_eod_prices(repository: Repository):
    ret = repository.connection.execute("""SELECT *
                    FROM eod_prices""").fetchdf()

    assert ret.shape == (0, 8)


def test_save_orders(repository: Repository):
    ret = repository.connection.execute("""SELECT *
                    FROM orders""").fetchdf()

    assert ret.shape == (0, 13)


def test_save_trades(repository: Repository):
    ret = repository.connection.execute("""SELECT *
                    FROM trades""").fetchdf()

    assert ret.shape == (0, 11)


def test_save_option_expiration(repository: Repository,
                                option_call_before_expiration_itm):
    ret = repository.connection.execute("""SELECT *
                    FROM option_expiration""").fetchdf()

    assert ret.shape == (0, 4)


