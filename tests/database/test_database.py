
import pandas as pd

from marketsim.agent import Agent
from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.market import Security

EXPECTED_TABLES = {
    "securities",
    "indices",
    "index_components",
    "futures",
    "options",
    "agents",
    "agent_position_history",
    "agent_futures_position_history",
    "agent_futures_value_history",
    "agent_eod_positions",
    "agent_eod_futures_position",
    "agent_eod_futures_value",
    "agent_portfolio_value_history",
    "agent_eod_portfolio_value",
    "agent_cash_history",
    "agent_eod_cash",
    "traded_prices",
    "eod_prices",
    "index_values",
    "eod_index_values",
    "future_rates",
    "eod_future_rates",
    "orders",
    "trades",
    "option_expiration",
}


###### test functions  #####


def test_database_creation(repository: Repository):
    assert repository is not None
    assert repository.localdb.name == "daedalus.duckdb"


def test_expected_tables_exist(repository):
    rows = repository.connection.execute("SHOW TABLES").fetchall()
    actual_tables = {row[0] for row in rows}

    assert actual_tables == EXPECTED_TABLES, (
        f"Missing: {EXPECTED_TABLES - actual_tables}; "
        f"Unexpected: {actual_tables - EXPECTED_TABLES}"
    )


def test_tables_are_empty_at_initialization(repository):
    for table in EXPECTED_TABLES:
        count = repository.connection.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

        assert count == 0, f"Table {table!r} should be empty, got {count} rows"



####  getting data


def test_get_eod_positions_filters_by_agent(repository: Repository):
    repository.connection.execute("""
        INSERT INTO agent_eod_positions
            (day, agent_id, asset_id, position)
        VALUES
            (0, 1, 10, 100),
            (0, 1, 11, 50),
            (0, 2, 10, 999),
            (1, 1, 10, 120)
    """)

    actual = repository.get_eod_positions(agent_id=1)
    actual = actual.sort_values(["day", "asset_id"]).reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 0, 1],
        "asset_id": [10, 11, 10],
        "position": [100, 50, 120],
    }).astype("int32")

    pd.testing.assert_frame_equal(actual, expected)


def test_get_eod_prices(repository: Repository):
    repository.connection.execute("""
        INSERT INTO eod_prices
            (day, asset_id, open, high, low, close, volume, theoretical)
        VALUES
            (0, 10, 100, 110, 95, 105, 1000, NULL),
            (1, 10, 105, 115, 100, 112, 1200, NULL),
            (0, 11, 200, 210, 190, 205, 500, NULL)
    """)

    actual = repository.get_eod_prices(asset_id=10)
    actual = actual.sort_values("day").reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 1],
        "open": [100.0, 105.0],
        "high": [110.0, 115.0],
        "low": [95.0, 100.0],
        "close": [105.0, 112.0],
        "volume": [1000.0, 1200.0],
        "theoretical": [float("nan"), float("nan")],
    }).astype({"day": "int32"})

    pd.testing.assert_frame_equal(actual, expected)


def test_get_eod_cash(repository: Repository):
    repository.connection.execute("""
        INSERT INTO agent_eod_cash
            (day, agent_id, cash, currency)
        VALUES
            (0, 1, 1000, 'PLN'),
            (0, 1, 100, 'USD'),
            (1, 1, 1200, 'PLN'),
            (0, 2, 9999, 'PLN')
    """)

    actual = repository.get_eod_cash(agent_id=1)
    actual = actual.sort_values(["day", "currency"]).reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 0, 1],
        "currency": ["PLN", "USD", "PLN"],
        "cash": [1000.0, 100.0, 1200.0],
    }).astype({"day": "int32"})

    pd.testing.assert_frame_equal(actual, expected)


def test_get_eod_portfolio_values(repository: Repository):
    repository.connection.execute("""
        INSERT INTO agent_eod_portfolio_value
            (day, agent_id, portfolio_value)
        VALUES
            (0, 1, 5200.00),
            (1, 1, 5276.80),
            (0, 2, 9999.00)
    """)

    actual = repository.get_eod_portfolio_values(agent_id=1)
    actual = actual.sort_values("day").reset_index(drop=True)

    expected = pd.DataFrame({
        "day": [0, 1],
        "portfolio_value": [5200.00, 5276.80],
    }).astype({"day": "int32"})

    pd.testing.assert_frame_equal(actual, expected)


def test_save_cash_history(repository: Repository):
    cash_history_df = pd.DataFrame([
        {
            "day": 0,
            "time_tick": 10,
            "agent_id": 1,
            "currency": "PLN",
            "cash": 1000.0,
        },
        {
            "day": 0,
            "time_tick": 10,
            "agent_id": 1,
            "currency": "USD",
            "cash": 100.0,
        },
        {
            "day": 0,
            "time_tick": 20,
            "agent_id": 1,
            "currency": "PLN",
            "cash": 900.0,
        },
    ])

    repository.save_cash_history(cash_history_df)

    actual = repository.connection.execute("""
        SELECT day, time_tick, agent_id, currency, cash
        FROM agent_cash_history
        ORDER BY time_tick, currency
    """).fetchdf()

    expected = (cash_history_df[
        ["day", "time_tick", "agent_id", "currency", "cash"]
    ].sort_values(["time_tick", "currency"]).reset_index(drop=True).
                astype({"day": "int32", "time_tick": "int32", "agent_id": "int32"}))

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
                        FROM agent_position_history 
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
                FROM agent_eod_positions 
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
                FROM agent_portfolio_value_history""").fetchdf()

    assert ret.shape == (0,4)


def test_save_eod_portfolio_value(repository: Repository,
                                  agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT *
                FROM agent_eod_portfolio_value 
                ORDER BY day DESC""").fetchdf()

    row = ret.iloc[0]

    assert ret.shape == (2, 3)
    assert row["day"] == agent.current_day - 1
    assert row["agent_id"] == agent.agent_id
    assert row["portfolio_value"] == agent.portfolio_value



def test_save_eod_cash(repository: Repository,
                       agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()

    ret = repository.connection.execute("""SELECT *
            FROM agent_eod_cash""").fetch_df()

    assert ret.shape == (6, 4)

def test_save_eod_cash_2(repository: Repository,
                       agent_sophisticated_1: Agent):
    agent = agent_sophisticated_1
    agent.eod()
    # two days so two records are saved
    agent.sod()
    agent.eod()

    ret = repository.connection.execute("""SELECT *
            FROM agent_eod_cash""").fetch_df()

    assert ret.shape == (9, 4)


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


