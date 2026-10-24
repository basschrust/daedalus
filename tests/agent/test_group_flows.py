"""Place at tests/database/test_group_flows.py.

Run with: pytest tests/database/test_group_flows.py -v
"""

import pytest

from marketsim.database.connectors.duckdb_storage import Repository


@pytest.fixture
def repository(tmp_path):
    repo = Repository(output_dir=tmp_path)
    repo.connection.executemany(
        "INSERT INTO agents (agent_id, name, group_name) VALUES (?, ?, ?)",
        [
            (1, "Buyer A", "MM"),
            (2, "Seller B", "NOISE"),
            (3, "Buyer C", "HBL"),
            (4, "No trades", "IDLE"),
        ],
    )
    yield repo
    repo.connection.close()


def add_execution(
    repo, *, day, tick, asset, buy_agent, sell_agent, buy_order, sell_order, volume, price
):
    """Insert both persisted sides of a matched execution."""
    cash = volume * price
    rows = [
        (day, asset, buy_agent, sell_order, tick, price, buy_order, 1, volume, -cash),
        (day, asset, sell_agent, buy_order, tick, price, sell_order, -1, volume, cash),
    ]
    repo.connection.executemany(
        """INSERT INTO trades
        (day, asset_id, agent_id, matched_with, executed_time, executed_price,
         order_id, order_side, executed_volume, cash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


def test_two_sides_and_cash_conservation(repository):
    add_execution(
        repository,
        day=0,
        tick=4,
        asset=10,
        buy_agent=1,
        sell_agent=2,
        buy_order=100,
        sell_order=200,
        volume=7,
        price=12,
    )
    repository.calculate_group_flows()
    buyer = repository.get_group_flows(agent_id=1).iloc[0]
    seller = repository.get_group_flows(agent_id=2).iloc[0]
    assert (buyer.agent_group, buyer.cp_group) == ("MM", "NOISE")
    assert (buyer.volume_buy, buyer.volume_sell, buyer.cash_buy, buyer.cash_sell) == (7, 0, 84, 0)
    assert (seller.agent_group, seller.cp_group) == ("NOISE", "MM")
    assert (seller.volume_buy, seller.volume_sell, seller.cash_buy, seller.cash_sell) == (
        0,
        7,
        0,
        84,
    )
    assert buyer.cash_buy == seller.cash_sell
    assert repository.get_group_flows(agent_id=4).empty


def test_multiday_multiasset_and_aggregation(repository):
    add_execution(
        repository,
        day=0,
        tick=4,
        asset=10,
        buy_agent=1,
        sell_agent=2,
        buy_order=100,
        sell_order=200,
        volume=7,
        price=12,
    )
    add_execution(
        repository,
        day=0,
        tick=4,
        asset=10,
        buy_agent=1,
        sell_agent=2,
        buy_order=101,
        sell_order=201,
        volume=3,
        price=12,
    )
    add_execution(
        repository,
        day=1,
        tick=4,
        asset=10,
        buy_agent=1,
        sell_agent=2,
        buy_order=102,
        sell_order=202,
        volume=5,
        price=14,
    )
    add_execution(
        repository,
        day=1,
        tick=4,
        asset=11,
        buy_agent=1,
        sell_agent=3,
        buy_order=103,
        sell_order=203,
        volume=2,
        price=20,
    )
    repository.calculate_group_flows()
    df = repository.get_group_flows(agent_id=1)
    assert len(df) == 3
    assert list(zip(df.day, df.asset_id, df.cp_group)) == [
        (0, 10, "NOISE"),
        (1, 10, "NOISE"),
        (1, 11, "HBL"),
    ]
    assert list(df.volume_buy) == [10, 5, 2]
    assert list(df.cash_buy) == [120, 70, 40]


def test_rebuild_is_idempotent(repository):
    add_execution(
        repository,
        day=0,
        tick=0,
        asset=10,
        buy_agent=1,
        sell_agent=2,
        buy_order=100,
        sell_order=200,
        volume=7,
        price=12,
    )
    repository.calculate_group_flows()
    before = repository.get_group_flows(agent_id=1).copy()
    repository.calculate_group_flows()
    after = repository.get_group_flows(agent_id=1)
    assert before.equals(after)
    assert repository.connection.execute("SELECT COUNT(*) FROM l1_group_flows").fetchone()[0] == 2


def test_unmatched_and_wrong_day_do_not_join(repository):
    repository.connection.execute(
        """INSERT INTO trades
        (day, asset_id, agent_id, matched_with, executed_time, order_id,
         order_side, executed_volume, cash)
        VALUES (0, 10, 1, 200, 3, 100, 1, 5, -50)"""
    )
    repository.connection.execute(
        """INSERT INTO trades
        (day, asset_id, agent_id, matched_with, executed_time, order_id,
         order_side, executed_volume, cash)
        VALUES (1, 10, 2, 100, 3, 200, -1, 5, 50)"""
    )
    repository.calculate_group_flows()
    assert repository.get_group_flows(agent_id=1).empty
    assert repository.get_group_flows(agent_id=2).empty
