"""Tests for daily L1 group-flow aggregation."""

import pandas as pd
import pytest

from marketsim.database.connectors.duckdb_storage import Repository
from marketsim.database.connectors.snowflake_storage import TABLES

COLUMNS = [
    "day",
    "asset_id",
    "agent_group",
    "cp_group",
    "volume_buy",
    "volume_sell",
    "cash_buy",
    "cash_sell",
]


@pytest.fixture
def repo(tmp_path):
    repository = Repository(output_dir=tmp_path)
    try:
        yield repository
    finally:
        repository.connection.close()


def seed_intraday_flows(repo):
    """Populate intraday L1 with multiple days, assets and group pairs."""
    rows = [
        # Same day/asset/groups, different ticks and agents.
        (0, 1, 10, 101, "Noise", "Maker", 3, 0, 300, 0),
        (0, 2, 10, 102, "Noise", "Maker", 2, 0, 210, 0),
        (0, 3, 10, 101, "Noise", "Maker", 0, 4, 0, 420),
        # Reverse group direction.
        (0, 3, 10, 201, "Maker", "Noise", 4, 0, 420, 0),
        # Different instrument.
        (0, 1, 11, 101, "Noise", "Maker", 7, 0, 700, 0),
        # Different day.
        (1, 1, 10, 101, "Noise", "Maker", 8, 1, 800, 100),
    ]

    repo.connection.executemany(
        """
        INSERT INTO l1_group_flows (
            day, time_tick, asset_id, agent_id,
            agent_group, cp_group,
            volume_buy, volume_sell, cash_buy, cash_sell
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def test_daily_aggregation(repo):
    """Ticks and agents aggregate, while days/assets/groups stay distinct."""
    seed_intraday_flows(repo)
    repo.calculate_eod_group_flows()

    result = repo.get_all_eod_group_flows()

    assert list(result.columns) == COLUMNS
    assert len(result) == 4

    keyed = result.set_index(["day", "asset_id", "agent_group", "cp_group"])

    row = keyed.loc[(0, 10, "Noise", "Maker")]

    assert row["volume_buy"] == pytest.approx(5)
    assert row["volume_sell"] == pytest.approx(4)
    assert row["cash_buy"] == pytest.approx(510)
    assert row["cash_sell"] == pytest.approx(420)

    # Reverse counterparty direction is distinct.
    assert keyed.loc[(0, 10, "Maker", "Noise"), "volume_buy"] == pytest.approx(4)

    # Different instrument and day remain separate.
    assert keyed.loc[(0, 11, "Noise", "Maker"), "volume_buy"] == pytest.approx(7)
    assert keyed.loc[(1, 10, "Noise", "Maker"), "volume_buy"] == pytest.approx(8)


def test_daily_totals_match_intraday(repo):
    """Aggregation must preserve all four numerical measures."""
    seed_intraday_flows(repo)
    repo.calculate_eod_group_flows()

    daily = repo.get_all_eod_group_flows()

    intraday = repo.connection.execute(
        """
        SELECT volume_buy, volume_sell, cash_buy, cash_sell
        FROM l1_group_flows
        """
    ).fetchdf()

    for column in (
        "volume_buy",
        "volume_sell",
        "cash_buy",
        "cash_sell",
    ):
        assert daily[column].sum() == pytest.approx(intraday[column].sum())


def test_rebuild_is_idempotent(repo):
    """Repeated calculation must not duplicate daily rows."""
    seed_intraday_flows(repo)

    repo.calculate_eod_group_flows()
    first = repo.get_all_eod_group_flows()

    repo.calculate_eod_group_flows()
    second = repo.get_all_eod_group_flows()

    pd.testing.assert_frame_equal(first, second)


def test_rebuild_includes_new_data(repo):
    """A rebuild should incorporate newly persisted intraday records."""
    seed_intraday_flows(repo)
    repo.calculate_eod_group_flows()

    repo.connection.execute(
        """
        INSERT INTO l1_group_flows
        VALUES (
            1, 2, 10, 103, 'Noise', 'Maker',
            2, 0, 200, 0
        )
        """
    )

    repo.calculate_eod_group_flows()
    result = repo.get_all_eod_group_flows()

    assert len(result) == 4

    row = result.set_index(["day", "asset_id", "agent_group", "cp_group"]).loc[
        (1, 10, "Noise", "Maker")
    ]

    assert row["volume_buy"] == pytest.approx(10)
    assert row["cash_buy"] == pytest.approx(1000)


def test_empty_daily_aggregation(repo):
    """No intraday data should produce an empty daily table."""
    repo.calculate_eod_group_flows()

    result = repo.get_all_eod_group_flows()

    assert result.empty
    assert list(result.columns) == COLUMNS


def test_snowflake_daily_table_schema():
    """Snowflake must define the same analytical columns."""
    assert list(TABLES["l1_eod_group_flows"]) == COLUMNS
