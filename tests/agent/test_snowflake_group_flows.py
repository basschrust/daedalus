"""Place at tests/database/test_snowflake_group_flows.py.

Requires the optional Snowflake connector package, but not Snowflake credentials.
"""
from unittest.mock import Mock

import pytest

pytest.importorskip("snowflake.connector")

from marketsim.database.connectors.snowflake_storage import SnowflakeRepository, TABLES


def test_l1_table_schema_present():
    assert "l1_group_flows" in TABLES
    assert set(TABLES["l1_group_flows"]) == {
        "day", "time_tick", "asset_id", "agent_id", "agent_group", "cp_group",
        "volume_buy", "volume_sell", "cash_buy", "cash_sell",
    }


def test_calculate_group_flows_executes_transaction_and_joins_both_agents():
    repo = object.__new__(SnowflakeRepository)  # bypass cloud connection
    repo._execute = Mock()
    repo.calculate_group_flows()
    sql = [call.args[0].upper() for call in repo._execute.call_args_list]
    assert sql[0] == "BEGIN TRANSACTION"
    assert sql[1] == "DELETE FROM L1_GROUP_FLOWS"
    assert "INSERT INTO L1_GROUP_FLOWS" in sql[2]
    assert "JOIN AGENTS A" in sql[2]
    assert "JOIN AGENTS CP" in sql[2]
    assert "T_CP.ORDER_ID = T.MATCHED_WITH" in sql[2]
    assert sql[3] == "COMMIT"


def test_calculate_group_flows_rolls_back_on_failure():
    repo = object.__new__(SnowflakeRepository)
    def execute(sql):
        if "INSERT INTO" in sql:
            raise RuntimeError("query failed")
        calls.append(sql)
    calls = []
    repo._execute = execute
    with pytest.raises(RuntimeError, match="query failed"):
        repo.calculate_group_flows()
    assert calls == ["BEGIN TRANSACTION", "DELETE FROM L1_GROUP_FLOWS", "ROLLBACK"]


def test_get_group_flows_uses_parameterized_query():
    repo = object.__new__(SnowflakeRepository)
    repo._read = Mock()
    repo.get_group_flows(17)
    sql, params, columns = repo._read.call_args.args
    assert "WHERE AGENT_ID = %s" in sql
    assert params == (17,)
    assert columns[0] == "day"
