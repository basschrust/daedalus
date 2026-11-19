"""Check that database refactoring keeps the existing portable query boundary."""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from marketsim.database import ikaros_queries, risk_queries


class DuckDBLike:
    """SQLite-backed isolated store to test ANSI SQL and DB boundary without cloud."""

    __module__ = "duckdb"

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")

    def cursor(self):
        return self.db.cursor()

    def execute(self, *args):
        return self.db.execute(*args)

    def executemany(self, *args):
        return self.db.executemany(*args)


class SnowflakeCursor:
    def __init__(self, inner):
        self.inner = inner

    def execute(self, sql, params=()):
        self.inner.execute(sql.replace("%s", "?"), params)
        return self

    def executemany(self, sql, params):
        return self.inner.executemany(sql.replace("%s", "?"), params)

    def fetchall(self):
        return self.inner.fetchall()

    def close(self):
        self.inner.close()


class SnowflakeLike:
    __module__ = "snowflake.connector"

    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")

    def cursor(self):
        return SnowflakeCursor(self.db.cursor())


@pytest.mark.parametrize("connection_type", [DuckDBLike, SnowflakeLike])
def test_daily_calibration_roundtrip_and_idempotency(connection_type):
    connection = connection_type()
    try:
        repo = SimpleNamespace(connection=connection)
        risk_queries.ensure_daily_calibration_tables(repo)
        risk_queries.publish_daily_calibration(
            repo,
            "SIM-1",
            [("SIM-1", "AAA", 0.001, 0.02), ("SIM-1", "BBB", 0.0, 0.03)],
            [("SIM-1", "AAA", "BBB", 0.00012)],
            day=0,
            as_of_date=date(2026, 11, 19),
            first_session=date(2026, 1, 1),
            observations=250,
            lookback_sessions=251,
            source="SIMULATED",
        )
        risk_queries.publish_daily_calibration(
            repo,
            "SIM-1",
            [("SIM-1", "AAA", 0.001, 0.02), ("SIM-1", "BBB", 0.0, 0.03)],
            [("SIM-1", "AAA", "BBB", 0.00012)],
            day=0,
            as_of_date=date(2026, 11, 19),
            first_session=date(2026, 1, 1),
            observations=250,
            lookback_sessions=251,
            source="SIMULATED",
        )
        assert connection.db.execute(
            "SELECT COUNT(*) FROM risk_calibration_history"
        ).fetchone() == (1,)
        assert connection.db.execute(
            "SELECT as_of_date, first_session FROM risk_calibration_history"
        ).fetchone() == ("2026-11-19", "2026-01-01")
        assert connection.db.execute(
            "SELECT COUNT(*) FROM risk_l2_instrument_statistics"
        ).fetchone() == (2,)
        assert connection.db.execute("SELECT COUNT(*) FROM risk_l2_pair_statistics").fetchone() == (
            1,
        )
    finally:
        connection.db.close()


def test_ikaros_query_returns_named_frame():
    class Cursor:
        description = (("SYMBOL",), ("WEIGHT",))
        params = None
        sql = ""

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            self.sql = sql
            self.params = params

        def fetchall(self):
            return [("AAA", 0.7), ("BBB", 0.3)]

    cursor = Cursor()
    connection = SimpleNamespace(cursor=lambda: cursor)
    result = ikaros_queries.fetch_wig20_constituents(connection, date(2026, 11, 19))
    assert list(result.columns) == ["symbol", "weight"]
    assert list(result.symbol) == ["AAA", "BBB"]
    assert cursor.params == (date(2026, 11, 19), date(2026, 11, 19))
    assert "INDEX_CONSTITUENTS" in cursor.sql


def test_no_sql_or_database_cursor_in_application_modules():
    root = Path(__file__).resolve().parents[2]
    forbidden = re.compile(
        r"\b(?:SELECT\s+\w+\s+FROM|INSERT\s+INTO|DELETE\s+FROM|"
        r"CREATE\s+TABLE|ALTER\s+TABLE|MERGE\s+INTO)\b|"
        r"\.cursor\s*\(|\.execute\s*\(|\.executemany\s*\(|"
        r"\bduckdb\.connect\s*\(|\bread_sql\s*\(",
        re.IGNORECASE,
    )
    bad = []
    for folder in ("marketsim", "run"):
        for path in (root / folder).rglob("*.py"):
            if path.is_relative_to(root / "marketsim" / "database"):
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.lstrip().startswith("#") and forbidden.search(line):
                    bad.append(f"{path.relative_to(root)}:{n}")
    for tool in ("duckdb_inspect.py", "last_run_summary.py", "test_snowflake_connection.py"):
        path = root / "tools" / tool
        if path.exists():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if forbidden.search(line):
                    bad.append(f"{path.relative_to(root)}:{n}")
    assert not bad, "SQL outside marketsim/database: " + ", ".join(bad)
