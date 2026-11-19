"""DuckDB inspection and diagnostic SQL used by thin CLI tools."""

from __future__ import annotations

from pathlib import Path

import duckdb


def _tables(conn):
    return conn.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """).fetchall()


def _columns(conn, table_name: str):
    return conn.execute(
        """
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'main' AND table_name = ?
        ORDER BY ordinal_position
    """,
        [table_name],
    ).fetchall()


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_database(db_path: str) -> None:
    """Original tools/duckdb_inspect.py output."""
    with duckdb.connect(db_path, read_only=True) as conn:
        tables = _tables(conn)
        if not tables:
            print("No tables found.")
            return
        for (table_name,) in tables:
            columns = _columns(conn, table_name)
            row_count = conn.execute(
                f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}"
            ).fetchone()[0]
            print(f"\nTable: {table_name}")
            print(f"Rows:  {row_count}")
            print("Columns:")
            for column_name, data_type in columns:
                print(f"  - {column_name}: {data_type}")


def inspect_run_summary(db_path: Path) -> None:
    """Original tools/last_run_summary.py output."""
    with duckdb.connect(str(db_path), read_only=True) as conn:
        tables = _tables(conn)
        if not tables:
            print("No tables found.")
            return
        for (table_name,) in tables:
            table = _quote_identifier(table_name)
            columns = {row[0] for row in _columns(conn, table_name)}
            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n{'=' * 60}")
            print(f"Table: {table_name}")
            print(f"Total rows: {total:,}")
            column_name = (
                "day" if "day" in columns else "agent_id" if "agent_id" in columns else None
            )
            if column_name is None:
                continue
            column = _quote_identifier(column_name)
            counts = conn.execute(f"""
                SELECT {column}, COUNT(*) AS row_count FROM {table}
                GROUP BY {column} ORDER BY {column} NULLS LAST
            """).fetchall()
            print(f"\nRows per {column_name}:")
            if not counts:
                print("  No rows.")
                continue
            for value, count in counts:
                print(f"  {value!s:>12}: {count:>10,}")


def test_snowflake_connection() -> None:
    """Original manual Snowflake connection diagnostic, moved under database."""
    import os
    from getpass import getpass

    import snowflake.connector

    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=getpass("Snowflake password: "),
        warehouse="DAEDALUS_WH",
        database="DAEDALUS",
        schema="SIMULATION",
        enable_server_session_keep_alive_auto_detection=False,
    )
    try:
        with conn.cursor() as cursor:
            cursor.execute("USE DATABASE DAEDALUS")
            cursor.execute("USE SCHEMA SIMULATION")
            cursor.execute("""
                SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE()
            """)
            print(cursor.fetchone())
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT CURRENT_VERSION(), CURRENT_USER(), CURRENT_DATABASE(),
                       CURRENT_SCHEMA(), CURRENT_WAREHOUSE()
            """)
            print(cursor.fetchone())
    finally:
        conn.close()
