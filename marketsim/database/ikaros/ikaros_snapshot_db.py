"""DuckDB operations for creating and validating Ikaros snapshots."""

from __future__ import annotations

from pathlib import Path

import duckdb


def checkpoint_ikaros_database(*, database_path: Path) -> None:
    """Flush the source WAL before copying the database file."""
    conn = duckdb.connect(str(database_path), read_only=False)
    try:
        conn.execute("CHECKPOINT")
    finally:
        conn.close()


def validate_ikaros_snapshot(*, database_path: Path) -> None:
    """Verify that the copied database can be opened and queried."""
    conn = duckdb.connect(str(database_path), read_only=True)
    try:
        conn.execute("SELECT 1").fetchone()
    finally:
        conn.close()
