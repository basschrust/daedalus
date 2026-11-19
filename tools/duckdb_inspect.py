"""Inspect a DuckDB database using the database diagnostic implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketsim.database.inspection import inspect_database

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_duckdb.py <database_file>")
        sys.exit(1)
    inspect_database(sys.argv[1])
