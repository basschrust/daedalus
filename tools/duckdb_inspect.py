import sys

import duckdb


def inspect_database(db_path: str) -> None:
    with duckdb.connect(db_path, read_only=True) as conn:
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """).fetchall()

        if not tables:
            print("No tables found.")
            return

        for (table_name,) in tables:
            # Get column names and types
            columns = conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = ?
                ORDER BY ordinal_position
            """, [table_name]).fetchall()

            # Get row count
            row_count = conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]

            print(f"\nTable: {table_name}")
            print(f"Rows:  {row_count}")
            print("Columns:")

            for column_name, data_type in columns:
                print(f"  - {column_name}: {data_type}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_duckdb.py <database_file>")
        sys.exit(1)

    inspect_database(sys.argv[1])
