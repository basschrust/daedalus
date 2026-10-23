import re
import sys
from pathlib import Path

import duckdb


def quote_identifier(name: str) -> str:
    """Safely quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def inspect_database(db_path: Path) -> None:
    with duckdb.connect(str(db_path), read_only=True) as conn:
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
            table = quote_identifier(table_name)

            columns = {
                row[0]
                for row in conn.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'main'
                      AND table_name = ?
                """,
                    [table_name],
                ).fetchall()
            }

            total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            print(f"\n{'=' * 60}")
            print(f"Table: {table_name}")
            print(f"Total rows: {total:,}")

            # Prefer grouping by day when both fields are present.
            group_column = (
                "day" if "day" in columns else "agent_id" if "agent_id" in columns else None
            )

            if group_column is None:
                continue

            column = quote_identifier(group_column)

            counts = conn.execute(f"""
                SELECT {column}, COUNT(*) AS row_count
                FROM {table}
                GROUP BY {column}
                ORDER BY {column} NULLS LAST
            """).fetchall()

            print(f"\nRows per {group_column}:")

            if not counts:
                print("  No rows.")
                continue

            for group_value, count in counts:
                print(f"  {group_value!s:>12}: {count:>10,}")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "marketsim" / "output"

    pattern = re.compile(r"^run_(\d{8}_\d{6})$")

    if not output_dir.is_dir():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    run_dirs = [path for path in output_dir.iterdir() if path.is_dir() and pattern.match(path.name)]

    if not run_dirs:
        print(f"No simulation run directories found in {output_dir}")
        sys.exit(1)

    latest_run = max(run_dirs, key=lambda path: path.name)
    db_path = latest_run / "daedalus.duckdb"

    if not db_path.is_file():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    print(f"Latest run: {latest_run.name}")
    print(f"Database:   {db_path}")

    inspect_database(db_path)


if __name__ == "__main__":
    main()
